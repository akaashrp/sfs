"""RouteBalance's CPU MiniLM + distance-weighted FAISS quality/length head.

Paper: https://arxiv.org/html/2606.17949v1#S4.SS2
The paper specifies MiniLM, FAISS, k=10 and distance weighting, but not the
distance kernel or tokenizer limit. We use normalized embeddings, Euclidean
distance (sqrt of FAISS squared L2), inverse distance, and MiniLM's published
256-wordpiece right truncation. Exact neighbors share all weight uniformly.
These choices are versioned in every artifact. No SFS predictor is used.

Only JSON and numeric .npy arrays are deserialized. The FAISS index is rebuilt
from checksum-verified embeddings, avoiding pickle/native index deserialization.
Heavy dependencies are lazy so importing other routing policies stays cheap.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Sequence


ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SCHEMA_VERSION = "routebalance_minilm_faiss_v1"
DEFAULT_MAX_COMPLETION_TOKENS = 8192
ARRAY_FILES = ("embeddings.npy", "qualities.npy", "output_lengths.npy")
MANIFEST_FILES = ("train_prompts.json", "validation_prompts.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def encoder_metadata() -> dict[str, Any]:
    return {
        "model_id": ENCODER_ID,
        "revision": ENCODER_REVISION,
        "device": "cpu",
        "implementation": "transformers_masked_mean_pool",
        "pooling": "attention_masked_mean",
        "normalize_embeddings": True,
        "max_sequence_length": 256,
        "truncation_side": "right",
        "dtype": "float32",
        "dimension": 384,
        "source": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    }


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("numpy", "torch", "transformers", "faiss-cpu"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _normalized_matrix(values: Any, *, label: str) -> Any:
    import numpy as np

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError(f"{label} must be a nonempty 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} contains nonfinite values")
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError(f"{label} contains a zero or invalid embedding")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class MiniLMCPUEmbedder:
    """The model-card implementation of MiniLM's sentence embedding pipeline.

    Loading is explicit and CPU-only. Runtime defaults to local_files_only so
    a benchmark never starts downloading a model on its timed request path.
    ``encode`` returns one normalized float32 matrix for a scheduling batch;
    internal microbatches bound memory for offline artifact building.
    """

    def __init__(
        self,
        metadata: dict[str, Any] | None = None,
        *,
        cache_dir: str | Path | None = None,
        local_files_only: bool = True,
        cpu_threads: int = 2,
    ) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.metadata = dict(metadata or encoder_metadata())
        validate_encoder_metadata(self.metadata)
        torch.set_num_threads(_positive_int(cpu_threads, "cpu_threads"))
        common = {
            "revision": self.metadata["revision"],
            "cache_dir": str(cache_dir) if cache_dir is not None else None,
            "local_files_only": local_files_only,
            "trust_remote_code": False,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(self.metadata["model_id"], **common)
        self.tokenizer.truncation_side = "right"
        self.model = AutoModel.from_pretrained(
            self.metadata["model_id"], use_safetensors=True, **common
        ).to(device="cpu", dtype=torch.float32)
        resolved_commit = getattr(self.model.config, "_commit_hash", None)
        if resolved_commit != self.metadata["revision"]:
            raise ValueError(f"Loaded MiniLM commit {resolved_commit!r} differs from artifact pin")
        weights_path = Path(hf_hub_download(
            self.metadata["model_id"], "model.safetensors",
            revision=self.metadata["revision"], cache_dir=common["cache_dir"],
            local_files_only=True,
        ))
        weights_checksum = file_sha256(weights_path)
        expected_checksum = self.metadata.get("weights_sha256")
        if expected_checksum is not None and weights_checksum != expected_checksum:
            raise ValueError("Cached MiniLM weights differ from the artifact checksum")
        self.metadata["weights_sha256"] = weights_checksum
        self.model.eval()

    def encode(self, prompts: Sequence[str], *, batch_size: int = 32) -> Any:
        import numpy as np
        import torch
        import torch.nn.functional as functional

        _positive_int(batch_size, "batch_size")
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Every prompt must be text")
        if not prompts:
            return np.empty((0, self.metadata["dimension"]), dtype=np.float32)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(prompts), batch_size):
                encoded = self.tokenizer(
                    list(prompts[start : start + batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.metadata["max_sequence_length"],
                    return_tensors="pt",
                )
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(dtype=hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                chunks.append(functional.normalize(pooled, p=2, dim=1).cpu().numpy())
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.float32)


def validate_encoder_metadata(metadata: dict[str, Any]) -> None:
    expected = encoder_metadata()
    if not re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("revision", ""))):
        raise ValueError("MiniLM encoder revision must be an immutable 40-character commit")
    for key in (
        "model_id", "device", "implementation", "pooling", "normalize_embeddings",
        "max_sequence_length", "truncation_side", "dtype", "dimension",
    ):
        if metadata.get(key) != expected[key]:
            raise ValueError(f"Unsupported MiniLM encoder metadata {key}={metadata.get(key)!r}")


def _faiss_index(dimension: int, cpu_threads: int) -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError(
            "RouteBalance requires faiss-cpu; install it in the dedicated baseline "
            "dependency target and add that target to PYTHONPATH."
        ) from exc
    faiss.omp_set_num_threads(cpu_threads)
    return faiss.IndexFlatL2(dimension)


class RouteBalancePredictor:
    """One batched lookup returns quality and output length for every model.

    ``predict_batch`` returns ``[{model: {quality: float, output_tokens: float}}]``
    in input request order. The caller must time the entire call (including CPU
    embedding) on the request path. Injected embedders/index factories exist for
    tests; production loading never falls back to an alternate estimator.
    """

    def __init__(
        self,
        *,
        embeddings: Any,
        qualities: Any,
        output_lengths: Any,
        model_labels: Sequence[str],
        embedder: Any,
        k: int = 10,
        distance_epsilon: float = 1e-6,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        embedding_batch_size: int = 32,
        cpu_threads: int = 2,
        metadata: dict[str, Any] | None = None,
        index_factory: Callable[[int], Any] | None = None,
    ) -> None:
        import numpy as np

        self.model_labels = tuple(model_labels)
        if (not self.model_labels or len(set(self.model_labels)) != len(self.model_labels)
                or any(not isinstance(label, str) or not label for label in self.model_labels)):
            raise ValueError("model_labels must be distinct nonempty strings")
        self.k = _positive_int(k, "k")
        self.max_completion_tokens = _positive_int(max_completion_tokens, "max_completion_tokens")
        self.embedding_batch_size = _positive_int(embedding_batch_size, "embedding_batch_size")
        self.cpu_threads = _positive_int(cpu_threads, "cpu_threads")
        self.distance_epsilon = float(distance_epsilon)
        if not math.isfinite(self.distance_epsilon) or self.distance_epsilon <= 0:
            raise ValueError("distance_epsilon must be finite and positive")
        self.embeddings = _normalized_matrix(embeddings, label="embeddings")
        self.qualities = np.asarray(qualities, dtype=np.float32)
        self.output_lengths = np.asarray(output_lengths, dtype=np.float32)
        shape = (len(self.embeddings), len(self.model_labels))
        if self.qualities.shape != shape or self.output_lengths.shape != shape:
            raise ValueError("Quality and length arrays must match prompts x model_labels")
        if not np.isfinite(self.qualities).all() or ((self.qualities < 0) | (self.qualities > 1)).any():
            raise ValueError("Quality labels must be finite and within [0,1]")
        if (not np.isfinite(self.output_lengths).all()
                or ((self.output_lengths < 0) | (self.output_lengths > self.max_completion_tokens)).any()):
            raise ValueError("Output lengths must be finite and within the completion cap")
        self.embedder = embedder
        self.metadata = dict(metadata or {})
        self._native_faiss = index_factory is None
        self.index = (index_factory(self.embeddings.shape[1]) if index_factory is not None
                      else _faiss_index(self.embeddings.shape[1], cpu_threads))
        self.index.add(self.embeddings)

    def predict_batch(
        self,
        prompts: Sequence[str],
        completion_caps: Sequence[int] | int | None = None,
    ) -> list[dict[str, dict[str, float]]]:
        if any(not isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Every prompt must be text")
        if not prompts:
            return []
        embeddings = self.embedder.encode(prompts, batch_size=self.embedding_batch_size)
        return self.predict_embeddings(embeddings, completion_caps=completion_caps)

    def predict_embeddings(
        self, embeddings: Any, completion_caps: Sequence[int] | int | None = None
    ) -> list[dict[str, dict[str, float]]]:
        """Artifact-validation helper; serving must use ``predict_batch``."""
        import numpy as np

        query = _normalized_matrix(embeddings, label="query embeddings")
        if query.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension differs from artifact")
        if completion_caps is None:
            caps = [self.max_completion_tokens] * len(query)
        elif isinstance(completion_caps, int):
            caps = [completion_caps] * len(query)
        else:
            caps = list(completion_caps)
        if len(caps) != len(query):
            raise ValueError("completion_caps must have one value per prompt")
        for cap in caps:
            _positive_int(cap, "completion cap")
            if cap > self.max_completion_tokens:
                raise ValueError("Request completion cap exceeds artifact calibration cap")
        if self._native_faiss:
            import faiss

            # OpenMP settings are thread-local: the scheduler calls this method
            # through asyncio.to_thread, which may differ from the loader thread.
            faiss.omp_set_num_threads(self.cpu_threads)
        squared, neighbors = self.index.search(query, min(self.k, len(self.embeddings)))
        squared = np.asarray(squared, dtype=np.float64)
        neighbors = np.asarray(neighbors)
        expected_shape = (len(query), min(self.k, len(self.embeddings)))
        if squared.shape != expected_shape or neighbors.shape != expected_shape:
            raise ValueError("FAISS returned malformed neighbor arrays")
        if (not np.isfinite(squared).all() or not np.issubdtype(neighbors.dtype, np.integer)
                or ((neighbors < 0) | (neighbors >= len(self.embeddings))).any()):
            raise ValueError("FAISS returned invalid neighbors or distances")
        # IndexFlatL2 may round exact dot-product matches slightly below zero.
        if (squared < -1e-5).any():
            raise ValueError("FAISS returned a negative squared distance")
        # Recompute selected distances in float64: FAISS's BLAS path may give
        # positive squared roundoff for an identical float32 query, which would
        # otherwise defeat the exact-match rule differently across batch sizes.
        delta = (query[:, None, :].astype(np.float64)
                 - self.embeddings[neighbors].astype(np.float64))
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        exact = distances <= self.distance_epsilon
        weights = 1.0 / np.maximum(distances, self.distance_epsilon)
        has_exact = exact.any(axis=1)
        weights[has_exact] = exact[has_exact].astype(np.float64)
        weights /= weights.sum(axis=1, keepdims=True)
        quality = np.einsum("nk,nkm->nm", weights, self.qualities[neighbors])
        length = np.einsum("nk,nkm->nm", weights, self.output_lengths[neighbors])
        # Convex averaging is bounded mathematically; clipping only removes FP
        # roundoff and applies the incoming request's explicit generation cap.
        quality = np.clip(quality, 0, 1)
        length = np.minimum(np.maximum(length, 0), np.asarray(caps)[:, None])
        return [
            {model: {"quality": float(quality[row, col]), "output_tokens": float(length[row, col])}
             for col, model in enumerate(self.model_labels)}
            for row in range(len(query))
        ]

    @classmethod
    def load(
        cls,
        artifact_dir: str | Path,
        *,
        embedder: Any = None,
        cache_dir: str | Path | None = None,
        cpu_threads: int = 2,
        index_factory: Callable[[int], Any] | None = None,
    ) -> "RouteBalancePredictor":
        import numpy as np

        root = Path(artifact_dir)
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported RouteBalance predictor artifact schema")
        metadata["runtime_cpu_threads"] = _positive_int(cpu_threads, "cpu_threads")
        validate_encoder_metadata(metadata["encoder"])
        if metadata.get("distance") != "euclidean_on_l2_normalized_embeddings":
            raise ValueError("Unsupported RouteBalance distance metric")
        if metadata.get("weighting") != "inverse_distance_exact_matches_only":
            raise ValueError("Unsupported RouteBalance neighbor weighting")
        checksums = metadata.get("file_sha256", {})
        for name in (*ARRAY_FILES, *MANIFEST_FILES):
            if file_sha256(root / name) != checksums.get(name):
                raise ValueError(f"RouteBalance artifact checksum mismatch: {name}")
        manifests = [json.loads((root / name).read_text(encoding="utf-8")) for name in MANIFEST_FILES]
        train, validation = manifests
        _validate_prompt_manifests(train, validation)
        arrays = [np.load(root / name, allow_pickle=False) for name in ARRAY_FILES]
        if len(train) != len(arrays[0]) or len(train) != metadata.get("num_train_prompts"):
            raise ValueError("Training prompt manifest does not match embedding rows")
        if len(validation) != metadata.get("num_validation_prompts"):
            raise ValueError("Validation prompt manifest length mismatch")
        if arrays[0].ndim != 2 or arrays[0].shape[1] != metadata["encoder"]["dimension"]:
            raise ValueError("Embedding dimension differs from MiniLM encoder metadata")
        if embedder is None:
            embedder = MiniLMCPUEmbedder(
                metadata["encoder"], cache_dir=cache_dir or metadata.get("encoder_cache_dir"),
                local_files_only=True, cpu_threads=cpu_threads,
            )
        return cls(
            embeddings=arrays[0], qualities=arrays[1], output_lengths=arrays[2],
            model_labels=metadata["model_labels"], embedder=embedder,
            k=metadata["k"], distance_epsilon=metadata["distance_epsilon"],
            max_completion_tokens=metadata["max_completion_tokens"],
            embedding_batch_size=metadata["embedding_batch_size"],
            cpu_threads=cpu_threads, metadata=metadata, index_factory=index_factory,
        )


def _validate_prompt_manifests(train: list[dict], validation: list[dict]) -> None:
    ids: set[tuple[str, str]] = set()
    for rows in (train, validation):
        for row in rows:
            index = row.get("prompt_index")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 2500:
                raise ValueError("Prompt manifest index is outside the [0,2500) calibration boundary")
            key = (row["bucket"], row["example_id"])
            if key in ids:
                raise ValueError("Duplicate or overlapping training/validation prompt IDs")
            ids.add(key)
            if not re.fullmatch(r"[0-9a-f]{64}", row["prompt_sha256"]):
                raise ValueError("Invalid prompt checksum in split manifest")
    if {row["prompt_sha256"] for row in train} & {row["prompt_sha256"] for row in validation}:
        raise ValueError("Training and validation contain identical prompt text")


def save_predictor_artifact(
    output_dir: str | Path,
    predictor: RouteBalancePredictor,
    *,
    train_prompts: list[dict],
    validation_prompts: list[dict],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write a new immutable artifact directory; never overwrite old artifacts."""
    import numpy as np

    root = Path(output_dir)
    _validate_prompt_manifests(train_prompts, validation_prompts)
    if len(train_prompts) != len(predictor.embeddings):
        raise ValueError("Training manifest must align with embedding rows")
    encoder = dict(provenance.get("encoder", encoder_metadata()))
    validate_encoder_metadata(encoder)
    if predictor.embeddings.shape[1] != encoder["dimension"]:
        raise ValueError("Cannot save embeddings incompatible with the MiniLM encoder")
    root.mkdir(parents=True, exist_ok=False)
    for name, value in zip(ARRAY_FILES, (predictor.embeddings, predictor.qualities, predictor.output_lengths)):
        np.save(root / name, value, allow_pickle=False)
    for name, values in zip(MANIFEST_FILES, (train_prompts, validation_prompts)):
        (root / name).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    metadata = {
        **provenance,
        "schema_version": SCHEMA_VERSION,
        "encoder": encoder,
        "model_labels": list(predictor.model_labels),
        "num_train_prompts": len(train_prompts),
        "num_validation_prompts": len(validation_prompts),
        "k": predictor.k,
        "distance": "euclidean_on_l2_normalized_embeddings",
        "weighting": "inverse_distance_exact_matches_only",
        "distance_epsilon": predictor.distance_epsilon,
        "exact_match_scope": "neighbors returned by k-nearest lookup; equal weights, other neighbors zero",
        "max_completion_tokens": predictor.max_completion_tokens,
        "embedding_batch_size": predictor.embedding_batch_size,
        "dependency_versions": dependency_versions(),
        "file_sha256": {name: file_sha256(root / name) for name in (*ARRAY_FILES, *MANIFEST_FILES)},
    }
    # metadata.json is the completion marker, written only after data/checksums.
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata
