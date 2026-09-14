from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sfs_core.routing.routebalance_predictor import (
    ENCODER_REVISION,
    RouteBalancePredictor,
    encoder_metadata,
    prompt_sha256,
    save_predictor_artifact,
    validate_encoder_metadata,
)


class NumpyTestIndex:
    """A test oracle for FAISS's API, never a production estimator fallback."""

    def __init__(self, dimension):
        self.dimension = dimension

    def add(self, embeddings):
        self.embeddings = embeddings

    def search(self, queries, k):
        squared = ((queries[:, None, :] - self.embeddings[None, :, :]) ** 2).sum(axis=2)
        neighbors = np.argsort(squared, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(squared, neighbors, axis=1), neighbors


class Embedder:
    def __init__(self, mappings):
        self.mappings = mappings
        self.calls = []

    def encode(self, prompts, *, batch_size):
        self.calls.append((list(prompts), batch_size))
        return np.asarray([self.mappings[prompt] for prompt in prompts], dtype=np.float32)


def make_predictor(**changes):
    kwargs = {
        "embeddings": [[1, 0], [0, 1]],
        "qualities": [[1, 0], [0, 1]],
        "output_lengths": [[10, 20], [30, 40]],
        "model_labels": ["small", "large"],
        "embedder": Embedder({"first": [1, 0], "second": [0, 1], "middle": [1, 1]}),
        "index_factory": NumpyTestIndex,
        "k": 10,
    }
    kwargs.update(changes)
    return RouteBalancePredictor(**kwargs)


def test_batch_mapping_exact_matches_and_request_caps():
    predictor = make_predictor()
    predictions = predictor.predict_batch(["second", "first"], [25, 15])
    assert predictions == [
        {"small": {"quality": 0, "output_tokens": 25}, "large": {"quality": 1, "output_tokens": 25}},
        {"small": {"quality": 1, "output_tokens": 10}, "large": {"quality": 0, "output_tokens": 15}},
    ]
    assert predictor.embedder.calls == [(["second", "first"], 32)]


def test_inverse_euclidean_distance_and_normalization():
    # Query has distances sqrt(.4) and sqrt(.8) to the two unit basis vectors.
    predictor = make_predictor()
    value = predictor.predict_embeddings([[4, 3]])[0]
    first_weight = (1 / np.sqrt(.4)) / (1 / np.sqrt(.4) + 1 / np.sqrt(.8))
    assert value["small"]["quality"] == pytest.approx(first_weight)
    assert value["small"]["output_tokens"] == pytest.approx(10 * first_weight + 30 * (1 - first_weight))
    assert predictor.predict_embeddings([[40, 30]]) == value_as_batch(value)


def value_as_batch(value):
    return [value]


def test_exact_duplicate_neighbors_share_weight_and_ignore_others():
    predictor = make_predictor(
        embeddings=[[1, 0], [1, 0], [0, 1]],
        qualities=[[1, .1], [0, .9], [1, 1]],
        output_lengths=[[10, 20], [30, 40], [8000, 8000]],
    )
    result = predictor.predict_batch(["first"])[0]
    assert result["small"] == {"quality": .5, "output_tokens": 20}
    assert result["large"]["quality"] == pytest.approx(.5)
    assert result["large"]["output_tokens"] == 30


def test_exact_match_rule_survives_faiss_squared_distance_roundoff():
    class RoundedIndex(NumpyTestIndex):
        def search(self, queries, k):
            distances, neighbors = super().search(queries, k)
            return distances + 1e-7, neighbors

    predictor = make_predictor(index_factory=RoundedIndex)
    assert predictor.predict_batch(["first"])[0]["small"]["quality"] == 1


@pytest.mark.parametrize("changes", [
    {"qualities": [[float("nan"), 0], [0, 1]]},
    {"qualities": [[1.1, 0], [0, 1]]},
    {"qualities": [[-.1, 0], [0, 1]]},
    {"output_lengths": [[8193, 0], [0, 1]]},
    {"output_lengths": [[-1, 0], [0, 1]]},
    {"output_lengths": [[float("inf"), 0], [0, 1]]},
    {"embeddings": [[0, 0], [0, 1]]},
    {"embeddings": [[float("nan"), 0], [0, 1]]},
    {"model_labels": ["small", "small"]},
    {"qualities": [[1], [0]]},
    {"k": 0},
    {"k": True},
    {"distance_epsilon": float("inf")},
])
def test_reject_invalid_training_state(changes):
    with pytest.raises(ValueError):
        make_predictor(**changes)


@pytest.mark.parametrize("caps", [0, -1, 8193, True, [1, 2], [float("nan")]])
def test_invalid_completion_caps(caps):
    with pytest.raises(ValueError):
        make_predictor().predict_batch(["first"], caps)


def test_empty_batch_does_not_embed_and_dimension_rejected():
    predictor = make_predictor()
    assert predictor.predict_batch([]) == []
    assert not predictor.embedder.calls
    with pytest.raises(ValueError, match="dimension"):
        predictor.predict_embeddings([[1, 2, 3]])


def padded(embeddings):
    values = np.zeros((len(embeddings), 384), dtype=np.float32)
    values[:, :2] = embeddings
    return values


def save_fixture(tmp_path):
    predictor = make_predictor(
        embeddings=padded([[1, 0], [0, 1]]),
        embedder=Embedder({"first": padded([[1, 0]])[0]}),
    )
    root = tmp_path / "artifact"
    train = [
        {"bucket": "a", "example_id": f"a:train:{i}", "prompt_index": i, "prompt_sha256": prompt_sha256(f"train{i}")}
        for i in range(2)
    ]
    validation = [{"bucket": "a", "example_id": "a:train:2", "prompt_index": 2, "prompt_sha256": prompt_sha256("validation")}]
    save_predictor_artifact(root, predictor, train_prompts=train, validation_prompts=validation, provenance={})
    return root, predictor


def test_artifact_roundtrip_has_pin_and_verifies_all_files(tmp_path):
    root, predictor = save_fixture(tmp_path)
    loaded = RouteBalancePredictor.load(root, embedder=predictor.embedder, index_factory=NumpyTestIndex)
    assert loaded.predict_batch(["first"]) == predictor.predict_batch(["first"])
    assert loaded.metadata["encoder"]["revision"] == ENCODER_REVISION
    assert loaded.metadata["encoder"]["max_sequence_length"] == 256
    assert loaded.metadata["num_train_prompts"] == 2
    with (root / "qualities.npy").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        RouteBalancePredictor.load(root, embedder=predictor.embedder, index_factory=NumpyTestIndex)


def test_artifact_rejects_overlap_and_overwrite(tmp_path):
    root, predictor = save_fixture(tmp_path)
    train = json.loads((root / "train_prompts.json").read_text())
    validation = json.loads((root / "validation_prompts.json").read_text())
    with pytest.raises(FileExistsError):
        save_predictor_artifact(root, predictor, train_prompts=train, validation_prompts=validation, provenance={})
    validation[0]["prompt_sha256"] = train[0]["prompt_sha256"]
    with pytest.raises(ValueError, match="identical prompt text"):
        save_predictor_artifact(tmp_path / "invalid", predictor, train_prompts=train, validation_prompts=validation, provenance={})


def test_loader_uses_recorded_encoder_cache_and_explicit_override(tmp_path, monkeypatch):
    import sfs_core.routing.routebalance_predictor as module

    root, predictor = save_fixture(tmp_path)
    metadata = json.loads((root / "metadata.json").read_text())
    metadata["encoder_cache_dir"] = "/saved/project/cache"
    (root / "metadata.json").write_text(json.dumps(metadata))
    calls = []
    def fake_embedder(metadata, **kwargs):
        calls.append(kwargs)
        return predictor.embedder
    monkeypatch.setattr(module, "MiniLMCPUEmbedder", fake_embedder)
    RouteBalancePredictor.load(root, index_factory=NumpyTestIndex)
    assert calls[-1]["cache_dir"] == "/saved/project/cache"
    assert calls[-1]["local_files_only"] is True
    RouteBalancePredictor.load(root, cache_dir="/explicit/cache", index_factory=NumpyTestIndex)
    assert calls[-1]["cache_dir"] == "/explicit/cache"


@pytest.mark.parametrize("changes", [
    {"revision": "main"}, {"device": "cuda"}, {"max_sequence_length": 512},
    {"pooling": "cls"}, {"normalize_embeddings": False},
])
def test_encoder_contract_cannot_silently_drift(changes):
    with pytest.raises(ValueError):
        validate_encoder_metadata({**encoder_metadata(), **changes})


def test_native_faiss_matches_test_oracle_when_installed():
    pytest.importorskip("faiss")
    oracle = make_predictor()
    native = make_predictor(index_factory=None)
    prompts = ["first", "middle", "second"] * 10
    assert native.predict_batch(prompts) == oracle.predict_batch(prompts)


def test_native_faiss_thread_limit_applies_in_scheduler_worker():
    from concurrent.futures import ThreadPoolExecutor

    faiss = pytest.importorskip("faiss")
    predictor = make_predictor(index_factory=None, cpu_threads=2)
    def worker():
        faiss.omp_set_num_threads(8)
        result = predictor.predict_batch(["first"])
        return result, faiss.omp_get_max_threads()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result, threads = pool.submit(worker).result()
    assert result[0]["small"]["quality"] == 1
    assert threads == 2
