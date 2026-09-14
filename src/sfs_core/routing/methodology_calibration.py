"""Versioned offline service calibration for the methodology baselines.

The JSON artifact contains a nonnegative singleton-prefill chunk model and a
native XGBoost JSON TPOT head per serving tier. Heavy imports are lazy so pure
routing tests and non-RouteBalance runs do not require XGBoost.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from sfs_core.routing.methodology_policies import _count, _nonnegative, _positive

TPOT_FEATURE_NAMES = ("decode_tokens", "prefill_tokens", "context_tokens")
PREFILL_FEATURE_NAMES = ("intercept", "prefill_tokens", "prefill_tokens_squared", "prefill_x_context")
CALIBRATION_SCHEMA_VERSION = 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serving_profile_sha256(profile: Mapping) -> str:
    return hashlib.sha256(json.dumps(dict(profile), sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class MethodologyCalibration:
    def __init__(self, path: Path, payload: dict) -> None:
        self.path = path
        self.payload = payload
        if payload.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
            raise ValueError("unsupported methodology calibration schema")
        if payload.get("data_role") != "calibration":
            raise ValueError("methodology fit must use calibration data")
        profile = payload.get("serving_profile")
        if not isinstance(profile, dict) or not profile:
            raise ValueError("serving_profile provenance is required")
        if payload.get("serving_profile_sha256") != serving_profile_sha256(profile):
            raise ValueError("serving profile checksum mismatch")
        if payload.get("serving_profile_verified") is not True:
            raise ValueError("calibration manifest must explicitly verify the serving profile")
        self.models = payload.get("models", {})
        if not self.models:
            raise ValueError("calibration artifact has no models")
        self.speeds = {}
        self._heads = {}
        for model, artifact in self.models.items():
            self.speeds[model] = _positive(artifact["service_rate_qps"], f"{model}.service_rate_qps")
            if not artifact.get("service_rate_definition"):
                raise ValueError("service_rate_definition is required")
            prefill = artifact["prefill"]
            if tuple(prefill["feature_names"]) != PREFILL_FEATURE_NAMES:
                raise ValueError("unsupported prefill feature schema")
            if len(prefill["coefficients_ms"]) != len(PREFILL_FEATURE_NAMES):
                raise ValueError("prefill coefficient count mismatch")
            prefill["coefficients_ms"] = [
                _nonnegative(coefficient, "prefill coefficient")
                for coefficient in prefill["coefficients_ms"]]
            if (prefill.get("parameterization") != "intercept_linear_causal_quadratic"
                    or not math.isclose(prefill["coefficients_ms"][3],
                                        2 * prefill["coefficients_ms"][2],
                                        rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError("unsupported prefill parameterization or context constraint")
            prefill["chunk_tokens"] = _count(prefill["chunk_tokens"], "chunk_tokens")
            if not prefill["chunk_tokens"]:
                raise ValueError("prefill chunk_tokens must be positive")
            tpot = artifact["tpot"]
            if tuple(tpot["feature_names"]) != TPOT_FEATURE_NAMES:
                raise ValueError("unsupported TPOT feature schema")
            head_path = (path.parent / tpot["model_file"]).resolve()
            if not head_path.is_relative_to(path.parent.resolve()):
                raise ValueError("TPOT model path must remain inside its artifact directory")
            if file_sha256(head_path) != tpot["model_sha256"]:
                raise ValueError(f"TPOT model checksum mismatch for {model}")

    @classmethod
    def load(cls, path: str | Path) -> "MethodologyCalibration":
        source = Path(path).expanduser().resolve()
        with source.open() as stream:
            return cls(source, json.load(stream))

    def validate_runtime_profile(self, profile: Mapping) -> None:
        """Compare the actual launch manifest with the complete fitted profile.

        Copy ``instance_metadata.serving_profile`` verbatim into the fitting
        manifest. A self-consistent artifact checksum establishes integrity;
        this separate comparison establishes launch-configuration compatibility.
        No missing-field defaults or scalar service-rate substitutions apply.
        """
        if not isinstance(profile, Mapping) or not profile:
            raise ValueError("actual runtime serving_profile is required")
        expected = self.payload["serving_profile"]
        if serving_profile_sha256(profile) != self.payload["serving_profile_sha256"]:
            missing = sorted(set(expected) - set(profile))
            extra = sorted(set(profile) - set(expected))
            changed = sorted(key for key in set(expected) & set(profile)
                             if json.dumps(expected[key], sort_keys=True) !=
                             json.dumps(profile[key], sort_keys=True))
            raise ValueError("methodology calibration serving profile does not match runtime: "
                             f"missing={missing}, extra={extra}, changed={changed}")

    def prefill_ms(self, model: str, prompt_tokens: int,
                   computed_prompt_tokens: int = 0) -> float:
        prompt = _count(prompt_tokens, "prompt_tokens")
        computed = _count(computed_prompt_tokens, "computed_prompt_tokens")
        if computed > prompt:
            raise ValueError("computed_prompt_tokens cannot exceed prompt_tokens")
        fit = self.models[model]["prefill"]
        intercept, linear, quadratic, context_coefficient = fit["coefficients_ms"]
        chunk_limit = fit["chunk_tokens"]
        total = 0.0
        while computed < prompt:
            chunk = min(prompt - computed, chunk_limit)
            total += (intercept + linear * chunk + quadratic * chunk * chunk
                      + context_coefficient * chunk * computed)
            computed += chunk
        return _nonnegative(total, "predicted_prefill_ms")

    def tpot_ms(self, model: str, features: Mapping[str, float]) -> float:
        if set(features) != set(TPOT_FEATURE_NAMES):
            raise ValueError(f"TPOT features must be exactly {TPOT_FEATURE_NAMES!r}")
        import numpy as np
        import xgboost as xgb

        if model not in self._heads:
            head = xgb.XGBRegressor(n_jobs=1)
            head.load_model(self.path.parent / self.models[model]["tpot"]["model_file"])
            self._heads[model] = head
        row = np.array([[_nonnegative(features[name], name) for name in TPOT_FEATURE_NAMES]],
                       dtype=float)
        value = float(self._heads[model].predict(row)[0])
        return _positive(value, "predicted_tpot_ms")

    def preload(self) -> None:
        """Load and warm CPU TPOT heads before measured request arrivals."""
        for model in self.models:
            self.tpot_ms(model, {"decode_tokens": 1.0,
                                 "prefill_tokens": 0.0, "context_tokens": 1.0})

    def describe(self) -> dict:
        return {
            "artifact_path": str(self.path), "artifact_sha256": file_sha256(self.path),
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "serving_profile_sha256": self.payload["serving_profile_sha256"],
            "speeds": self.speeds.copy(),
            "tpot_feature_names": list(TPOT_FEATURE_NAMES),
            "prefill_variant": "nonnegative_singleton_chunk_execution",
            "prefill_parameterization": "intercept_linear_causal_quadratic",
            "tpot_variant": "xgboost_decode_iteration_execution",
        }

    @property
    def metadata(self) -> dict:
        return self.describe()
