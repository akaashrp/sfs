from __future__ import annotations

import csv
import json

import pytest

from scripts.prep.fit_methodology_calibration import fit_manifest, fit_prefill, load_trace_rows
from sfs_core.routing.methodology_calibration import MethodologyCalibration


def synthetic_rows():
    rows = []
    for index in range(80):
        p = 16 + (index % 7) * 16
        ctx = (index % 5) * 50
        ms = 2 + .01*p + .0001*p*p + .0002*p*ctx
        rows.append(dict(prefill=p, prefill_sq_sum=p*p, decode=0, num_seqs=1,
                         sum_tokens=p+ctx, prefill_x_processed_ctx_sum=p*ctx,
                         exec=ms/1000, trace_index=0, row_index=index))
    for index in range(80):
        decode = 1 + index % 16
        p = (index % 3) * 50
        ctx = 128 * decode + p
        ms = 3 + .5*decode + .001*p + .00001*ctx
        rows.append(dict(prefill=p, prefill_sq_sum=p*p, decode=decode,
                         num_seqs=decode+(p > 0), sum_tokens=ctx,
                         prefill_x_processed_ctx_sum=0, exec=ms/1000,
                         trace_index=0, row_index=index+80))
    return rows


def write_manifest(tmp_path, *, verified=True):
    trace = tmp_path / "synthetic_batch.csv"
    rows = synthetic_rows()
    columns = [name for name in rows[0] if name not in ("trace_index", "row_index")]
    with trace.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "data_role": "calibration", "serving_profile_verified": verified,
        "serving_profile": {"max_num_batched_tokens": 128, "profile": "synthetic_cpu_test"},
        "models": {"a": {"service_rate_qps": 2.5,
                           "service_rate_definition": "synthetic_test_only",
                           "traces": [trace.name]}},
    }))
    return manifest, trace


def test_prefill_fit_excludes_mixed_rows_and_identifies_partial_chunk_context():
    fit = fit_prefill(synthetic_rows(), chunk_tokens=128)
    assert fit["coefficients_ms"] == pytest.approx([2, .01, .0001, .0002], rel=1e-7)
    assert fit["coverage"]["fit_rows"] == 80
    assert fit["coverage"]["partial_prefill_rows"] == 64
    assert fit["coverage"]["heldout"]["mae_ms"] < 1e-7


def test_full_prefill_only_identifies_tied_context_model_and_reports_coverage():
    rows = synthetic_rows()
    for row in rows:
        if row["decode"] == 0:
            p = row["prefill"]
            row["exec"] = (2 + .01*p + .0001*p*p) / 1000
        row["prefill_x_processed_ctx_sum"] = 0
    fit = fit_prefill(rows, chunk_tokens=128)
    assert fit["coefficients_ms"] == pytest.approx([2, .01, .0001, .0002], rel=1e-7)
    assert fit["coverage"]["partial_prefill_observed"] is False
    assert fit["parameterization"] == "intercept_linear_causal_quadratic"
    for row in rows:
        if row["decode"] == 0:
            row["prefill"] = 16
            row["prefill_sq_sum"] = 256
    with pytest.raises(ValueError, match="cannot identify"):
        fit_prefill(rows, chunk_tokens=128)
    with pytest.raises(ValueError, match="singleton"):
        fit_prefill(rows[:2], chunk_tokens=128)


def test_missing_profile_provenance_prevents_fitting_and_output_creation(tmp_path):
    manifest, _ = write_manifest(tmp_path, verified=False)
    with pytest.raises(ValueError, match="verify"):
        fit_manifest(manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_native_artifact_roundtrip_predicts_tpot_and_remaining_prefill(tmp_path):
    pytest.importorskip("xgboost")
    manifest, trace = write_manifest(tmp_path)
    original = trace.read_bytes()
    path = fit_manifest(manifest, tmp_path / "output", n_estimators=8)
    calibration = MethodologyCalibration.load(path)
    assert calibration.speeds == {"a": 2.5}
    calibration.preload()
    assert set(calibration._heads) == {"a"}
    calibration.validate_runtime_profile(
        {"max_num_batched_tokens": 128, "profile": "synthetic_cpu_test"})
    with pytest.raises(ValueError, match="does not match runtime"):
        calibration.validate_runtime_profile(
            {"max_num_batched_tokens": 64, "profile": "synthetic_cpu_test"})
    with pytest.raises(ValueError, match="actual runtime"):
        calibration.validate_runtime_profile({})
    # 200 remaining prompt tokens split into 128 + 72 at declared profile cap.
    expected = (2 + .01*128 + .0001*128**2
                + 2 + .01*72 + .0001*72**2 + .0002*72*128)
    assert calibration.prefill_ms("a", 200) == pytest.approx(expected)
    assert calibration.prefill_ms("a", 200, 128) == pytest.approx(
        2 + .01*72 + .0001*72**2 + .0002*72*128)
    assert calibration.prefill_ms("a", 200, 200) == 0
    assert calibration.tpot_ms("a", {"decode_tokens": 4, "prefill_tokens": 0, "context_tokens": 512}) > 0
    assert trace.read_bytes() == original
    with pytest.raises(ValueError, match="already exists"):
        fit_manifest(manifest, tmp_path / "output", n_estimators=8)
    with pytest.raises(ValueError, match="exactly"):
        calibration.tpot_ms("a", {"decode_batch_size": 4})
    payload = json.loads(path.read_text())
    payload["serving_profile"]["max_num_batched_tokens"] = 256
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        MethodologyCalibration.load(path)


def test_trace_loader_records_invalid_rows_and_input_checksums(tmp_path):
    _, trace = write_manifest(tmp_path)
    with trace.open("a") as stream:
        stream.write("bad,0,0,1,0,0,0.01\n")
    rows, provenance = load_trace_rows([trace])
    assert len(rows) == 160
    assert provenance[0]["invalid_rows"] == 1
    assert len(provenance[0]["sha256"]) == 64
