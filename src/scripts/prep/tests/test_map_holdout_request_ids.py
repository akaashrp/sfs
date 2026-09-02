from pathlib import Path

from scripts.prep.map_holdout_request_ids import build_jobs


def test_build_jobs_accepts_family_specific_cache_paths():
    qps_cache = Path("/workspace/ministral/holdout_cache_4000")
    delta_cache = Path("/workspace/ministral/holdout_cache_2000")

    jobs = build_jobs(
        "all",
        qps_cache_dir=qps_cache,
        delta_cache_dir=delta_cache,
    )

    assert [job["cache_dir"] for job in jobs] == [qps_cache, delta_cache]
    assert [job["num_requests"] for job in jobs] == [16000, 8000]
