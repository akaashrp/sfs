"""Validate actual producer telemetry, including successful-row TTFT values."""
import math


def require_complete_ttft(run):
    rows = run['per_request']
    missing = sum(isinstance(r.get('system_entry_e2e_ttft_ms'), bool)
                  or not isinstance(r.get('system_entry_e2e_ttft_ms'), (float, int))
                  or not math.isfinite(r['system_entry_e2e_ttft_ms'])
                  or r['system_entry_e2e_ttft_ms'] < 0 for r in rows)
    summary_missing = run['summary'].get('system_entry_e2e_ttft_slo_missing_count')
    if type(summary_missing) is not int or summary_missing != missing or missing:
        raise ValueError('Incomplete or inconsistent end-to-end TTFT coverage')
