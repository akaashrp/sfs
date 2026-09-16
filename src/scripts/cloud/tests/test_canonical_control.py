from types import SimpleNamespace

import pytest

from scripts.cloud.canonical_control import REQUEST_FIELDS, request_fingerprint


@pytest.mark.parametrize('field', REQUEST_FIELDS)
def test_reference_identity_detects_each_workload_or_slo_change(field):
    row = dict(zip(REQUEST_FIELDS, ('req-0', 'chat', 123, 2000., 3., 159.)))
    expected = request_fingerprint([SimpleNamespace(**row)])
    row[field] = row[field] + ('x' if isinstance(row[field], str) else 1)
    assert request_fingerprint([SimpleNamespace(**row)]) != expected


def test_reference_identity_detects_arrival_order_change():
    rows = [SimpleNamespace(**dict(zip(REQUEST_FIELDS, (f'req-{i}', 'chat', 123, 2000., 3., 159.))))
            for i in range(2)]
    assert request_fingerprint(rows) != request_fingerprint(rows[::-1])
