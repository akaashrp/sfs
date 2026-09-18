"""The pool refuses to evaluate when an engine's batch-latency coefficients do not describe that engine."""
import pytest

from scripts.cloud.batch_residual import audit_pool, residuals

COEFFICIENTS = {'intercept': 0.004, 'prefill_coeff': 9.2e-06, 'decode_coeff': 7.3e-06,
                'sum_coeff': 3.8e-08, 'prefill_sq_coeff': 3.7e-10, 'sum_sq_coeff': 8.6e-10}
HEADER = 'prefill,prefill_sq_sum,decode,num_seqs,sum_tokens,sum_sq_tokens,prefill_x_processed_ctx_sum,exec\n'


def _trace(path, rows=400, feature_set='cross_term'):
    """A trace whose exec times are exactly what COEFFICIENTS predicts under `feature_set`."""
    lines = [HEADER]
    for i in range(rows):
        prefill, decode = (0, 60) if i % 3 else (2048, 40)
        context = 8000 + 37 * i
        row = {'prefill': prefill, 'prefill_sq_sum': prefill * prefill, 'decode': decode,
               'sum_tokens': context, 'sum_sq_tokens': context * context * 50.0,
               'prefill_x_processed_ctx_sum': prefill * context}
        exec_s = (COEFFICIENTS['intercept'] + COEFFICIENTS['prefill_coeff'] * row['prefill']
                  + COEFFICIENTS['prefill_sq_coeff'] * row['prefill_sq_sum']
                  + COEFFICIENTS['decode_coeff'] * row['decode']
                  + COEFFICIENTS['sum_coeff'] * row['sum_tokens']
                  + COEFFICIENTS['sum_sq_coeff'] * row[{'legacy': 'sum_sq_tokens',
                                                        'cross_term': 'prefill_x_processed_ctx_sum'}[feature_set]])
        lines.append(f"{row['prefill']},{row['prefill_sq_sum']},{row['decode']},8,{row['sum_tokens']},"
                     f"{row['sum_sq_tokens']},{row['prefill_x_processed_ctx_sum']},{exec_s}\n")
    path.write_text(''.join(lines))
    return path


def test_matching_feature_set_passes_and_a_mismatch_is_refused(tmp_path):
    _trace(tmp_path/'batch_stats_m.csv')
    instances = [{'model_id': 'm', 'ttft_batch_model': COEFFICIENTS, 'batch_time_feature_set': 'cross_term'}]
    report = audit_pool(instances, tmp_path)
    assert report['engines']['m']['rows'] == 400
    assert report['engines']['m']['median_ratio'] == pytest.approx(1.0, abs=1e-6)

    # The Ministral defect: the same coefficients read against the sum of squared context lengths.
    mismatched = [dict(instances[0], batch_time_feature_set='legacy')]
    with pytest.raises(ValueError, match='do not describe this engine'):
        audit_pool(mismatched, tmp_path)
    assert residuals(tmp_path/'batch_stats_m.csv', COEFFICIENTS, 'legacy')['median_ratio'] > 50


def test_a_short_or_unreadable_trace_is_an_error_not_a_pass(tmp_path):
    _trace(tmp_path/'batch_stats_m.csv', rows=10)
    with pytest.raises(ValueError, match='usable batch rows'):
        audit_pool([{'model_id': 'm', 'ttft_batch_model': COEFFICIENTS,
                     'batch_time_feature_set': 'cross_term'}], tmp_path)
    with pytest.raises(ValueError, match='Unsupported batch-time feature set'):
        residuals(tmp_path/'batch_stats_m.csv', COEFFICIENTS, 'quadratic')
    with pytest.raises(FileNotFoundError):
        audit_pool([{'model_id': 'absent', 'ttft_batch_model': COEFFICIENTS,
                     'batch_time_feature_set': 'legacy'}], tmp_path)
