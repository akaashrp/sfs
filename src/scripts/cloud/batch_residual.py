"""Audit that each engine's batch-latency model predicts the engine it was actually launched against.

The simulator's batch clock is only as good as the coefficients handed to the server, and a coefficient
fitted for one feature set and consumed against another is silently wrong rather than loudly wrong: the
run completes, the audits pass, and only the wait estimates are nonsense.  That is exactly how the
Ministral ``cross_term`` fit came to be read as ``legacy`` (scripts/cloud/reports/ministral-sfs-gap-20260918),
over-predicting batch time by 107x-257x and leaving SFS with no feasible candidate on 99.9% of decisions.
Every pool therefore replays its own coefficients against the batch statistics its engines just wrote and
refuses to evaluate when the two disagree.
"""
import csv
from statistics import median

# The batch-latency fit's sixth feature, by the feature set the coefficients were fitted under.
FEATURE_COLUMN = {'legacy': 'sum_sq_tokens', 'cross_term': 'prefill_x_processed_ctx_sum'}
# A correct model sits near 1.0; these bounds catch a feature or unit mismatch, not ordinary fit error.
BOUNDS = (0.5, 2.0)
MIN_ROWS = 200


def predicted_seconds(row, coefficients, feature_set):
    """The fitted batch execution time for one batch-statistics row, in seconds."""
    return (float(coefficients['intercept'])
            + float(coefficients['prefill_coeff']) * row['prefill']
            + float(coefficients['prefill_sq_coeff']) * row['prefill_sq_sum']
            + float(coefficients['decode_coeff']) * row['decode']
            + float(coefficients['sum_coeff']) * row['sum_tokens']
            + float(coefficients['sum_sq_coeff']) * row[FEATURE_COLUMN[feature_set]])


def residuals(path, coefficients, feature_set, *, min_rows=MIN_ROWS):
    """Median and p90 predicted/actual batch-time ratio over the rows of one batch-statistics CSV."""
    if feature_set not in FEATURE_COLUMN:
        raise ValueError(f'Unsupported batch-time feature set {feature_set!r}')
    columns = ('prefill', 'prefill_sq_sum', 'decode', 'sum_tokens', FEATURE_COLUMN[feature_set], 'exec')
    ratios, predicted, actual = [], [], []
    with open(path) as handle:
        for raw in csv.DictReader(handle):
            try:
                row = {key: float(raw[key]) for key in columns}
            except (KeyError, TypeError, ValueError):
                continue  # a partially written final row, or a build whose CSV predates a column
            if row['exec'] <= 0:
                continue
            estimate = predicted_seconds(row, coefficients, feature_set)
            predicted.append(estimate)
            actual.append(row['exec'])
            ratios.append(estimate / row['exec'])
    if len(ratios) < min_rows:
        raise ValueError(f'{path} holds {len(ratios)} usable batch rows; {min_rows} are required to audit the fit')
    ordered = sorted(ratios)
    return {'rows': len(ratios), 'feature_set': feature_set,
            'median_ratio': median(ratios), 'p90_ratio': ordered[int(0.9 * (len(ordered) - 1))],
            'median_predicted_ms': median(predicted) * 1000.0, 'median_actual_ms': median(actual) * 1000.0}


def audit_pool(instances, output, *, bounds=BOUNDS, min_rows=MIN_ROWS):
    """Replay every engine's coefficients against its own batch statistics; raise when a model disagrees."""
    low, high = bounds
    report = {'bounds': [low, high], 'min_rows': min_rows, 'engines': {}}
    for row in instances:
        model = row['model_id']
        measured = residuals(f'{output}/batch_stats_{model}.csv', row['ttft_batch_model'],
                             row['batch_time_feature_set'], min_rows=min_rows)
        report['engines'][model] = measured
        if not low <= measured['median_ratio'] <= high:
            raise ValueError(
                f'{model} batch-latency model predicts {measured["median_ratio"]:.3g}x its measured batch time '
                f'over {measured["rows"]} batches ({measured["median_predicted_ms"]:.2f} ms predicted against '
                f'{measured["median_actual_ms"]:.2f} ms measured); the coefficients do not describe this engine')
    return report
