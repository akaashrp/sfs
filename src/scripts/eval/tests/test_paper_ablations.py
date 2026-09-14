from copy import deepcopy
import json

import pytest

from scripts.eval.judge_ablation import paired_summary
from scripts.prep.paper_ablation_data import MODELS, indexed


def judged(group, score):
    return [{"bucket": "alpaca", "model_label": model, "prompt": f"prompt {group}",
        "prompt_metadata": {"example_id": f"alpaca:{group}"},
        "response": {"output_text": model}, "quality": score, "quality_metric": "judge",
        "judge_alias_to_model": dict(zip("ABC", MODELS))} for model in MODELS]


def test_query_disagreement_pairs_ids_not_file_order():
    left = judged(1, .1)+judged(2, .8)
    right = list(reversed(judged(1, .4)+judged(2, .6)))
    result, pairs = paired_summary(left, right)
    assert result["complete_queries"] == 2 and len(pairs) == 6
    assert result["mean_per_query_absolute_disagreement"] == pytest.approx(.25)


def test_imputation_excludes_entire_query_from_query_average():
    left, right = judged(1, .1)+judged(2, .8), judged(1, .4)+judged(2, .6)
    right[0]["quality_metric"] = "judge_default_bucket_mean"
    result, _ = paired_summary(left, right)
    assert result["complete_queries"] == 1 and result["excluded_queries"] == 1
    assert result["mean_per_query_absolute_disagreement"] == pytest.approx(.2)


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.append(deepcopy(r[0])), "Duplicate"),
    (lambda r: r.pop(), "matching canonical"),
    (lambda r: r[0]["response"].update(output_text="different"), "changed"),
    (lambda r: r[0].pop("judge_alias_to_model"), "alias mapping"),
    (lambda r: r[0].update(quality=float("nan")), "Invalid"),
])
def test_judge_audit_rejects_invalid_comparison(mutation, match):
    left, right = judged(1, .1), judged(1, .4)
    mutation(right)
    with pytest.raises(ValueError, match=match):
        paired_summary(left, right)


def test_generation_local_index_is_not_canonical_identity(tmp_path):
    path = tmp_path/"scores.jsonl"
    record = judged(2500, .1)[0]
    record.update(prompt_index=0)
    record["response"]["completion_tokens"] = 1
    path.write_text(json.dumps(record)+"\n")
    assert set(indexed(path, model=MODELS[0])) == {("alpaca", "alpaca:2500")}


def test_scalar_error_metrics_have_correct_units():
    from scripts.eval.estimator_ablation import errors
    result = errors([0, 2], [1, 0])
    assert result == {"n": 2, "mae": 1.5, "rmse": pytest.approx(2.5**.5)}
    with pytest.raises(ValueError, match="nonfinite"):
        errors([1], [float("inf")])


@pytest.mark.parametrize("groups,model_scores,per_bucket", [(8000,24000,2000), (16000,24000,4000)])
def test_prepared_audit_rejects_reduced_or_mislabeled_holdout(tmp_path, groups, model_scores, per_bucket):
    from scripts.prep.paper_ablation_data import validate
    (tmp_path/"data_audit.json").write_text(json.dumps({"status":"PASS", "models":list(MODELS),
        "holdout_prompt_groups":groups, "calibration_prompt_groups":10000,
        "counts":{"holdout_model_scores":model_scores,"calibration_model_scores":30000},
        "canonical_cache_profile":{"holdout_start_index":2500,"holdout_prompts_per_bucket":per_bucket}}))
    with pytest.raises(ValueError,match="Invalid prepared data audit"):
        validate(tmp_path)
