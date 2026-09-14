"""Quality evaluation utilities for offline LLM benchmarking."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import math
import os
import random
import re
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple
from rouge_score import rouge_scorer
from google import genai
from sfs_core.paths import BUCKETED_OUTPUTS_ROOT

DEFAULT_ROUGE = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
DEFAULT_JUDGE_MODEL = os.environ.get("QUALITY_JUDGE_MODEL", "gemini-3.1-pro-preview")
PROGRESS_LOG_INTERVAL = int(os.environ.get("QUALITY_PROGRESS_INTERVAL", "1000"))
GEMINI_CLIENT = None

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator.\n"
    "Evaluate the MODEL RESPONSE against the REFERENCE RESPONSE for the given PROMPT.\n"
    "Focus on correctness and completeness. Ignore style, tone, and phrasing unless they affect meaning, accuracy, or instruction-following.\n"
    "Penalize contradictions, hallucinations, unsupported claims, and missing key points.\n"
    "Use the full 0 to 10 scale and assign scores granularly across the range to capture real quality differences.\n"
    "Do not bunch most responses into a narrow band.\n"
    "Use decimals when useful.\n\n"
    "Score anchors:\n"
    "0 = completely wrong, irrelevant, or unusable\n"
    "2 = mostly wrong with very little useful content\n"
    "4 = partially correct but with major mistakes or omissions\n"
    "6 = substantially correct but missing important details or containing minor errors\n"
    "8 = strong, correct, and mostly complete with small gaps only\n"
    "10 = fully correct, complete, and faithful to the prompt and reference\n\n"
    "Choose any value from 0 to 10 (including decimal values such as 5.3 or 6.9) based on where the response falls between these anchors.\n\n"
    "Return ONLY valid JSON with exactly these keys:\n"
    '  "score": float from 0 to 10\n'
    '  "explanation": short string of at most 30 words\n'
    "Do not return any other text.\n"
)

JUDGE_GROUP_SYSTEM_PROMPT = (
    "You are a strict evaluator.\n"
    "Evaluate each candidate response against the REFERENCE RESPONSE for the given PROMPT.\n"
    "Multiple candidate responses are provided for the same prompt. Score each one independently, "
    "but use the other candidates as comparison context to better calibrate quality differences.\n"
    "Focus on correctness and completeness. Ignore style, tone, and phrasing unless they affect meaning, "
    "accuracy, or instruction-following.\n"
    "Penalize contradictions, hallucinations, unsupported claims, and missing key points.\n"
    "Use the full 0 to 10 scale and assign scores granularly across the range to capture meaningful "
    "differences in quality.\n"
    "Do not bunch most responses into a narrow band.\n"
    "Use decimals when useful.\n\n"
    "Scoring anchors:\n"
    "0 = completely wrong, irrelevant, or unusable\n"
    "2 = mostly wrong with very little useful content\n"
    "4 = partially correct but with major mistakes or omissions\n"
    "6 = substantially correct but missing important details or containing minor errors\n"
    "8 = strong, correct, and mostly complete with only small gaps\n"
    "10 = fully correct, complete, and faithful to the prompt and reference\n\n"
    "Choose any value from 0 to 10, including decimal values, based on where each response falls between "
    "these anchors.\n"
    "Scores should reflect absolute quality while preserving real differences between candidates answering "
    "the same prompt.\n"
    "If two responses are extremely similar in quality, their scores may be close or equal.\n"
    "If one response is clearly better or worse, reflect that in the scores.\n\n"
    "Return ONLY valid JSON.\n"
    "The JSON must contain exactly one top-level key, \"scores\".\n"
    "The \"scores\" object must contain exactly these keys: \"A\", \"B\", and \"C\".\n"
    "Use this exact structure:\n"
    "{\n"
    '  "scores": {\n'
    '    "A": number from 0 to 10,\n'
    '    "B": number from 0 to 10,\n'
    '    "C": number from 0 to 10\n'
    "  }\n"
    "}\n"
    "DO NOT return any other text since this is for an evaluation task."
)

@dataclass
class ExampleRecord:
    """Lightweight mirror of the generation metadata."""

    source: str
    dataset_id: str
    split: str
    example_id: str
    prompt: str
    ref_output: str
    prompt_tokens: int
    ref_output_tokens: int


@dataclass(frozen=True)
class GroupedJudgeResult:
    """Scores and the randomized candidate aliases used for one judge call."""

    scores_by_model: dict[str, float]
    alias_to_model: dict[str, str]


class JudgeScoreParseError(ValueError):
    """Raised when the judge response does not contain a usable score."""


def _get_gemini_client():
    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY to use the Gemini judge."
        )
    global GEMINI_CLIENT
    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT
    GEMINI_CLIENT = genai.Client()
    return GEMINI_CLIENT


def _extract_numeric_score(text: str) -> float:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "score" in data:
            return float(data["score"])
    except Exception:
        pass
    match = re.search(r"(-?\d+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1))
    raise JudgeScoreParseError(f"Unable to parse judge score from response: {text}")


ARTICLES = {"a", "an", "the"}
WHITESPACE_RE = re.compile(r"\s+")
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_text(text: str) -> str:
    """Lowercase, drop punctuation/articles, and squish whitespace."""
    text = text.lower()
    text = text.translate(PUNCT_TABLE)
    tokens = [
        token
        for token in WHITESPACE_RE.split(text)
        if token and token not in ARTICLES
    ]
    return " ".join(tokens)


def compute_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    """Token-level F1, taking the best score across gold answers."""
    if not gold_answers:
        return 0.0
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_text(gold).split()
        if not gold_tokens:
            continue
        common = 0
        gold_counts = {}
        for token in gold_tokens:
            gold_counts[token] = gold_counts.get(token, 0) + 1
        for token in pred_tokens:
            if gold_counts.get(token, 0) > 0:
                common += 1
                gold_counts[token] -= 1
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best = max(best, f1)
    return best


def compute_rouge(prediction: str, gold: str) -> float:
    """Return ROUGE-Lsum * 100 (to mimic common reporting style)."""
    scores = DEFAULT_ROUGE.score(gold, prediction)
    return scores["rougeLsum"].fmeasure * 100.0


def judge_score(prompt: str, prediction: str, gold: Optional[str], example_id: Optional[str] = None) -> float:
    """Call Gemini to score a response."""
    logger.info("Judging example_id=%s", example_id or "N/A")

    client = _get_gemini_client()
    gold = gold or "N/A"
    judge_input = (
        "<PROMPT>\n"
        f"{prompt}\n"
        "</PROMPT>\n\n"
        "<MODEL_RESPONSE>\n"
        f"{prediction}\n"
        "</MODEL_RESPONSE>\n\n"
        "<REFERENCE_RESPONSE>\n"
        f"{gold}\n"
        "</REFERENCE_RESPONSE>\n"
    )

    response = client.models.generate_content(
        model=DEFAULT_JUDGE_MODEL,
        config={"system_instruction": JUDGE_SYSTEM_PROMPT},
        contents=[{"role": "user", "parts": [{"text": judge_input}]}],
    )
    try:
        content = response.candidates[0].content.parts[0].text if response.candidates else ""
    except Exception as exc:
        raise JudgeScoreParseError("Error retrieving judge response content: %s" % exc) from exc
    score = _extract_numeric_score(content)
    return max(0.0, min(10.0, score))


def judge_scores_for_prompt_group(
    *,
    prompt: str,
    gold: Optional[str],
    example_id: Optional[str],
    candidates_by_model: dict[str, str],
) -> GroupedJudgeResult:
    """Return raw 0-10 scores and the realized aliases for one prompt group."""

    client = _get_gemini_client()

    shuffled_candidates = list(candidates_by_model.items())
    random.shuffle(shuffled_candidates)

    alias_to_model: dict[str, str] = {}
    alias_sections: list[str] = []
    for idx, (model_name, response_text) in enumerate(shuffled_candidates):
        alias = chr(ord("A") + idx)
        alias_to_model[alias] = model_name
        alias_sections.append(
            f"<CANDIDATE_{alias}>\n{response_text}\n</CANDIDATE_{alias}>"
        )
    logger.info(
        "Group-judging example_id=%s alias_to_model=%s",
        example_id or "N/A",
        json.dumps(alias_to_model, sort_keys=True),
    )

    judge_input = (
        "<PROMPT>\n"
        f"{prompt}\n"
        "</PROMPT>\n\n"
        "<REFERENCE_RESPONSE>\n"
        f"{gold}\n"
        "</REFERENCE_RESPONSE>\n\n"
        "<CANDIDATE_RESPONSES>\n"
        f"{chr(10).join(alias_sections)}\n"
        "</CANDIDATE_RESPONSES>\n"
    )

    response = client.models.generate_content(
        model=DEFAULT_JUDGE_MODEL,
        config={"system_instruction": JUDGE_GROUP_SYSTEM_PROMPT},
        contents=[
            {"role": "user", "parts": [{"text": judge_input}]},
        ],
    )
    try:
        content = response.candidates[0].content.parts[0].text if response.candidates else ""
    except Exception as exc:
        raise JudgeScoreParseError("Error retrieving grouped judge response content: %s" % exc) from exc

    content = content.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, flags=re.S)
    if m:
        content = m.group(1)
    else:
        start = content.find("{")
        if start == -1:
            raise JudgeScoreParseError(f"No JSON object found: {content}")
        content = content[start:]

    try:
        parsed = json.JSONDecoder().raw_decode(content)[0]
    except Exception as exc:
        raise JudgeScoreParseError(f"Unable to parse grouped judge JSON response: {content}") from exc
    if not isinstance(parsed, dict) or not isinstance(parsed.get("scores"), dict):
        raise JudgeScoreParseError(f"Grouped judge response missing scores dictionary: {content}")

    alias_scores = {str(alias): float(score) for alias, score in parsed["scores"].items()}
    missing_aliases = [alias for alias in alias_to_model if alias not in alias_scores]
    if missing_aliases:
        raise JudgeScoreParseError(
            f"Grouped judge response missing aliases {missing_aliases}: {content}"
        )

    return GroupedJudgeResult(
        scores_by_model={
            model_name: max(0.0, min(10.0, alias_scores[alias]))
            for alias, model_name in alias_to_model.items()
        },
        alias_to_model=dict(alias_to_model),
    )


QA_DATASETS = {"natural_questions", "hotpotqa/hotpot_qa"}
INSTRUCTION_DATASETS = {"tatsu-lab/alpaca", "openassistant/oasst1"}
SUMMARIZATION_DATASETS = {
    "cnn_dailymail",
    "ccdv/govreport-summarization",
    "ccdv/arxiv-summarization",
    "ccdv/pubmed-summarization",
}


def _dataset_category(dataset_id: str) -> str:
    ds = dataset_id.lower()
    if any(key in ds for key in QA_DATASETS):
        return "qa"
    if any(key in ds for key in INSTRUCTION_DATASETS):
        return "instruction"
    if any(key in ds for key in SUMMARIZATION_DATASETS):
        return "summarization"
    return "instruction"


def _bucket_override(bucket: Optional[str]) -> Optional[str]:
    if not bucket:
        return None
    bucket = bucket.lower()
    overrides = {
        "alpaca": "judge",
        "writingprompts": "judge",
        "govreport-summarization": "judge",
        "hotpot_qa": "judge",
    }
    return overrides.get(bucket)


def _select_metric(dataset_id: str, bucket: Optional[str]) -> str:
    override = _bucket_override(bucket)
    if override is not None:
        return override
    category = _dataset_category(dataset_id)
    return {
        "qa": "f1",
        "instruction": "judge",
        "summarization": "rouge",
    }[category]


def _example_from_metadata(prompt: str, metadata: dict) -> ExampleRecord:
    return ExampleRecord(
        source=str(metadata.get("source", "")),
        dataset_id=str(metadata.get("dataset_id", "")),
        split=str(metadata.get("split", "")),
        example_id=str(metadata.get("example_id", "")),
        prompt=prompt,
        ref_output=str(metadata.get("ref_output", "")),
        prompt_tokens=int(metadata.get("prompt_tokens") or 0),
        ref_output_tokens=int(metadata.get("ref_output_tokens") or 0),
    )


def _collect_gold_answers(metadata: dict, fallback: str) -> list[str]:
    golds = metadata.get("ref_outputs")
    if isinstance(golds, str):
        candidates = [golds]
    elif isinstance(golds, Iterable):
        candidates = [str(g) for g in golds if g]
    else:
        candidates = []
    if not candidates and fallback:
        candidates = [fallback]
    return candidates


def compute_quality(
    example: ExampleRecord,
    prediction: str,
    *,
    gold_answers: Optional[Sequence[str]] = None,
    bucket: Optional[str] = None,
) -> float:
    """
    Compute normalized quality score A(x, m) in [0, 1].

    Args:
        example: Metadata describing the prompt/gold.
        prediction: Model output to score.
        gold_answers: Optional list of references (defaults to example.ref_output).
        bucket: Optional bucket label to enforce metric policy.
    """

    if gold_answers is None:
        gold_answers = [example.ref_output] if example.ref_output else []

    metric = _select_metric(example.dataset_id, bucket)

    if metric == "f1":
        return compute_f1(prediction, gold_answers)

    if metric == "rouge":
        rouge_value = compute_rouge(prediction, gold_answers[0] if gold_answers else "")
        return rouge_value / 100.0

    raw_score = judge_score(
        example.prompt, prediction, gold_answers[0] if gold_answers else None, example.example_id
    )
    return max(0.0, min(1.0, raw_score / 10.0))


def evaluate_generated_record(
    record: dict,
) -> Tuple[float, str]:
    """
    Compute the normalized quality score for a generated output record.

    Returns:
        (score, metric_name)
    """

    metadata = record.get("prompt_metadata") or {}
    prompt = record.get("prompt", "")
    example = _example_from_metadata(prompt, metadata)
    prediction = (record.get("response") or {}).get("output_text", "")
    bucket = record.get("bucket") or metadata.get("bucket")
    golds = _collect_gold_answers(metadata, example.ref_output)
    metric = _select_metric(example.dataset_id, bucket)
    score = compute_quality(
        example,
        prediction,
        gold_answers=golds,
        bucket=bucket,
    )
    return score, metric


def annotate_jsonl_with_quality(
    jsonl_path: str | Path,
    *,
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Read a generated-output JSONL file, compute quality per record, and persist it.

    Args:
        jsonl_path: Path to the original generated outputs (one JSON per line).
        output_path: Optional destination path. If None, writes alongside the source
            using the ``*_scored.jsonl`` suffix.
    """

    src_path = Path(jsonl_path)
    if output_path is None:
        output_path = src_path.with_name(f"{src_path.stem}_scored{src_path.suffix}")
    dst_path = Path(output_path)
    
    existing_prompt_records = set()
    if dst_path.exists():
        for existing_line_number, line in enumerate(dst_path.open("r", encoding="utf-8"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed existing scored record in %s line %d: %s",
                    dst_path,
                    existing_line_number,
                    exc,
                )
                continue
            prompt_index = int(record.get("prompt_index"))
            existing_prompt_records.add(prompt_index)
    else:
        with dst_path.open("w", encoding="utf-8") as f:
            print("Creating new scored output file at %s" % dst_path)
    
    print("Path: %s, num existing_prompt_records: %d" % (jsonl_path, len(existing_prompt_records)))

    logger.info("Scoring %s -> %s", src_path, dst_path)
    processed = 0
    skipped = 0
    reused = 0
    last_logged_at = time.time()
    with src_path.open("r", encoding="utf-8") as src, dst_path.open(
        "a", encoding="utf-8"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt_index = int(record.get("prompt_index"))
            if prompt_index in existing_prompt_records:
                reused += 1
                continue
            metadata = record.get("prompt_metadata") or {}
            bucket = record.get("bucket") or metadata.get("bucket")
            metric = _select_metric(str(metadata.get("dataset_id", "")), bucket)
            try:
                score, metric = evaluate_generated_record(record)
            except JudgeScoreParseError as exc:
                skipped += 1
                logger.warning(
                    "Skipping %s line %d: %s",
                    src_path.name,
                    line_number,
                    exc,
                )
                continue
            record["quality"] = score
            record["quality_metric"] = metric
            dst.write(json.dumps(record, ensure_ascii=False))
            dst.write("\n")
            processed += 1
            if PROGRESS_LOG_INTERVAL and processed % PROGRESS_LOG_INTERVAL == 0:
                now = time.time()
                logger.info(
                    "  %s: processed %d records (last metric=%s, +%0.1fs)",
                    src_path.name,
                    processed,
                    metric,
                    now - last_logged_at,
                )
                last_logged_at = now

    logger.info(
        "Completed %s -> %s (%d scored, %d reused, %d skipped)",
        src_path,
        dst_path,
        processed,
        reused,
        skipped,
    )


def _record_alignment_key(record: dict) -> str:
    """Build a stable join key across model outputs for the same prompt."""
    metadata = record.get("prompt_metadata") or {}
    return str(metadata["example_id"])


def _alignment_key_sort_key(alignment_key: str) -> int:
    """Sort by numeric suffix of example_id."""
    return int(alignment_key.rsplit(":", 1)[-1])

def _alignment_key_to_str(alignment_key: str) -> str:
    return alignment_key


def _load_records_by_alignment_key(src_path: Path, model_name: str) -> dict[str, dict]:
    records_by_key: dict[str, dict] = {}
    with src_path.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed record in %s (%s) line %d: %s",
                    src_path,
                    model_name,
                    line_number,
                    exc,
                )
                continue

            alignment_key = _record_alignment_key(record)

            if alignment_key in records_by_key:
                logger.warning(
                    "Duplicate key %s in %s (%s) line %d; keeping first occurrence.",
                    _alignment_key_to_str(alignment_key),
                    src_path,
                    model_name,
                    line_number,
                )
                continue
            records_by_key[alignment_key] = record
    return records_by_key


def _load_existing_scored_keys(dst_path: Path) -> set[str]:
    existing_keys: set[str] = set()
    if not dst_path.exists():
        return existing_keys

    with dst_path.open("r", encoding="utf-8") as dst:
        for line_number, line in enumerate(dst, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed existing scored record in %s line %d: %s",
                    dst_path,
                    line_number,
                    exc,
                )
                continue

            existing_keys.add(_record_alignment_key(record))
    return existing_keys


def _load_existing_judged_quality_mean(dst_path: Path) -> tuple[float, int]:
    """Return the mean of non-imputed judge scores already stored in a file."""
    values: list[float] = []
    with dst_path.open("r", encoding="utf-8") as dst:
        for line_number, line in enumerate(dst, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Skipping malformed existing scored record in %s line %d: %s",
                    dst_path,
                    line_number,
                    exc,
                )
                continue

            if record.get("quality_imputed") is True:
                continue
            if record.get("quality_metric") == "judge_default_bucket_mean":
                continue

            quality = record.get("quality")
            if isinstance(quality, bool) or not isinstance(quality, (int, float)):
                continue
            quality = float(quality)
            if math.isfinite(quality):
                values.append(quality)

    if not values:
        raise RuntimeError(
            "Cannot impute failed judge requests because no successful judge scores "
            f"exist in {dst_path}."
        )
    return sum(values) / len(values), len(values)


def sort_scored_jsonl_file(scored_jsonl_path: str | Path) -> None:
    """Sort an existing *_scored.jsonl file in-place by example_id numeric suffix."""
    scored_path = Path(scored_jsonl_path)
    records: list[dict] = []

    with scored_path.open("r", encoding="utf-8") as src:
        for line in src:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    records.sort(key=lambda record: _alignment_key_sort_key(_record_alignment_key(record)))

    with scored_path.open("w", encoding="utf-8") as dst:
        for record in records:
            dst.write(json.dumps(record, ensure_ascii=False))
            dst.write("\n")


def annotate_bucket_group_with_quality(
    jsonl_paths_by_model: dict[str, Path],
    output_paths_by_model: Optional[dict[str, Path]],
    *,
    judge_concurrency: int = 20,
    judge_retries: int = 3,
    individual_retries: int = 3,
    retry_sleep_s: float = 10.0,
) -> dict[str, int]:
    """Score one bucket across models using alignment keys instead of file order."""
    if judge_concurrency < 1:
        raise ValueError("judge_concurrency must be at least 1")
    if judge_retries < 1:
        raise ValueError("judge_retries must be at least 1")
    if individual_retries < 1:
        raise ValueError("individual_retries must be at least 1")
    if retry_sleep_s < 0:
        raise ValueError("retry_sleep_s must be non-negative")

    if not jsonl_paths_by_model or not output_paths_by_model:
        logger.warning("No model JSONL paths or output paths provided for grouped bucket scoring.")
        return {
            "total_groups": 0,
            "pending_groups": 0,
            "batch_scored_groups": 0,
            "individual_fallback_groups": 0,
            "individual_recovered_groups": 0,
            "imputed_groups": 0,
            "judged_records_written": 0,
            "imputed_records_written": 0,
            "reused_records": 0,
        }

    model_names = sorted(jsonl_paths_by_model.keys())
    records_by_model: dict[str, dict[str, dict]] = {}
    for model_name in model_names:
        src_path = jsonl_paths_by_model[model_name]
        records_by_key = _load_records_by_alignment_key(src_path, model_name)
        records_by_model[model_name] = records_by_key
        logger.info(
            "Loaded %d alignable records for %s from %s",
            len(records_by_key),
            model_name,
            src_path,
        )

    key_sets = [set(records_by_model[model_name].keys()) for model_name in model_names]
    if not key_sets:
        logger.warning("No model records loaded for grouped bucket scoring.")
        return {
            "total_groups": 0,
            "pending_groups": 0,
            "batch_scored_groups": 0,
            "individual_fallback_groups": 0,
            "individual_recovered_groups": 0,
            "imputed_groups": 0,
            "judged_records_written": 0,
            "imputed_records_written": 0,
            "reused_records": 0,
        }

    common_keys = set.intersection(*key_sets)
    if len(common_keys) != len(key_sets[0]):
        logger.warning(
            "Only %d/%d records have common alignment keys across models for grouped bucket scoring.",
            len(common_keys),
            len(key_sets[0]),
        )

    existing_keys_by_model = {
        model_name: _load_existing_scored_keys(output_paths_by_model[model_name])
        for model_name in model_names
    }

    keys_to_score = [
        key
        for key in common_keys
        if any(key not in existing_keys_by_model[model_name] for model_name in model_names)
    ]
    if not keys_to_score:
        logger.info("All common keys already scored across selected models.")
        return {
            "total_groups": len(common_keys),
            "pending_groups": 0,
            "batch_scored_groups": 0,
            "individual_fallback_groups": 0,
            "individual_recovered_groups": 0,
            "imputed_groups": 0,
            "judged_records_written": 0,
            "imputed_records_written": 0,
            "reused_records": sum(
                len(existing_keys_by_model[model_name]) for model_name in model_names
            ),
        }

    keys_to_score.sort(key=_alignment_key_sort_key)

    group_jobs: list[dict[str, object]] = []
    for alignment_key in keys_to_score:
        anchor_record = records_by_model[model_names[0]][alignment_key]
        metadata = anchor_record.get("prompt_metadata") or {}

        prompt = str(anchor_record.get("prompt") or "")
        gold = str(metadata["ref_output"])
        example_id = str(metadata["example_id"])

        candidates_by_model = {
            model_name: str(
                (records_by_model[model_name][alignment_key].get("response") or {}).get(
                    "output_text", ""
                )
            )
            for model_name in model_names
        }

        group_jobs.append(
            {
                "alignment_key": alignment_key,
                "prompt": prompt,
                "gold": gold,
                "example_id": example_id,
                "candidates_by_model": candidates_by_model,
            }
        )

    reused_records = 0
    judged_records_written = 0
    imputed_records_written = 0
    batch_scored_groups = 0
    individual_recovered_groups = 0

    outputs = {}
    for model_name in model_names:
        dst_path = output_paths_by_model[model_name]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if not dst_path.exists():
            with dst_path.open("w", encoding="utf-8"):
                logger.info("Creating new scored output file at %s", dst_path)
        outputs[model_name] = dst_path.open("a", encoding="utf-8")

    def _write_group_scores(
        alignment_key: str,
        grouped_result: GroupedJudgeResult,
    ) -> None:
        nonlocal reused_records
        nonlocal judged_records_written

        grouped_scores = grouped_result.scores_by_model
        missing_models = [
            model_name for model_name in model_names if model_name not in grouped_scores
        ]
        if missing_models:
            raise JudgeScoreParseError(
                "Grouped judge response missing model scores for "
                f"{_alignment_key_to_str(alignment_key)}: {missing_models}"
            )

        for model_name in model_names:
            if alignment_key in existing_keys_by_model[model_name]:
                reused_records += 1
                continue
            record = dict(records_by_model[model_name][alignment_key])
            raw_score = float(grouped_scores[model_name])
            record["quality"] = max(0.0, min(1.0, raw_score / 10.0))
            record["quality_metric"] = "judge"
            record["judge_alias_to_model"] = dict(grouped_result.alias_to_model)
            outputs[model_name].write(json.dumps(record, ensure_ascii=False))
            outputs[model_name].write("\n")
            existing_keys_by_model[model_name].add(alignment_key)
            judged_records_written += 1

    def _flush_outputs() -> None:
        for handle in outputs.values():
            handle.flush()

    def _score_group_job(job: dict[str, object]) -> GroupedJudgeResult:
        return judge_scores_for_prompt_group(
            prompt=str(job["prompt"]),
            gold=str(job["gold"]),
            example_id=str(job["example_id"]),
            candidates_by_model=dict(job["candidates_by_model"]),
        )

    failed_group_jobs: list[dict[str, object]] = []

    try:
        pending_attempt_jobs = list(group_jobs)
        for attempt in range(1, judge_retries + 1):
            retry_jobs: list[dict[str, object]] = []
            attempt_concurrency = max(
                1, judge_concurrency // (2 ** (attempt - 1))
            )
            worker_count = min(attempt_concurrency, len(pending_attempt_jobs))
            logger.info(
                "Judge retry wave %d/%d: %d prompt groups with concurrency %d.",
                attempt,
                judge_retries,
                len(pending_attempt_jobs),
                worker_count,
            )
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures_to_jobs = {
                    executor.submit(_score_group_job, job): job
                    for job in pending_attempt_jobs
                }
                for future in as_completed(futures_to_jobs):
                    job = futures_to_jobs[future]
                    alignment_key = str(job["alignment_key"])
                    try:
                        grouped_scores = future.result()
                    except Exception as exc:
                        logger.warning(
                            "Error scoring %s in retry wave %d/%d: %s",
                            _alignment_key_to_str(alignment_key),
                            attempt,
                            judge_retries,
                            exc,
                        )
                        retry_jobs.append(job)
                        continue

                    try:
                        _write_group_scores(alignment_key, grouped_scores)
                        batch_scored_groups += 1
                        _flush_outputs()
                    except Exception as exc:
                        logger.warning(
                            "Error storing score for %s in retry wave %d/%d: %s",
                            _alignment_key_to_str(alignment_key),
                            attempt,
                            judge_retries,
                            exc,
                        )
                        retry_jobs.append(job)

            pending_attempt_jobs = retry_jobs
            if not pending_attempt_jobs:
                break
            logger.warning(
                "%d prompt groups remain after retry wave %d/%d.",
                len(pending_attempt_jobs),
                attempt,
                judge_retries,
            )
            if attempt < judge_retries and retry_sleep_s:
                time.sleep(retry_sleep_s * (2 ** (attempt - 1)))

        failed_group_jobs.extend(pending_attempt_jobs)

        if failed_group_jobs:
            logger.warning(
                "%d prompt groups exhausted concurrent attempts; retrying them "
                "individually after all concurrency batches.",
                len(failed_group_jobs),
            )

        unresolved_group_jobs: list[dict[str, object]] = []
        for job_number, job in enumerate(failed_group_jobs, start=1):
            alignment_key = str(job["alignment_key"])
            for attempt in range(1, individual_retries + 1):
                try:
                    grouped_scores = judge_scores_for_prompt_group(
                        prompt=str(job["prompt"]),
                        gold=str(job["gold"]),
                        example_id=str(job["example_id"]),
                        candidates_by_model=dict(job["candidates_by_model"]),
                    )
                    _write_group_scores(alignment_key, grouped_scores)
                    _flush_outputs()
                    individual_recovered_groups += 1
                    break
                except Exception as exc:
                    logger.warning(
                        "Individual fallback failed for %s (%d/%d, attempt %d/%d): %s",
                        _alignment_key_to_str(alignment_key),
                        job_number,
                        len(failed_group_jobs),
                        attempt,
                        individual_retries,
                        exc,
                    )
                    if attempt < individual_retries and retry_sleep_s:
                        time.sleep(retry_sleep_s * (2 ** (attempt - 1)))
            else:
                unresolved_group_jobs.append(job)
    finally:
        for handle in outputs.values():
            handle.close()

    if unresolved_group_jobs:
        quality_means_by_model: dict[str, float] = {}
        for model_name in model_names:
            mean_quality, score_count = _load_existing_judged_quality_mean(
                output_paths_by_model[model_name]
            )
            quality_means_by_model[model_name] = mean_quality
            logger.warning(
                "Using %.12g as the %s bucket-mean fallback from %d successful scores.",
                mean_quality,
                model_name,
                score_count,
            )

        imputation_outputs = {
            model_name: output_paths_by_model[model_name].open("a", encoding="utf-8")
            for model_name in model_names
        }
        try:
            for job in unresolved_group_jobs:
                alignment_key = str(job["alignment_key"])
                for model_name in model_names:
                    if alignment_key in existing_keys_by_model[model_name]:
                        reused_records += 1
                        continue
                    record = dict(records_by_model[model_name][alignment_key])
                    record["quality"] = quality_means_by_model[model_name]
                    record["quality_metric"] = "judge_default_bucket_mean"
                    record["quality_imputed"] = True
                    record["quality_imputed_reason"] = "judge_refusal_or_unavailable"
                    imputation_outputs[model_name].write(
                        json.dumps(record, ensure_ascii=False)
                    )
                    imputation_outputs[model_name].write("\n")
                    existing_keys_by_model[model_name].add(alignment_key)
                    imputed_records_written += 1
        finally:
            for handle in imputation_outputs.values():
                handle.close()

    for model_name in model_names:
        missing_keys = common_keys - existing_keys_by_model[model_name]
        if missing_keys:
            raise RuntimeError(
                f"Judge output for {model_name} remains incomplete: "
                f"{len(missing_keys)} common prompt groups are missing."
            )
        sort_scored_jsonl_file(output_paths_by_model[model_name])

    summary = {
        "total_groups": len(common_keys),
        "pending_groups": len(group_jobs),
        "batch_scored_groups": batch_scored_groups,
        "individual_fallback_groups": len(failed_group_jobs),
        "individual_recovered_groups": individual_recovered_groups,
        "imputed_groups": len(unresolved_group_jobs),
        "judged_records_written": judged_records_written,
        "imputed_records_written": imputed_records_written,
        "reused_records": reused_records,
    }
    logger.info(
        "Completed grouped bucket scoring (%d judged records, %d imputed records, "
        "%d reused records; %d individual recoveries, %d imputed groups)",
        judged_records_written,
        imputed_records_written,
        reused_records,
        individual_recovered_groups,
        len(unresolved_group_jobs),
    )
    return summary


def main() -> None:
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    parser = argparse.ArgumentParser(
        description="Score bucketed prompt generations for model variants."
    )
    default_root = BUCKETED_OUTPUTS_ROOT
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=default_root,
        help="Root directory containing model subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["qwen3-0.6b", "qwen3-8b", "qwen3-32b"],
        help="Names of subdirectories to score.",
    )
    parser.add_argument(
        "--judge-callback",
        type=str,
        default=None,
        help="Optional module:function path for the judge scoring callback.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override the Gemini model used for LLM-as-a-judge scoring.",
    )
    parser.add_argument(
        "--judge-concurrency",
        "--judge-batch-size",
        dest="judge_concurrency",
        type=int,
        default=20,
        help=(
            "Number of prompt-group requests issued concurrently per batch "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--judge-retries",
        type=int,
        default=3,
        help=(
            "Total attempts for each request during concurrent batch processing "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--individual-retries",
        type=int,
        default=3,
        help=(
            "Total attempts for each failed prompt during the final individual "
            "cleanup pass (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=10.0,
        help=(
            "Initial retry delay; subsequent retry delays use exponential backoff "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help=(
            "Write a JSON run summary here (default: "
            "<outputs-root>/judge_run_summary.json)."
        ),
    )
    args = parser.parse_args()

    if args.judge_concurrency < 1:
        parser.error("--judge-concurrency/--judge-batch-size must be at least 1")
    if args.judge_retries < 1:
        parser.error("--judge-retries must be at least 1")
    if args.individual_retries < 1:
        parser.error("--individual-retries must be at least 1")
    if args.retry_sleep_seconds < 0:
        parser.error("--retry-sleep-seconds must be non-negative")

    if args.judge_model:
        global DEFAULT_JUDGE_MODEL
        DEFAULT_JUDGE_MODEL = args.judge_model

    root = args.outputs_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Outputs root not found: {root}")

    model_dirs: dict[str, Path] = {}
    for model_name in args.models:
        model_dir = root / model_name
        if not model_dir.exists():
            print(f"[WARN] Skipping {model_name}, directory not found: {model_dir}")
            continue
        model_dirs[model_name] = model_dir

    if not model_dirs:
        print("[WARN] No valid model directories found under outputs root.")
        return

    bucket_names_by_model: dict[str, set[str]] = {}
    for model_name, model_dir in model_dirs.items():
        bucket_names_by_model[model_name] = {
            jsonl_file.name
            for jsonl_file in model_dir.glob("*.jsonl")
            if not jsonl_file.name.endswith("_scored.jsonl")
        }

    common_bucket_names = set.intersection(*bucket_names_by_model.values())
    if not common_bucket_names:
        print("[WARN] No common bucket JSONL files found across selected models.")
        return

    bucket_summaries: dict[str, dict[str, int]] = {}
    for bucket_name in sorted(common_bucket_names):
        jsonl_paths_by_model = {
            model_name: model_dir / bucket_name
            for model_name, model_dir in model_dirs.items()
        }
        output_paths_by_model = {
            model_name: src_path.with_name(f"{src_path.stem}_scored{src_path.suffix}")
            for model_name, src_path in jsonl_paths_by_model.items()
        }
        print(f"[INFO] Scoring bucket {bucket_name} across models: {', '.join(sorted(jsonl_paths_by_model.keys()))}")
        bucket_summaries[bucket_name] = annotate_bucket_group_with_quality(
            jsonl_paths_by_model,
            output_paths_by_model=output_paths_by_model,
            judge_concurrency=args.judge_concurrency,
            judge_retries=args.judge_retries,
            individual_retries=args.individual_retries,
            retry_sleep_s=args.retry_sleep_seconds,
        )

    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path is not None
        else root / "judge_run_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "outputs_root": str(root),
                "models": sorted(model_dirs),
                "judge_model": DEFAULT_JUDGE_MODEL,
                "judge_batch_size": args.judge_concurrency,
                "judge_retries": args.judge_retries,
                "individual_retries": args.individual_retries,
                "retry_sleep_seconds": args.retry_sleep_seconds,
                "buckets": bucket_summaries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote judge run summary to %s", summary_path)


if __name__ == "__main__":
    main()
