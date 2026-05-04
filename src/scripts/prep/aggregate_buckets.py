#!/usr/bin/env python3
"""
Aggregate prompts from multiple Hugging Face datasets into buckets.

This script is designed to be robust to missing datasets / fields:
- It will try each dataset, skip if unavailable, and continue.
- It uses (prompt_tokens, ref_output_tokens) where ref output exists; otherwise it
  falls back to a heuristic ref output (e.g., answers/summary fields if present).
- Tokenization defaults to a fast local tokenizer if available; otherwise uses a
  cheap whitespace approximation.

Output: JSONL files per dataset + a metadata summary.

Install:
  pip install datasets transformers tqdm

Example:
  python aggregate_buckets.py \
    --out_dir ./experiments/data/prompts/bucketed_prompts/qwen3-0.6b \
    --max_per_dataset 20000 \
    --max_per_bucket 20000 \
    --tokenizer Qwen/Qwen3-0.6B

  python aggregate_buckets.py \
    --out_dir ./experiments/data/prompts/bucketed_prompts/qwen3-8b \
    --max_per_dataset 20000 \
    --max_per_bucket 20000 \
    --tokenizer Qwen/Qwen3-8B
    
  python aggregate_buckets.py \
    --out_dir ./experiments/data/prompts/bucketed_prompts/qwen3-32b \
    --max_per_dataset 20000 \
    --max_per_bucket 20000 \
    --tokenizer Qwen/Qwen3-32B
"""

import argparse
import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from datasets import load_dataset

_TOKENIZER = None
from transformers import AutoTokenizer

def init_tokenizer(tokenizer_name: str) -> None:
    """Initialize a HF tokenizer if possible; otherwise leave as None."""
    global _TOKENIZER
    if tokenizer_name.lower() in ("none", "whitespace", "basic"):
        _TOKENIZER = None
        return
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    except Exception as e:
        print(f"[WARN] Could not load tokenizer '{tokenizer_name}'. Falling back to whitespace. Error: {e}")
        _TOKENIZER = None


def count_tokens(text: str) -> int:
    """Count tokens using tokenizer if available, else approximate with whitespace."""
    if not text:
        return 0
    if _TOKENIZER is None:
        return len(text.split())
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


@dataclass
class ExampleRecord:
    source: str
    dataset_id: str
    split: str
    example_id: str
    prompt: str
    ref_output: str
    prompt_tokens: int
    ref_output_tokens: int


def safe_get(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            if isinstance(d[k], str) and not d[k].strip():
                continue
            return d[k]
    return None


def to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


# ---- Adapter: Alpaca-style (instruction/input/output)
def extract_alpaca(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    instr = safe_get(ex, ["instruction", "prompt", "query"])
    inp = safe_get(ex, ["input", "context"])
    out = safe_get(ex, ["output", "response", "answer"])
    if instr is None and inp is None:
        return None
    prompt = to_str(instr)
    if inp:
        prompt = prompt.strip() + "\n\n" + to_str(inp).strip()
    ref = to_str(out)
    return prompt.strip(), ref.strip()


# ---- Adapter: OpenAssistant / chat logs
def extract_oasst(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    text = safe_get(ex, ["text", "message", "content"])
    role = safe_get(ex, ["role"])
    msgs = safe_get(ex, ["messages", "conversation", "thread"])
    if isinstance(msgs, list) and len(msgs) >= 2:
        user_msg = None
        asst_msg = None
        for m in msgs:
            r = (m.get("role") or m.get("from") or m.get("speaker") or "").lower()
            t = m.get("content") or m.get("text") or m.get("message")
            if user_msg is None and r in ("user", "human"):
                user_msg = t
            elif user_msg is not None and asst_msg is None and r in ("assistant", "gpt", "bot"):
                asst_msg = t
                break
        if user_msg:
            return to_str(user_msg).strip(), to_str(asst_msg).strip()
    if text is not None:
        if isinstance(role, str) and role.lower() in ("assistant", "gpt", "bot"):
            return None
        return to_str(text).strip(), ""
    return None


# ---- Adapter: Natural Questions (question + long context + short answer)
def extract_nq(ex: Dict[str, Any], dataset_id: str, split: str, idx: int, include_context: bool = True) -> Optional[Tuple[str, str]]:
    q = safe_get(ex, ["question", "query"])
    if q is None:
        return None
    context = safe_get(ex, ["document_text", "context", "long_answer", "passage"])
    prompt = f"Question: {to_str(q).strip()}"
    if include_context and context:
        prompt += "\n\nContext:\n" + to_str(context).strip()
    ans = ""
    a = safe_get(ex, ["answers", "short_answers", "answer"])
    if isinstance(a, dict):
        ans = to_str(safe_get(a, ["text", "answer"])) or ""
    elif isinstance(a, list) and len(a) > 0:
        first = a[0]
        if isinstance(first, dict):
            ans = to_str(safe_get(first, ["text", "answer"])) or ""
        else:
            ans = to_str(first)
    elif isinstance(a, str):
        ans = a
    return prompt.strip(), ans.strip()


# ---- Adapter: HotpotQA (question + supporting context + answer)
def extract_hotpotqa(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    q = safe_get(ex, ["question"])
    if q is None:
        return None
    ctx = safe_get(ex, ["context"])
    prompt = f"Question: {to_str(q).strip()}"
    
    sections = []
    if isinstance(ctx, dict):
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        if isinstance(titles, list) and isinstance(sentences, list):
            for title, sents in zip(titles, sentences):
                if isinstance(sents, list):
                    sections.append((to_str(title), [to_str(s).strip() for s in sents]))
    
    prompt += "\n\nSupporting Context:\n"
    for title, sents in sections:
        prompt += f"Title: {title}\n"
        for s in sents:
            prompt += f"- {s}\n"
    
    ans = to_str(safe_get(ex, ["answer"])).strip()
    return prompt.strip(), ans


# ---- Adapter: CNN/DailyMail (article + highlights)
def extract_cnn_dm(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    article = safe_get(ex, ["article", "document", "text"])
    summ = safe_get(ex, ["highlights", "summary", "abstract"])
    if article is None:
        return None
    prompt = "Summarize the following article:\n\n" + to_str(article).strip()
    ref = to_str(summ).strip()
    return prompt, ref


# ---- Adapter: GovReport (report + summary)
def extract_govreport(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    doc = safe_get(ex, ["report", "document", "text", "source"])
    summ = safe_get(ex, ["summary", "target", "abstract"])
    if doc is None:
        return None
    prompt = "Summarize the following government report:\n\n" + to_str(doc).strip()
    ref = to_str(summ).strip()
    return prompt, ref


# ---- Adapter: scientific_papers (arxiv/pubmed) (article + abstract)
def extract_scientific_papers(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    article = safe_get(ex, ["article", "paper", "text"])
    abstract = safe_get(ex, ["abstract", "summary"])
    if article is None:
        return None
    prompt = "Write a concise summary of the following paper:\n\n" + to_str(article).strip()
    ref = to_str(abstract).strip()
    return prompt, ref


# ---- Adapter: WritingPrompts (prompt + story)
def extract_writingprompts(ex: Dict[str, Any], dataset_id: str, split: str, idx: int) -> Optional[Tuple[str, str]]:
    p = safe_get(ex, ["prompt", "source_prompt", "writing_prompt"])
    story = safe_get(ex, ["story", "target", "response"])
    if p is None:
        return None
    prompt = "Write a story based on the following writing prompt:\n\n" + to_str(p).strip()
    ref = to_str(story).strip()
    return prompt, ref


DEFAULT_SOURCES = [
    ("tatsu-lab/alpaca", None, "extract_alpaca"), # short prefill short decode
    ("euclaise/writingprompts", None, "extract_writingprompts"), # short prefill long decode
    ("hotpotqa/hotpot_qa", {"name": "distractor"}, "extract_hotpotqa"), # long prefill short decode
    ("ccdv/govreport-summarization", None, "extract_govreport"), # long prefill long decode
]

ADAPTERS = {
    "extract_alpaca": extract_alpaca,
    "extract_oasst": extract_oasst,
    "extract_nq": extract_nq,
    "extract_hotpotqa": extract_hotpotqa,
    "extract_cnn_dm": extract_cnn_dm,
    "extract_govreport": extract_govreport,
    "extract_scientific_papers": extract_scientific_papers,
    "extract_writingprompts": extract_writingprompts,
}



# def bucket_name(prompt_toks: int, out_toks: int, prefill_long: int, decode_long: int) -> str:
#     prefill = "long" if prompt_toks >= prefill_long else "short"
#     decode = "long" if out_toks >= decode_long else "short"
#     return f"{prefill}_prefill__{decode}_decode"

# BUCKETS = [
#     "short_prefill__short_decode",
#     "short_prefill__long_decode",
#     "long_prefill__short_decode",
#     "long_prefill__long_decode",
# ]


def bucket_name(dataset_id: str) -> str:
    return dataset_id.split("/")[1]

BUCKETS = [
    bucket_name(dataset_id) for dataset_id, _, adapter_name in DEFAULT_SOURCES if adapter_name in ADAPTERS
]

def parse_dataset_spec(spec: str) -> Tuple[str, Optional[str]]:
    """
    Parse dataset spec:
      - "dataset_id" -> (dataset_id, None)
      - "dataset_id:config" -> (dataset_id, config)
    """
    if ":" in spec:
        ds, cfg = spec.split(":", 1)
        return ds.strip(), cfg.strip()
    return spec.strip(), None


def load_and_iterate(dataset_id: str, config: Optional[str], split: str):
    """Load HF dataset and yield examples."""
    kwargs = {}
    if config:
        kwargs["name"] = config
    ds = load_dataset(dataset_id, **kwargs, split=split)
    for ex in ds:
        yield ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--tokenizer", type=str, default="Qwen3/Qwen3-8B", help="HF tokenizer name, or 'whitespace'")
    ap.add_argument("--prefill_long", type=int, default=1024)
    ap.add_argument("--decode_long", type=int, default=256)
    ap.add_argument("--max_per_dataset", type=int, default=8192)
    ap.add_argument("--max_per_bucket", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=69)

    ap.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Optional override: HF dataset specs like 'cnn_dailymail:3.0.0' or 'hotpot_qa'. "
             "If not provided, uses a default set.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Comma-separated splits to try in order. Will skip missing splits.",
    )
    ap.add_argument(
        "--include_context_for_nq",
        action="store_true",
        help="If set, Natural Questions prompts include long document context (more long-prefill examples).",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    init_tokenizer(args.tokenizer)

    writers = {}
    paths = {}
    counts = {b: 0 for b in BUCKETS}
    for b in BUCKETS:
        path = os.path.join(args.out_dir, f"{b}.jsonl")
        paths[b] = path
        writers[b] = open(path, "w", encoding="utf-8")

    summary = {
        "prefill_long_threshold": args.prefill_long,
        "decode_long_threshold": args.decode_long,
        "tokenizer": args.tokenizer,
        "max_per_dataset": args.max_per_dataset,
        "max_per_bucket": args.max_per_bucket,
        "datasets_attempted": [],
        "datasets_loaded": [],
        "skipped": [],
        "bucket_counts": counts,
    }

    sources = []
    if args.dataset:
        for spec in args.dataset:
            ds, cfg = parse_dataset_spec(spec)
            sources.append((ds, {"name": cfg} if cfg else None, None))
    else:
        sources = DEFAULT_SOURCES

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    def close_all():
        for f in writers.values():
            try:
                f.close()
            except Exception:
                pass

    try:
        for entry in sources:
            dataset_id, loader_kwargs, adapter_name = entry
            if adapter_name is None:
                summary["skipped"].append({"dataset": dataset_id, "reason": "No adapter specified (use defaults or extend script)."})
                continue
            adapter = ADAPTERS.get(adapter_name)
            if adapter is None:
                summary["skipped"].append({"dataset": dataset_id, "reason": f"Unknown adapter {adapter_name}"})
                continue

            ds_tag = dataset_id
            cfg = None
            if loader_kwargs and "name" in loader_kwargs and loader_kwargs["name"]:
                cfg = loader_kwargs["name"]
                ds_tag = f"{dataset_id}:{cfg}"

            summary["datasets_attempted"].append(ds_tag)

            loaded_any_split = False
            for split in splits:
                try:
                    ds = load_dataset(dataset_id, **(loader_kwargs or {}), split=split)
                    loaded_any_split = True
                    summary["datasets_loaded"].append({"dataset": ds_tag, "split": split, "adapter": adapter_name})
                except Exception:
                    continue

                n_seen = 0
                it = iter(ds)
                pbar = tqdm(total=min(args.max_per_dataset, len(ds) if hasattr(ds, "__len__") else args.max_per_dataset),
                            desc=f"{ds_tag} [{split}]",
                            unit="ex")
                for idx, ex in enumerate(it):
                    if n_seen >= args.max_per_dataset:
                        break

                    try:
                        if adapter_name == "extract_nq":
                            pair = extract_nq(ex, dataset_id, split, idx, include_context=args.include_context_for_nq)
                        else:
                            pair = adapter(ex, dataset_id, split, idx)
                    except Exception:
                        pair = None

                    if not pair:
                        continue
                    prompt, ref = pair
                    if not prompt:
                        continue

                    prompt_toks = count_tokens(prompt)
                    ref_toks = count_tokens(ref)

                    # b = bucket_name(prompt_toks, ref_toks, args.prefill_long, args.decode_long)
                    b = bucket_name(dataset_id)
                    if b not in counts:
                        continue
                    if counts[b] >= args.max_per_bucket:
                        n_seen += 1
                        pbar.update(1)
                        continue

                    rec = {
                        "source": adapter_name,
                        "dataset_id": dataset_id,
                        "config": cfg,
                        "split": split,
                        "example_id": f"{ds_tag}:{split}:{idx}",
                        "prompt": prompt,
                        "ref_output": ref,
                        "prompt_tokens": prompt_toks,
                        "ref_output_tokens": ref_toks,
                        "bucket": b,
                    }

                    writers[b].write(json.dumps(rec, ensure_ascii=False) + "\n")
                    counts[b] += 1

                    n_seen += 1
                    pbar.update(1)

                    if all(counts[x] >= args.max_per_bucket for x in BUCKETS):
                        pbar.close()
                        raise StopIteration

                pbar.close()

            if not loaded_any_split:
                summary["skipped"].append({"dataset": ds_tag, "reason": "No requested splits found or dataset unavailable."})

    except StopIteration:
        pass
    finally:
        close_all()
        summary["bucket_counts"] = counts
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Wrote:")
    for b in BUCKETS:
        print(f"  {b}: {counts[b]} -> {paths[b]}")
    print(f"Summary: {os.path.join(args.out_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
