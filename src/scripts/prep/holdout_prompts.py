from __future__ import annotations

import argparse
import json
from pathlib import Path

from sfs_core.prep.holdout_cache import prepare_holdout_prompt_cache
from sfs_core.shared.shared_experiment_helpers import (
    DEFAULT_SYSTEM_PROMPT,
    parse_chat_template_kwargs_json,
    resolve_chat_template_kwargs,
)
from sfs_core.shared.tokenizer_helpers import TOKENIZER_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare/cache a sliced holdout prompt set from bucketed JSONL files."
    )
    parser.add_argument("--source-bucket-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-id", type=str, required=True)
    parser.add_argument(
        "--tokenizer-mode",
        choices=TOKENIZER_MODES,
        default="auto",
        help="Tokenizer backend used to build and validate the holdout cache.",
    )
    parser.add_argument("--holdout-start-index", type=int, default=2500)
    parser.add_argument("--holdout-prompts-per-bucket", type=int, default=1000)
    parser.add_argument("--holdout-context-length", type=int, default=65536)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument(
        "--prompt-token-limit",
        type=int,
        default=32768,
        help="Maximum prompt-only tokens before truncation.",
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--chat-template-kwargs-json",
        dest="chat_template_kwargs",
        type=parse_chat_template_kwargs_json,
        default=None,
        help=(
            "JSON object passed to the tokenizer chat template and vLLM chat API. "
            "Defaults to Qwen's non-thinking mode; use '{}' for model families "
            "without Qwen's enable_thinking option."
        ),
    )
    args = parser.parse_args()
    args.chat_template_kwargs = resolve_chat_template_kwargs(
        args.chat_template_kwargs
    )
    return args


def main() -> None:
    args = parse_args()
    cache_path, manifest, rebuilt = prepare_holdout_prompt_cache(
        source_bucket_dir=args.source_bucket_dir,
        cache_dir=args.cache_dir,
        tokenizer_id=args.tokenizer_id,
        tokenizer_mode=args.tokenizer_mode,
        holdout_start_index=int(args.holdout_start_index),
        holdout_prompts_per_bucket=int(args.holdout_prompts_per_bucket),
        holdout_context_length=int(args.holdout_context_length),
        max_completion_tokens=int(args.max_completion_tokens),
        prompt_token_limit=int(args.prompt_token_limit),
        rebuild=bool(args.rebuild),
        system_prompt=str(args.system_prompt),
        chat_template_kwargs=args.chat_template_kwargs,
    )
    print(
        json.dumps(
            {
                "cache_dir": str(cache_path),
                "rebuilt": bool(rebuilt),
                "bucket_counts": manifest.get("bucket_counts", {}),
                "prompt_token_limit": manifest.get("prompt_token_limit"),
                "tokenizer_mode": manifest.get("tokenizer_mode"),
                "system_prompt": manifest.get("system_prompt"),
                "chat_template_kwargs": manifest.get("chat_template_kwargs"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
