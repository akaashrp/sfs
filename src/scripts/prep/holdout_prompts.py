from __future__ import annotations

import argparse
import json
from pathlib import Path

from sfs_core.prep.holdout_cache import prepare_holdout_prompt_cache
from sfs_core.shared.shared_experiment_helpers import DEFAULT_SYSTEM_PROMPT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare/cache a sliced holdout prompt set from bucketed JSONL files."
    )
    parser.add_argument("--source-bucket-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-id", type=str, required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path, manifest, rebuilt = prepare_holdout_prompt_cache(
        source_bucket_dir=args.source_bucket_dir,
        cache_dir=args.cache_dir,
        tokenizer_id=args.tokenizer_id,
        holdout_start_index=int(args.holdout_start_index),
        holdout_prompts_per_bucket=int(args.holdout_prompts_per_bucket),
        holdout_context_length=int(args.holdout_context_length),
        max_completion_tokens=int(args.max_completion_tokens),
        prompt_token_limit=int(args.prompt_token_limit),
        rebuild=bool(args.rebuild),
        system_prompt=str(args.system_prompt),
    )
    print(
        json.dumps(
            {
                "cache_dir": str(cache_path),
                "rebuilt": bool(rebuilt),
                "bucket_counts": manifest.get("bucket_counts", {}),
                "prompt_token_limit": manifest.get("prompt_token_limit"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
