from __future__ import annotations

from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
SFS_ROOT = SRC_ROOT.parent

ASSETS_ROOT = SRC_ROOT / "assets"
PREDICTORS_ROOT = ASSETS_ROOT / "predictors"
TEMPLATES_ROOT = ASSETS_ROOT / "templates"
EXPERIMENTS_ROOT = SFS_ROOT / "experiments"
EXPERIMENT_DATA_ROOT = EXPERIMENTS_ROOT / "data"
PROMPTS_DATA_ROOT = EXPERIMENT_DATA_ROOT / "prompts"
BUCKETED_PROMPTS_ROOT = PROMPTS_DATA_ROOT / "bucketed_prompts"
DEFAULT_BUCKET_POOL_QWEN3_0_6B = BUCKETED_PROMPTS_ROOT / "qwen3-0.6b"
BUCKETED_OUTPUTS_ROOT = EXPERIMENTS_ROOT / "bucketed_prompt_outputs"


def ensure_experiments_root() -> Path:
    EXPERIMENTS_ROOT.mkdir(parents=True, exist_ok=True)
    return EXPERIMENTS_ROOT


def ensure_experiment_data_root() -> Path:
    EXPERIMENT_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return EXPERIMENT_DATA_ROOT


def ensure_prompts_data_root() -> Path:
    PROMPTS_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return PROMPTS_DATA_ROOT


def ensure_bucketed_outputs_root() -> Path:
    BUCKETED_OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    return BUCKETED_OUTPUTS_ROOT


__all__ = [
    "SRC_ROOT",
    "SFS_ROOT",
    "ASSETS_ROOT",
    "PREDICTORS_ROOT",
    "TEMPLATES_ROOT",
    "EXPERIMENTS_ROOT",
    "EXPERIMENT_DATA_ROOT",
    "PROMPTS_DATA_ROOT",
    "BUCKETED_PROMPTS_ROOT",
    "DEFAULT_BUCKET_POOL_QWEN3_0_6B",
    "BUCKETED_OUTPUTS_ROOT",
    "ensure_experiments_root",
    "ensure_experiment_data_root",
    "ensure_prompts_data_root",
    "ensure_bucketed_outputs_root",
]
