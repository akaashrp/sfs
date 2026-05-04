from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_literal_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as src:
        for raw in src:
            line = raw.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue
            if isinstance(record, dict):
                yield record


def _resolve_path_from_router_json(raw_path: str, *, router_json_path: Path) -> Path:
    candidate = Path(raw_path).expanduser()
    candidates = [candidate]
    router_dir = router_json_path.parent
    if not candidate.is_absolute():
        candidates.append((router_dir / candidate).expanduser())
    candidates.append((router_dir / candidate.name).expanduser())

    for path in candidates:
        if path.exists():
            return path
    return candidate


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of empty values.")
    if len(values) == 1:
        return float(values[0])
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))

    sorted_values = sorted(values)
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _plot_distribution(
    *,
    values: list[float],
    output_path: Path,
    title: str,
    bins: int,
    x_label: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_values = sorted(values)
    n = len(sorted_values)
    ecdf_y = [(i + 1) / n for i in range(n)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (hist_ax, cdf_ax) = plt.subplots(1, 2, figsize=(12, 4.8))

    hist_ax.hist(values, bins=bins, alpha=0.85)
    hist_ax.set_xlabel(x_label)
    hist_ax.set_ylabel("Count")
    hist_ax.set_title("Histogram")
    hist_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    cdf_ax.plot(sorted_values, ecdf_y, linewidth=1.5)
    cdf_ax.set_xlabel(x_label)
    cdf_ax.set_ylabel("ECDF")
    cdf_ax.set_ylim(0.0, 1.0)
    cdf_ax.set_title("Cumulative Distribution")
    cdf_ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
