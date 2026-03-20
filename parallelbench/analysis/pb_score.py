"""PBx score: summary metric for the speed-quality trade-off.

PBx = maximum tokens-per-step (TPS) that achieves >= x% average score
across all ParallelBench tasks.

Grouping logic:
  - Top-k methods: group by (unmasking, k) where k = max_tokens / steps.
    TPS = k (deterministic from config).
  - Threshold methods: group by (unmasking, alg_threshold).
    TPS = max_tokens / nfe (measured after generation).
  - Factor methods: group by (unmasking, alg_factor).
    TPS = max_tokens / nfe (measured after generation).

All configs assume block_length = max_tokens (fully parallel within block).

Example:
    PB80 = 8.0  means average TPS ~8 still yields >= 80% average score.
    PB70 = 16.0 means average TPS ~16 still yields >= 70% average score.
"""

from __future__ import annotations

from parallelbench.models.unmasking_registry import get_method_type

DEFAULT_THRESHOLDS = [90, 80, 70, 60]


def _make_config_key(row: dict) -> tuple | None:
    """Create a hashable config key and compute TPS for a row.

    Returns:
        (config_key, tps) tuple, or None if the row cannot be processed.
    """
    try:
        unmasking = str(row.get("unmasking", ""))
        nfe = float(row["nfe"])
        max_tokens = int(row["max_tokens"])
    except (KeyError, ValueError, TypeError):
        return None

    if nfe <= 0 or max_tokens <= 0:
        return None

    try:
        method_type = get_method_type(unmasking)
    except KeyError:
        return None

    # Use tokens_per_step metric (actual measured parallelism) for TPS
    try:
        tps = float(row["tokens_per_step"])
    except (KeyError, ValueError, TypeError):
        tps = max_tokens / nfe

    if method_type == "threshold":
        threshold = row.get("alg_threshold", "")
        config_key = (unmasking, f"threshold={threshold}")
    elif method_type == "factor":
        factor = row.get("alg_factor", "")
        config_key = (unmasking, f"factor={factor}")
    else:
        # Top-k: use k from gen_kwargs, or derive from max_tokens / steps
        try:
            k = float(row["k"])
        except (KeyError, ValueError, TypeError):
            try:
                steps = int(row["steps"])
            except (KeyError, ValueError, TypeError):
                return None
            if steps <= 0:
                return None
            k = max_tokens / steps
        config_key = (unmasking, f"k={k}")
        tps = k

    return config_key, tps


def compute_pb_scores(
    rows: list[dict],
    thresholds: list[int] | None = None,
) -> dict[str, float | None]:
    """Compute PBx scores from result rows.

    Args:
        rows: List of result dicts. Each dict should contain:
            - "task" (str): task name
            - "score" (float): task score in [0, 100]
            - "nfe" (float): number of forward evaluations
            - "max_tokens" (int): maximum output length
            - "steps" (int): denoising steps (for top-k methods)
            - "unmasking" (str): unmasking method name
            - "alg_threshold" / "alg_factor": for adaptive methods
        thresholds: Score thresholds on a 0-100 scale (e.g., [90, 80, 70, 60]).

    Returns:
        Dict mapping "PBx" to the maximum TPS achieving that threshold,
        or None if no config achieves it.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Group by config: collect per-task (score, tps) pairs
    config_data: dict[tuple, dict[str, list[tuple[float, float]]]] = {}

    for row in rows:
        try:
            score = float(row["score"])
        except (KeyError, ValueError, TypeError):
            continue

        result = _make_config_key(row)
        if result is None:
            continue

        config_key, tps = result
        task = row["task"]

        if config_key not in config_data:
            config_data[config_key] = {}
        if task not in config_data[config_key]:
            config_data[config_key][task] = []
        config_data[config_key][task].append((score, tps))

    # Compute per-config: average score, average TPS
    config_summaries: list[tuple[float, float]] = []  # (average_tps, average_score)
    for task_data in config_data.values():
        task_avg_scores = []
        task_avg_tps = []
        for entries in task_data.values():
            scores = [e[0] for e in entries]
            tps_values = [e[1] for e in entries]
            task_avg_scores.append(sum(scores) / len(scores))
            task_avg_tps.append(sum(tps_values) / len(tps_values))

        avg_score = sum(task_avg_scores) / len(task_avg_scores)
        avg_tps = sum(task_avg_tps) / len(task_avg_tps)
        config_summaries.append((avg_tps, avg_score))

    # Decide scoring strategy based on method type.
    # Top-k methods have deterministic TPS (= k), so use discrete scoring.
    # All other methods (threshold, factor, klass, etc.) have measured TPS,
    # so use linear interpolation between adjacent (TPS, score) points.
    method_types = set()
    for config_key in config_data:
        try:
            method_types.add(get_method_type(config_key[0]))
        except KeyError:
            pass
    use_interpolation = bool(method_types) and "topk" not in method_types

    # Sort by TPS ascending; deduplicate same-TPS configs (keep highest score).
    sorted_configs = sorted(config_summaries, key=lambda x: x[0])
    deduped: list[tuple[float, float]] = []
    for tps, score in sorted_configs:
        if deduped and deduped[-1][0] == tps:
            if score > deduped[-1][1]:
                deduped[-1] = (tps, score)
        else:
            deduped.append((tps, score))
    sorted_configs = deduped

    pb_scores: dict[str, float | None] = {}
    for threshold in sorted(thresholds, reverse=True):
        result: float | None = None

        if use_interpolation:
            # Linear interpolation: find the rightmost crossing point
            # where score transitions from at-or-above to below threshold.
            if sorted_configs and sorted_configs[-1][1] >= threshold:
                result = sorted_configs[-1][0]
            else:
                for i in range(len(sorted_configs) - 1, 0, -1):
                    tps_hi, score_hi = sorted_configs[i]
                    tps_lo, score_lo = sorted_configs[i - 1]
                    if score_lo >= threshold > score_hi and score_lo != score_hi:
                        result = tps_lo + (score_lo - threshold) / (
                            score_lo - score_hi
                        ) * (tps_hi - tps_lo)
                        break
        else:
            # Discrete: max TPS among configs that meet the threshold
            qualifying = [tps for tps, avg in sorted_configs if avg >= threshold]
            result = max(qualifying) if qualifying else None

        pb_scores[f"PB{threshold}"] = result

    return pb_scores
