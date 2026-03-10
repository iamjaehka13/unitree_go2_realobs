from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from scenario_labels import (
    scenario_key as _scenario_key,
    scenario_label as _scenario_label,
    scenario_lookup_keys as _scenario_lookup_keys,
)


MIRROR_PAIRS = ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))
JOINT_TYPE_GROUPS = {
    "hip": (0, 3, 6, 9),
    "thigh": (1, 4, 7, 10),
    "calf": (2, 5, 8, 11),
}


def _extract_motor_id(path: Path) -> int | None:
    # Expected directory names like "..._m10" or "..._m10.log" prefix.
    m = re.search(r"_m(\d+)(?:$|[_\\.])", path.name)
    if m is None:
        return None
    return int(m.group(1))


def _latest_eval_json(run_dir: Path) -> Path | None:
    eval_files = [
        p for p in run_dir.glob("eval_*.json")
        if not p.name.endswith("_meta.json")
    ]
    if len(eval_files) == 0:
        return None
    eval_files.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    return eval_files[-1]


def _load_eval_json(eval_json_path: Path) -> dict[str, Any]:
    with eval_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_runs(eval_results_dir: Path, glob_pattern: str) -> dict[int, dict[str, Any]]:
    runs_by_motor: dict[int, dict[str, Any]] = {}
    for run_dir in sorted(eval_results_dir.glob(glob_pattern)):
        if not run_dir.is_dir():
            continue
        motor_id = _extract_motor_id(run_dir)
        if motor_id is None:
            continue
        eval_json = _latest_eval_json(run_dir)
        if eval_json is None:
            continue
        cur = {
            "run_dir": str(run_dir),
            "eval_json": str(eval_json),
            "mtime_ns": int(eval_json.stat().st_mtime_ns),
        }
        prev = runs_by_motor.get(motor_id)
        if prev is None or cur["mtime_ns"] >= prev["mtime_ns"]:
            runs_by_motor[motor_id] = cur
    return runs_by_motor


def _metric_mean(eval_result: dict[str, Any], scenario: str, metric: str) -> float:
    scenario_key = _scenario_key(scenario)
    for candidate in _scenario_lookup_keys(scenario_key):
        s = eval_result.get(candidate, {})
        if not isinstance(s, dict):
            continue
        v = s.get(metric, None)
        if isinstance(v, dict):
            return float(v.get("mean", float("nan")))
        if isinstance(v, (int, float)):
            return float(v)
    return float("nan")


def _percentile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return float((1.0 - weight) * ordered[lower] + weight * ordered[upper])


def _nanstats(values: dict[int, float]) -> dict[str, Any]:
    arr = [values[k] for k in sorted(values.keys())]
    finite = [v for v in arr if math.isfinite(v)]
    n_total = int(len(arr))
    n_valid = int(len(finite))
    if n_valid == 0:
        return {
            "n": n_total,
            "n_valid": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p10": float("nan"),
            "max": float("nan"),
        }
    std = statistics.pstdev(finite) if n_valid > 1 else 0.0
    return {
        "n": n_total,
        "n_valid": n_valid,
        "mean": float(statistics.fmean(finite)),
        "std": float(std),
        "min": float(min(finite)),
        "p10": float(_percentile(finite, 0.10)),
        "max": float(max(finite)),
    }


def _pair_gaps(values: dict[int, float]) -> tuple[dict[str, float], float]:
    gaps: dict[str, float] = {}
    for a, b in MIRROR_PAIRS:
        va = values.get(a, float("nan"))
        vb = values.get(b, float("nan"))
        if math.isfinite(va) and math.isfinite(vb):
            gaps[f"{a}-{b}"] = abs(va - vb)
    return gaps, float(gaps.get("7-10", float("nan")))


def _joint_group_means(values: dict[int, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for group_name, group_ids in JOINT_TYPE_GROUPS.items():
        group_vals = [values.get(i, float("nan")) for i in group_ids]
        group_vals = [v for v in group_vals if math.isfinite(v)]
        out[group_name] = float(statistics.fmean(group_vals)) if len(group_vals) > 0 else float("nan")
    return out


def _worst_id(values: dict[int, float]) -> tuple[int | None, float]:
    finite = [(k, v) for k, v in values.items() if math.isfinite(v)]
    if len(finite) == 0:
        return None, float("nan")
    worst = min(finite, key=lambda x: x[1])
    return int(worst[0]), float(worst[1])


def _build_variant_summary(
    runs_by_motor: dict[int, dict[str, Any]],
    scenarios: list[str],
    metrics: list[str],
) -> dict[str, Any]:
    per_motor_eval: dict[int, dict[str, Any]] = {}
    for motor_id, entry in runs_by_motor.items():
        per_motor_eval[motor_id] = _load_eval_json(Path(entry["eval_json"]))

    summary: dict[str, Any] = {"runs": runs_by_motor, "scenarios": {}}
    for scenario in scenarios:
        scenario_out: dict[str, Any] = {}
        for metric in metrics:
            values: dict[int, float] = {}
            for motor_id, eval_result in per_motor_eval.items():
                values[motor_id] = _metric_mean(eval_result, scenario=scenario, metric=metric)

            stats = _nanstats(values)
            pair_gap_all, pair_gap_7_10 = _pair_gaps(values)
            group_means = _joint_group_means(values)
            worst_motor_id, worst_value = _worst_id(values)
            scenario_out[metric] = {
                **stats,
                "worst_motor_id": worst_motor_id,
                "worst_value": worst_value,
                "pair_gap_7_10": pair_gap_7_10,
                "pair_gaps": pair_gap_all,
                "group_means": group_means,
                "values_by_motor": {str(k): float(v) for k, v in sorted(values.items())},
                "vector_id0_11": [float(values.get(i, float("nan"))) for i in range(12)],
            }
        summary["scenarios"][scenario] = scenario_out
    return summary


def _rows_for_csv(combined_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant, variant_summary in combined_summary.items():
        scenarios = variant_summary.get("scenarios", {})
        for scenario, metric_map in scenarios.items():
            for metric, payload in metric_map.items():
                row = {
                    "variant": variant,
                    "scenario": _scenario_label(scenario),
                    "scenario_key": scenario,
                    "metric": metric,
                    "n": payload.get("n"),
                    "n_valid": payload.get("n_valid"),
                    "mean": payload.get("mean"),
                    "std": payload.get("std"),
                    "min": payload.get("min"),
                    "p10": payload.get("p10"),
                    "max": payload.get("max"),
                    "worst_motor_id": payload.get("worst_motor_id"),
                    "worst_value": payload.get("worst_value"),
                    "pair_gap_7_10": payload.get("pair_gap_7_10"),
                    "hip_mean": payload.get("group_means", {}).get("hip"),
                    "thigh_mean": payload.get("group_means", {}).get("thigh"),
                    "calf_mean": payload.get("group_means", {}).get("calf"),
                }
                vector = payload.get("vector_id0_11", [])
                for i in range(12):
                    row[f"id{i}"] = vector[i] if i < len(vector) else float("nan")
                rows.append(row)
    return rows


def _print_compact_table(summary: dict[str, Any], scenario: str, metric: str):
    scenario_key = _scenario_key(scenario)
    scenario_label = _scenario_label(scenario_key)
    print("=" * 112)
    print(
        f"{'Variant':<12} | {'Scenario':<10} | {'Metric':<18} | {'Mean±Std':<20} | "
        f"{'Worst(min/p10)':<24} | {'pair_gap(7,10)':>14}"
    )
    print("-" * 112)
    for variant, variant_summary in summary.items():
        payload = (
            variant_summary.get("scenarios", {})
            .get(scenario_key, {})
            .get(metric, {})
        )
        if len(payload) == 0:
            continue
        mean = float(payload.get("mean", float("nan")))
        std = float(payload.get("std", float("nan")))
        min_v = float(payload.get("min", float("nan")))
        p10 = float(payload.get("p10", float("nan")))
        gap = float(payload.get("pair_gap_7_10", float("nan")))
        print(
            f"{variant:<12} | {scenario_label:<10} | {metric:<18} | "
            f"{mean:.4f} +/- {std:.4f}   | min={min_v:.4f}, p10={p10:.4f}   | {gap:>14.4f}"
        )
    print("=" * 112)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Summarize motor-id sweep results from per-motor evaluation directories "
            "(mean±std + worst + mirror-pair gaps)."
        )
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant spec in the form name=glob. Repeatable.",
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="unitree_go2_realobs/scripts/rsl_rl/eval_results",
        help="Directory containing per-motor run folders.",
    )
    parser.add_argument(
        "--realobs_glob",
        type=str,
        default=None,
        help="Glob pattern for RealObs per-motor run directories.",
    )
    parser.add_argument(
        "--baseline_glob",
        type=str,
        default=None,
        help="Glob pattern for Baseline per-motor run directories.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["nominal", "moderate", "severe", "critical"],
        help="Scenario names to summarize (paper labels or legacy internal keys).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["survived", "survival_walk_only", "walk_ratio", "stand_ratio", "shuffle_ratio"],
        help="Per-scenario scalar metrics to summarize across motor ids.",
    )
    parser.add_argument(
        "--table_scenario",
        type=str,
        default="critical",
        help="Scenario used for compact stdout table (paper label or legacy internal key).",
    )
    parser.add_argument(
        "--table_metric",
        type=str,
        default="survived",
        help="Metric used for compact stdout table.",
    )
    parser.add_argument("--out_json", type=str, required=True, help="Output summary JSON path.")
    parser.add_argument("--out_csv", type=str, required=True, help="Output summary CSV path.")
    args = parser.parse_args()

    eval_results_dir = Path(args.eval_results_dir)

    variant_specs: dict[str, str] = {}
    if args.variant:
        for item in args.variant:
            if "=" not in item:
                raise ValueError(f"Invalid --variant format: {item!r} (expected name=glob).")
            name, glob_pattern = item.split("=", 1)
            name = name.strip()
            glob_pattern = glob_pattern.strip()
            if name == "" or glob_pattern == "":
                raise ValueError(f"Invalid --variant format: {item!r} (expected name=glob).")
            variant_specs[name] = glob_pattern
    elif args.realobs_glob and args.baseline_glob:
        variant_specs = {
            "realobs": str(args.realobs_glob),
            "baseline": str(args.baseline_glob),
        }
    else:
        raise ValueError("Provide either repeatable --variant name=glob or both --realobs_glob and --baseline_glob.")

    combined_summary = {}
    scenario_keys = [_scenario_key(s) for s in list(args.scenarios)]
    for variant_name, glob_pattern in variant_specs.items():
        variant_runs = _collect_runs(eval_results_dir, glob_pattern)
        combined_summary[variant_name] = _build_variant_summary(
            variant_runs,
            scenarios=scenario_keys,
            metrics=list(args.metrics),
        )
        combined_summary[variant_name]["scenario_labels"] = {
            key: _scenario_label(key) for key in scenario_keys
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(combined_summary, f, indent=2, ensure_ascii=False)

    rows = _rows_for_csv(combined_summary)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) > 0:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        out_csv.write_text("", encoding="utf-8")

    _print_compact_table(
        summary=combined_summary,
        scenario=str(args.table_scenario),
        metric=str(args.table_metric),
    )
    print(f"[DONE] JSON summary: {out_json}")
    print(f"[DONE] CSV summary : {out_csv}")


if __name__ == "__main__":
    main()
