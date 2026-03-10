#!/usr/bin/env python3
from __future__ import annotations

"""Aggregate safety-tail and post-unlatch transition metrics from eval JSON files."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from scenario_labels import scenario_label, scenario_lookup_keys


METRIC_SPECS: dict[str, dict[str, str]] = {
    # Legacy tail metrics
    "peak_saturation_episode": {"PeakSat_max": "max", "PeakSat_p95": "p95"},
    "crit_latch_mean_dur_s_ep": {"LatchDur_s_max": "max", "LatchDur_s_p95": "p95"},
    "crit_latch_mean_dur_steps_ep": {"LatchDur_steps_max": "max", "LatchDur_steps_p95": "p95"},
    # Recovery and post-unlatch transition metrics
    "crit_time_to_first_latch_s": {"T1Latch_s_mean": "mean", "T1Latch_s_p95": "p95"},
    "crit_time_to_unlatch_s": {"TUnlatch_s_mean": "mean", "TUnlatch_s_p95": "p95"},
    "crit_time_to_unlatch_after_zero_s": {
        "TUnlatchAfterZero_s_mean": "mean",
        "TUnlatchAfterZero_s_p95": "p95",
    },
    "crit_post_unlatch_peak_sat_p95_ep": {
        "PostUnlatchPeakSat95_mean": "mean",
        "PostUnlatchPeakSat95_p95": "p95",
    },
    "crit_post_unlatch_fixedwin_peak_sat_p95_ep": {
        "PostUnlatchFixWinPeakSat95_mean": "mean",
        "PostUnlatchFixWinPeakSat95_p95": "p95",
    },
    "crit_unlatch_success_ep": {
        "UnlatchSucc_mean": "mean",
        "UnlatchSucc_p95": "p95",
    },
    "crit_unlatch_after_zero_success_ep": {
        "UnlatchAfterZeroSucc_mean": "mean",
        "UnlatchAfterZeroSucc_p95": "p95",
    },
    "crit_post_unlatch_action_ramp_active_ratio_ep": {
        "PostUnlatchRampActiveRatio_mean": "mean",
        "PostUnlatchRampActiveRatio_p95": "p95",
    },
    "crit_action_transition_delta_norm_mean_ep": {
        "ActionTransitionDeltaNorm_mean": "mean",
        "ActionTransitionDeltaNorm_p95": "p95",
    },
    "crit_action_transition_delta_norm_max_ep": {
        "ActionTransitionDeltaNormMax_mean": "mean",
        "ActionTransitionDeltaNormMax_p95": "p95",
    },
}

META_KEYS = {
    "eval_protocol_mode": "protocol",
    "eval_cmd_profile_effective": "cmd",
    "fault_fixed_motor_id_eval": "motor_id",
    "critical_governor_sat_trigger_hi_effective": "sat_hi",
    "critical_governor_sat_trigger_lo_effective": "sat_lo",
    "critical_governor_unlatch_require_low_cmd_effective": "unlatch_low_cmd",
    "critical_governor_unlatch_require_sat_recovery_effective": "unlatch_sat_recovery",
    "critical_governor_unlatch_stable_steps_effective": "stable_steps",
    "critical_governor_post_unlatch_action_ramp_s_effective": "post_ramp_s",
    "critical_governor_post_unlatch_action_delta_max_effective": "post_delta_max",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _extract_ts(path: Path) -> str:
    m = re.search(r"eval_(\d{8}_\d{6})", path.name)
    return m.group(1) if m else ""


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.3f}"
    return str(v)


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, float) and math.isnan(v):
        return True
    return False


def _pick_latest_per_dir(files: list[Path]) -> list[Path]:
    best: dict[Path, tuple[str, Path]] = {}
    for f in files:
        ts = _extract_ts(f)
        parent = f.parent
        if parent not in best or ts > best[parent][0]:
            best[parent] = (ts, f)
    return [v[1] for v in sorted(best.values(), key=lambda x: x[0])]


def _run_label(run_dir: str) -> str:
    if "tuneA" in run_dir:
        return "A"
    if "tuneB" in run_dir:
        return "B"
    if "tuneC" in run_dir:
        return "C"
    if "tuneD" in run_dir or "govOFF" in run_dir:
        return "D"
    if "modeE" in run_dir:
        return "E"
    if "fixA" in run_dir:
        return "A_fix"
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize eval JSON with safety-tail, recovery, and post-unlatch transition metrics."
    )
    parser.add_argument("--root", type=Path, required=True, help="Root directory that contains eval run dirs.")
    parser.add_argument("--glob", type=str, default="*/eval_*.json", help="Glob under root.")
    parser.add_argument("--all", action="store_true", help="Use all eval files (default: latest per run dir only).")
    parser.add_argument(
        "--only-label",
        type=str,
        default="",
        help="Comma-separated labels to include (e.g., A,B,C,D or A_fix). Empty = include all.",
    )
    parser.add_argument("--csv_out", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument(
        "--latest-per-label",
        action="store_true",
        help="After filtering, keep only latest row per label (A/B/C/D/A_fix).",
    )
    parser.add_argument(
        "--prefer-having-p95",
        type=str,
        default="",
        help=(
            "With --latest-per-label, prefer runs that have non-missing p95 columns. "
            "Comma-separated column names or 'auto' for all *_p95 columns."
        ),
    )
    args = parser.parse_args()

    files = sorted(args.root.glob(args.glob))
    if not args.all:
        files = _pick_latest_per_dir(files)

    allowed_labels: set[str] | None = None
    raw_label_filter = str(args.only_label).strip()
    if raw_label_filter:
        allowed_labels = {x.strip().upper() for x in raw_label_filter.split(",") if x.strip()}

    rows: list[dict[str, Any]] = []
    for eval_path in files:
        if eval_path.name.endswith("_meta.json"):
            continue
        try:
            data = _load_json(eval_path)
        except Exception:
            continue

        critical = None
        for candidate in scenario_lookup_keys("critical"):
            payload = data.get(candidate)
            if isinstance(payload, dict):
                critical = payload
                break
        if not isinstance(critical, dict):
            continue

        label = _run_label(eval_path.parent.name)
        if allowed_labels is not None and label.upper() not in allowed_labels:
            continue

        meta_path = eval_path.with_name(f"{eval_path.stem}_meta.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = _load_json(meta_path)
            except Exception:
                meta = {}

        row: dict[str, Any] = {
            "label": label,
            "scenario": scenario_label("critical"),
            "scenario_key": "critical",
            "ts": _extract_ts(eval_path),
            "run_dir": eval_path.parent.name,
            "file": eval_path.name,
            "Survival%": float(critical.get("survived", {}).get("mean", float("nan"))) * 100.0,
            "TrackErr": float(critical.get("mean_tracking_error_xy", {}).get("mean", float("nan"))),
        }

        for meta_key, out_key in META_KEYS.items():
            row[out_key] = meta.get(meta_key)

        for metric_key, col_to_stat in METRIC_SPECS.items():
            stats = critical.get(metric_key, {})
            if isinstance(stats, dict):
                for col_name, stat_key in col_to_stat.items():
                    row[col_name] = stats.get(stat_key)
                row[f"{metric_key}_count"] = stats.get("count")
            else:
                for col_name in col_to_stat:
                    row[col_name] = None
                row[f"{metric_key}_count"] = None

        p95_cols_all = [col for spec in METRIC_SPECS.values() for col, stat in spec.items() if stat == "p95"]
        row["has_p95"] = bool(all(not _is_missing(row.get(col)) for col in p95_cols_all))
        rows.append(row)

    if not rows:
        print("No matching eval JSON files found.")
        return

    if args.latest_per_label:
        raw_prefer = str(args.prefer_having_p95).strip()
        if raw_prefer.lower() == "auto":
            prefer_cols = [col for spec in METRIC_SPECS.values() for col, stat in spec.items() if stat == "p95"]
        elif raw_prefer:
            prefer_cols = [x.strip() for x in raw_prefer.split(",") if x.strip()]
        else:
            prefer_cols = []

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get("label", "")), []).append(row)

        latest_by_label: dict[str, dict[str, Any]] = {}
        labels_missing_p95: list[str] = []
        for label, cands in grouped.items():
            cands_sorted = sorted(cands, key=lambda r: str(r.get("ts", "")), reverse=True)
            chosen = cands_sorted[0]
            if prefer_cols:
                chosen = next(
                    (
                        r
                        for r in cands_sorted
                        if all(not _is_missing(r.get(col)) for col in prefer_cols)
                    ),
                    cands_sorted[0],
                )
            latest_by_label[label] = chosen
            if prefer_cols and any(_is_missing(chosen.get(col)) for col in prefer_cols):
                labels_missing_p95.append(label)

        rows = [latest_by_label[k] for k in sorted(latest_by_label.keys())]
        if labels_missing_p95:
            missing_txt = ", ".join(sorted(labels_missing_p95))
            print(
                f"[WARN] latest-per-label selected runs without full p95 columns for labels: {missing_txt}. "
                "Run post-p95 evaluations or adjust --prefer-having-p95 columns."
            )

    columns = [
        "label",
        "ts",
        "run_dir",
        "protocol",
        "cmd",
        "motor_id",
        "sat_hi",
        "sat_lo",
        "unlatch_low_cmd",
        "unlatch_sat_recovery",
        "stable_steps",
        "post_ramp_s",
        "post_delta_max",
        "has_p95",
        "Survival%",
        "TrackErr",
        "PeakSat_max",
        "PeakSat_p95",
        "LatchDur_s_max",
        "LatchDur_s_p95",
        "LatchDur_steps_max",
        "LatchDur_steps_p95",
        "T1Latch_s_mean",
        "T1Latch_s_p95",
        "TUnlatch_s_mean",
        "TUnlatch_s_p95",
        "TUnlatchAfterZero_s_mean",
        "TUnlatchAfterZero_s_p95",
        "UnlatchSucc_mean",
        "UnlatchSucc_p95",
        "PostUnlatchPeakSat95_mean",
        "PostUnlatchPeakSat95_p95",
        "UnlatchAfterZeroSucc_mean",
        "UnlatchAfterZeroSucc_p95",
        "PostUnlatchFixWinPeakSat95_mean",
        "PostUnlatchFixWinPeakSat95_p95",
        "PostUnlatchRampActiveRatio_mean",
        "PostUnlatchRampActiveRatio_p95",
        "ActionTransitionDeltaNorm_mean",
        "ActionTransitionDeltaNorm_p95",
        "ActionTransitionDeltaNormMax_mean",
        "ActionTransitionDeltaNormMax_p95",
    ]
    for k in list(rows[0].keys()):
        if k.endswith("_count"):
            columns.append(k)

    str_rows = [[_fmt(row.get(col)) for col in columns] for row in rows]
    widths = [len(c) for c in columns]
    for r in str_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    print(_line(columns))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for r in str_rows:
        print(_line(r))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(str_rows)
        print(f"\nCSV saved to: {args.csv_out}")


if __name__ == "__main__":
    main()
