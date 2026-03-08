from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing import event_accumulator

_MIRROR_PAIRS_12: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 4),
    (2, 5),
    (6, 9),
    (7, 10),
    (8, 11),
)


def _extract_motor_id_from_dirname(name: str) -> int | None:
    if not name.startswith("m"):
        return None
    raw = name[1:]
    if not raw.isdigit():
        return None
    v = int(raw)
    if not (0 <= v <= 11):
        return None
    return v


def _parse_name_glob(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Invalid spec '{text}' (expected NAME=GLOB).")
    name, glob_pat = text.split("=", 1)
    name = name.strip()
    glob_pat = glob_pat.strip()
    if len(name) == 0 or len(glob_pat) == 0:
        raise ValueError(f"Invalid spec '{text}' (expected NAME=GLOB).")
    return name, glob_pat


def _latest_eval_json(run_dir: Path) -> Path | None:
    files = [p for p in run_dir.glob("eval_*.json") if not p.name.endswith("_meta.json")]
    if len(files) == 0:
        return None
    files.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    return files[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric(eval_data: dict[str, Any], scenario: str, metric_name: str, field: str = "mean") -> float:
    s = eval_data.get(scenario, {})
    if not isinstance(s, dict):
        return float("nan")
    m = s.get(metric_name, None)
    if isinstance(m, dict):
        v = m.get(field, float("nan"))
        try:
            return float(v)
        except Exception:
            return float("nan")
    if isinstance(m, (int, float)):
        return float(m)
    return float("nan")


def _collect_variant(eval_results_dir: Path, glob_pattern: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for run_dir in sorted(eval_results_dir.glob(glob_pattern)):
        if not run_dir.is_dir():
            continue
        mid = _extract_motor_id_from_dirname(run_dir.name)
        if mid is None:
            continue
        eval_json = _latest_eval_json(run_dir)
        if eval_json is None:
            continue
        data = _load_json(eval_json)
        out[mid] = {"run_dir": str(run_dir), "eval_json": str(eval_json), "eval": data}
    return out


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _mean(values: list[float]) -> float:
    vals = _finite(values)
    return float(statistics.fmean(vals)) if len(vals) > 0 else float("nan")


def _bottomk_mean(values_by_motor: dict[int, float], k: int) -> float:
    vals = sorted([v for v in values_by_motor.values() if math.isfinite(v)])
    if len(vals) == 0:
        return float("nan")
    kk = min(k, len(vals))
    return float(statistics.fmean(vals[:kk]))


def _topk_mean(values_by_motor: dict[int, float], k: int) -> float:
    vals = sorted([v for v in values_by_motor.values() if math.isfinite(v)], reverse=True)
    if len(vals) == 0:
        return float("nan")
    kk = min(k, len(vals))
    return float(statistics.fmean(vals[:kk]))


def _worst(values_by_motor: dict[int, float]) -> tuple[int | None, float]:
    finite = [(m, v) for m, v in values_by_motor.items() if math.isfinite(v)]
    if len(finite) == 0:
        return None, float("nan")
    m, v = min(finite, key=lambda x: x[1])
    return int(m), float(v)


def _find_latest_tb_run(tb_root: Path, run_glob: str) -> Path | None:
    cands = [p for p in tb_root.glob(run_glob) if p.is_dir()]
    if len(cands) == 0:
        return None
    cands.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    return cands[-1]


def _event_acc(run_dir: Path) -> event_accumulator.EventAccumulator:
    acc = event_accumulator.EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    return acc


def _scalar_values(acc: event_accumulator.EventAccumulator, tag: str) -> list[float]:
    tags = acc.Tags().get("scalars", [])
    if tag not in tags:
        return []
    return [float(e.value) for e in acc.Scalars(tag)]


@dataclass
class TBFloorCapSummary:
    data_available: bool
    event_dir: str | None
    pair_prob_min_seen: float
    pair_prob_max_seen: float
    pair_prob_violation_floor_count: int
    pair_prob_violation_cap_count: int
    pair_prob_violation_total_count: int
    pair_exposure_final_min_ratio: float
    pair_exposure_final_max_ratio: float


def _compute_tb_floor_cap(
    tb_root: Path,
    run_glob: str,
    floor: float,
    cap: float,
) -> TBFloorCapSummary:
    run_dir = _find_latest_tb_run(tb_root, run_glob)
    if run_dir is None:
        return TBFloorCapSummary(
            data_available=False,
            event_dir=None,
            pair_prob_min_seen=float("nan"),
            pair_prob_max_seen=float("nan"),
            pair_prob_violation_floor_count=0,
            pair_prob_violation_cap_count=0,
            pair_prob_violation_total_count=0,
            pair_exposure_final_min_ratio=float("nan"),
            pair_exposure_final_max_ratio=float("nan"),
        )
    try:
        acc = _event_acc(run_dir)
    except Exception:
        return TBFloorCapSummary(
            data_available=False,
            event_dir=str(run_dir),
            pair_prob_min_seen=float("nan"),
            pair_prob_max_seen=float("nan"),
            pair_prob_violation_floor_count=0,
            pair_prob_violation_cap_count=0,
            pair_prob_violation_total_count=0,
            pair_exposure_final_min_ratio=float("nan"),
            pair_exposure_final_max_ratio=float("nan"),
        )

    prob_series: list[float] = []
    floor_count = 0
    cap_count = 0
    eps = 1e-8
    for k in range(6):
        vals = _scalar_values(acc, f"phm/fault_pair_sampling_prob_pair_{k}")
        if len(vals) == 0:
            continue
        prob_series.extend(vals)
        floor_count += sum(1 for v in vals if v < (floor - eps))
        cap_count += sum(1 for v in vals if v > (cap + eps))

    exp_last: list[float] = []
    for k in range(6):
        vals = _scalar_values(acc, f"phm/fault_step_exposure_ratio_pair_{k}")
        if len(vals) > 0:
            exp_last.append(float(vals[-1]))

    data_ok = len(prob_series) > 0
    return TBFloorCapSummary(
        data_available=data_ok,
        event_dir=str(run_dir),
        pair_prob_min_seen=float(min(prob_series)) if len(prob_series) > 0 else float("nan"),
        pair_prob_max_seen=float(max(prob_series)) if len(prob_series) > 0 else float("nan"),
        pair_prob_violation_floor_count=int(floor_count),
        pair_prob_violation_cap_count=int(cap_count),
        pair_prob_violation_total_count=int(floor_count + cap_count),
        pair_exposure_final_min_ratio=float(min(exp_last)) if len(exp_last) > 0 else float("nan"),
        pair_exposure_final_max_ratio=float(max(exp_last)) if len(exp_last) > 0 else float("nan"),
    )


@dataclass
class VariantSummary:
    name: str
    critical_survival: dict[int, float]
    critical_peaksat_p95: dict[int, float]
    used_survival: dict[int, float]
    aged_survival: dict[int, float]
    critical_track_err: dict[int, float]
    worst_motor_id: int | None
    worst_survival: float
    bottom3_survival_mean: float
    cvar_top3_peaksat_p95_mean: float
    used_survival_mean: float
    aged_survival_mean: float
    critical_track_err_mean: float
    mirror_pair_gap_mean: float
    mirror_pair_gap_max: float
    mirror_pair_gap_max_pair: str
    tb_floor_cap: TBFloorCapSummary | None = None

    def as_row(self) -> dict[str, Any]:
        row = {
            "variant": self.name,
            "worst_motor_id": self.worst_motor_id,
            "worst_survival": self.worst_survival,
            "bottom3_survival_mean": self.bottom3_survival_mean,
            "cvar_top3_peaksat_p95_mean": self.cvar_top3_peaksat_p95_mean,
            "used_survival_mean": self.used_survival_mean,
            "aged_survival_mean": self.aged_survival_mean,
            "critical_track_err_mean": self.critical_track_err_mean,
            "mirror_pair_gap_mean": self.mirror_pair_gap_mean,
            "mirror_pair_gap_max": self.mirror_pair_gap_max,
            "mirror_pair_gap_max_pair": self.mirror_pair_gap_max_pair,
        }
        for i in range(12):
            row[f"critical_survival_m{i:02d}"] = self.critical_survival.get(i, float("nan"))
            row[f"critical_peaksat_p95_m{i:02d}"] = self.critical_peaksat_p95.get(i, float("nan"))
        if self.tb_floor_cap is not None:
            row.update(
                {
                    "tb_data_available": bool(self.tb_floor_cap.data_available),
                    "pair_prob_min_seen": self.tb_floor_cap.pair_prob_min_seen,
                    "pair_prob_max_seen": self.tb_floor_cap.pair_prob_max_seen,
                    "pair_prob_violation_floor_count": self.tb_floor_cap.pair_prob_violation_floor_count,
                    "pair_prob_violation_cap_count": self.tb_floor_cap.pair_prob_violation_cap_count,
                    "pair_prob_violation_total_count": self.tb_floor_cap.pair_prob_violation_total_count,
                    "pair_exposure_final_min_ratio": self.tb_floor_cap.pair_exposure_final_min_ratio,
                    "pair_exposure_final_max_ratio": self.tb_floor_cap.pair_exposure_final_max_ratio,
                    "tb_event_dir": self.tb_floor_cap.event_dir or "",
                }
            )
        return row


def _build_summary(name: str, runs: dict[int, dict[str, Any]]) -> VariantSummary:
    c_surv: dict[int, float] = {}
    c_peak_p95: dict[int, float] = {}
    u_surv: dict[int, float] = {}
    a_surv: dict[int, float] = {}
    c_track: dict[int, float] = {}
    for mid, payload in runs.items():
        ev = payload["eval"]
        c_surv[mid] = _metric(ev, "critical", "survived", "mean")
        c_peak_p95[mid] = _metric(ev, "critical", "peak_saturation_episode", "p95")
        u_surv[mid] = _metric(ev, "used", "survived", "mean")
        a_surv[mid] = _metric(ev, "aged", "survived", "mean")
        c_track[mid] = _metric(ev, "critical", "mean_tracking_error_xy", "mean")

    worst_id, worst_val = _worst(c_surv)
    pair_gap_values: list[tuple[tuple[int, int], float]] = []
    for a, b in _MIRROR_PAIRS_12:
        va = c_surv.get(a, float("nan"))
        vb = c_surv.get(b, float("nan"))
        if math.isfinite(va) and math.isfinite(vb):
            pair_gap_values.append(((a, b), abs(va - vb)))

    if len(pair_gap_values) > 0:
        mirror_pair_gap_mean = float(statistics.fmean(v for _, v in pair_gap_values))
        max_pair, max_gap = max(pair_gap_values, key=lambda x: x[1])
        mirror_pair_gap_max = float(max_gap)
        mirror_pair_gap_max_pair = f"{int(max_pair[0])}-{int(max_pair[1])}"
    else:
        mirror_pair_gap_mean = float("nan")
        mirror_pair_gap_max = float("nan")
        mirror_pair_gap_max_pair = ""

    return VariantSummary(
        name=name,
        critical_survival=c_surv,
        critical_peaksat_p95=c_peak_p95,
        used_survival=u_surv,
        aged_survival=a_surv,
        critical_track_err=c_track,
        worst_motor_id=worst_id,
        worst_survival=worst_val,
        bottom3_survival_mean=_bottomk_mean(c_surv, k=3),
        cvar_top3_peaksat_p95_mean=_topk_mean(c_peak_p95, k=3),
        used_survival_mean=_mean(list(u_surv.values())),
        aged_survival_mean=_mean(list(a_surv.values())),
        critical_track_err_mean=_mean(list(c_track.values())),
        mirror_pair_gap_mean=mirror_pair_gap_mean,
        mirror_pair_gap_max=mirror_pair_gap_max,
        mirror_pair_gap_max_pair=mirror_pair_gap_max_pair,
    )


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def _judge(
    control: VariantSummary,
    cand: VariantSummary,
    tol: float,
    pass_floor_cap: bool,
) -> dict[str, Any]:
    pass_worst = _is_finite(cand.worst_survival) and _is_finite(control.worst_survival) and (
        cand.worst_survival >= control.worst_survival
    )
    pass_bottom3 = _is_finite(cand.bottom3_survival_mean) and _is_finite(control.bottom3_survival_mean) and (
        cand.bottom3_survival_mean >= control.bottom3_survival_mean
    )
    pass_cvar = _is_finite(cand.cvar_top3_peaksat_p95_mean) and _is_finite(control.cvar_top3_peaksat_p95_mean) and (
        cand.cvar_top3_peaksat_p95_mean <= control.cvar_top3_peaksat_p95_mean
    )
    pass_used = _is_finite(cand.used_survival_mean) and _is_finite(control.used_survival_mean) and (
        cand.used_survival_mean >= (control.used_survival_mean - tol)
    )
    pass_aged = _is_finite(cand.aged_survival_mean) and _is_finite(control.aged_survival_mean) and (
        cand.aged_survival_mean >= (control.aged_survival_mean - tol)
    )
    pass_all = bool(pass_worst and pass_bottom3 and pass_cvar and pass_used and pass_aged and pass_floor_cap)
    return {
        "pass_worst": pass_worst,
        "pass_bottom3": pass_bottom3,
        "pass_cvar_top3": pass_cvar,
        "pass_used_survival": pass_used,
        "pass_aged_survival": pass_aged,
        "pass_floor_cap": bool(pass_floor_cap),
        "pass_all": pass_all,
        "delta_worst_survival": cand.worst_survival - control.worst_survival,
        "delta_bottom3_survival_mean": cand.bottom3_survival_mean - control.bottom3_survival_mean,
        "delta_cvar_top3_peaksat_p95_mean": cand.cvar_top3_peaksat_p95_mean - control.cvar_top3_peaksat_p95_mean,
        "delta_used_survival_mean": cand.used_survival_mean - control.used_survival_mean,
        "delta_aged_survival_mean": cand.aged_survival_mean - control.aged_survival_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare C0/C1/C2 default exposure variants with global acceptance criteria "
            "(worst, bottom3, CVaR-top3, used/aged retention, floor/cap compliance)."
        )
    )
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default="unitree_go2_phm/scripts/rsl_rl/eval_results",
        help="Base directory used to resolve eval variant globs.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant spec in NAME=GLOB format for eval results.",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="C0",
        help="Control variant name used for pass/fail deltas.",
    )
    parser.add_argument(
        "--used_aged_tolerance",
        type=float,
        default=0.02,
        help="Allowed absolute regression tolerance for used/aged mean survival.",
    )
    parser.add_argument(
        "--tb_log_root",
        type=str,
        default="unitree_go2_phm/scripts/rsl_rl/logs/rsl_rl/unitree_go2_realobs",
        help="TB run root directory.",
    )
    parser.add_argument(
        "--tb_run",
        action="append",
        default=[],
        help="TB run spec in NAME=GLOB format (same variant NAME key).",
    )
    parser.add_argument(
        "--pair_prob_floor",
        type=float,
        default=None,
        help="Expected floor for pair sampling probability compliance check.",
    )
    parser.add_argument(
        "--pair_prob_cap",
        type=float,
        default=None,
        help="Expected cap for pair sampling probability compliance check.",
    )
    parser.add_argument("--out_json", type=str, required=True, help="Output JSON path.")
    parser.add_argument("--out_csv", type=str, required=True, help="Output CSV path.")
    args = parser.parse_args()

    eval_root = Path(args.eval_results_dir)
    tb_root = Path(args.tb_log_root)
    specs = [_parse_name_glob(v) for v in args.variant]
    tb_specs = dict(_parse_name_glob(v) for v in args.tb_run)

    summaries: dict[str, VariantSummary] = {}
    for name, glob_pat in specs:
        runs = _collect_variant(eval_root, glob_pat)
        if len(runs) == 0:
            raise RuntimeError(f"No eval runs found for variant '{name}' with glob '{glob_pat}'.")
        summaries[name] = _build_summary(name, runs)

    floor_cap_check_enabled = args.pair_prob_floor is not None and args.pair_prob_cap is not None
    if floor_cap_check_enabled:
        floor = float(args.pair_prob_floor)
        cap = float(args.pair_prob_cap)
        if not (0.0 <= floor <= 1.0 and 0.0 <= cap <= 1.0 and floor <= cap):
            raise ValueError(f"Invalid floor/cap: floor={floor}, cap={cap}")

        for name, s in summaries.items():
            run_glob = tb_specs.get(name, "")
            if len(run_glob) == 0:
                s.tb_floor_cap = TBFloorCapSummary(
                    data_available=False,
                    event_dir=None,
                    pair_prob_min_seen=float("nan"),
                    pair_prob_max_seen=float("nan"),
                    pair_prob_violation_floor_count=0,
                    pair_prob_violation_cap_count=0,
                    pair_prob_violation_total_count=0,
                    pair_exposure_final_min_ratio=float("nan"),
                    pair_exposure_final_max_ratio=float("nan"),
                )
            else:
                s.tb_floor_cap = _compute_tb_floor_cap(tb_root, run_glob, floor=floor, cap=cap)

    if args.control not in summaries:
        raise RuntimeError(f"--control '{args.control}' not found in variants: {list(summaries.keys())}")
    control = summaries[args.control]

    rows: list[dict[str, Any]] = []
    out_obj: dict[str, Any] = {"control": args.control, "variants": {}}
    for name, s in summaries.items():
        row = s.as_row()
        pass_floor_cap = True
        if floor_cap_check_enabled:
            if s.tb_floor_cap is None:
                pass_floor_cap = False
            else:
                pass_floor_cap = bool(
                    s.tb_floor_cap.data_available and s.tb_floor_cap.pair_prob_violation_total_count == 0
                )

        if name != args.control:
            judge = _judge(control, s, tol=float(args.used_aged_tolerance), pass_floor_cap=pass_floor_cap)
        else:
            judge = {
                "pass_worst": True,
                "pass_bottom3": True,
                "pass_cvar_top3": True,
                "pass_used_survival": True,
                "pass_aged_survival": True,
                "pass_floor_cap": bool(pass_floor_cap),
                "pass_all": bool(pass_floor_cap),
                "delta_worst_survival": 0.0,
                "delta_bottom3_survival_mean": 0.0,
                "delta_cvar_top3_peaksat_p95_mean": 0.0,
                "delta_used_survival_mean": 0.0,
                "delta_aged_survival_mean": 0.0,
            }
        row.update(judge)
        rows.append(row)

        out_obj["variants"][name] = {
            "summary": {
                "worst_motor_id": s.worst_motor_id,
                "worst_survival": s.worst_survival,
                "bottom3_survival_mean": s.bottom3_survival_mean,
                "cvar_top3_peaksat_p95_mean": s.cvar_top3_peaksat_p95_mean,
                "used_survival_mean": s.used_survival_mean,
                "aged_survival_mean": s.aged_survival_mean,
                "critical_track_err_mean": s.critical_track_err_mean,
                "mirror_pair_gap_mean": s.mirror_pair_gap_mean,
                "mirror_pair_gap_max": s.mirror_pair_gap_max,
                "mirror_pair_gap_max_pair": s.mirror_pair_gap_max_pair,
            },
            "tb_floor_cap": None
            if s.tb_floor_cap is None
            else {
                "data_available": s.tb_floor_cap.data_available,
                "event_dir": s.tb_floor_cap.event_dir,
                "pair_prob_min_seen": s.tb_floor_cap.pair_prob_min_seen,
                "pair_prob_max_seen": s.tb_floor_cap.pair_prob_max_seen,
                "pair_prob_violation_floor_count": s.tb_floor_cap.pair_prob_violation_floor_count,
                "pair_prob_violation_cap_count": s.tb_floor_cap.pair_prob_violation_cap_count,
                "pair_prob_violation_total_count": s.tb_floor_cap.pair_prob_violation_total_count,
                "pair_exposure_final_min_ratio": s.tb_floor_cap.pair_exposure_final_min_ratio,
                "pair_exposure_final_max_ratio": s.tb_floor_cap.pair_exposure_final_max_ratio,
            },
            "judgement_vs_control": judge,
            "vectors": {
                "critical_survival": {str(k): float(v) for k, v in sorted(s.critical_survival.items())},
                "critical_peaksat_p95": {str(k): float(v) for k, v in sorted(s.critical_peaksat_p95.items())},
            },
        }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 148)
    print(
        f"{'Variant':<8} | {'WorstSurv':>9} | {'Bottom3Surv':>11} | {'CVaR-Top3 PeakSat p95':>22} | "
        f"{'UsedMean':>8} | {'AgedMean':>8} | {'floorcap':>8} | {'pass_all':>8}"
    )
    print("-" * 148)
    for r in rows:
        print(
            f"{str(r['variant']):<8} | "
            f"{float(r['worst_survival']):>9.4f} | "
            f"{float(r['bottom3_survival_mean']):>11.4f} | "
            f"{float(r['cvar_top3_peaksat_p95_mean']):>22.4f} | "
            f"{float(r['used_survival_mean']):>8.4f} | "
            f"{float(r['aged_survival_mean']):>8.4f} | "
            f"{str(bool(r['pass_floor_cap'])):>8} | "
            f"{str(bool(r['pass_all'])):>8}"
        )
    print("=" * 148)
    print(f"[DONE] JSON: {out_json}")
    print(f"[DONE] CSV : {out_csv}")


if __name__ == "__main__":
    main()
