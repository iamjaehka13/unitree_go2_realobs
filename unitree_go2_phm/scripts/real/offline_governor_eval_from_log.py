#!/usr/bin/env python3
from __future__ import annotations

"""
Offline governor evaluation from a 50Hz log CSV.

Input columns (minimum):
- t, vx, vy, yaw_rate, temp_max_c, vpack_v, vcell_min_v

Output:
- summary.json with safety/performance proxy metrics
"""

import argparse
import csv
import json
from pathlib import Path
import sys
from dataclasses import dataclass

_THIS_DIR = Path(__file__).resolve().parent
_RSL_RL_DIR = _THIS_DIR.parent / "rsl_rl"
if str(_RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_DIR))


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class _FallbackGovernorConfig:
    temp_warn_c: float = 65.0
    temp_crit_c: float = 70.0
    temp_stop_c: float = 75.0
    cell_warn_v: float = 3.20
    cell_scale_stop_v: float = 3.05
    cell_hard_stop_v: float = 3.00
    pack_hard_stop_v: float = 24.5
    temp_prediction_horizon_s: float = 1.0
    temp_filter_tau_s: float = 1.0
    cell_filter_tau_s: float = 0.2
    pack_filter_tau_s: float = 0.2
    yaw_exponent: float = 1.5
    max_scale_rise_per_s: float = 0.8
    max_scale_drop_per_s: float = 2.0


class _FallbackThermalVoltageGovernor:
    def __init__(self, cfg: _FallbackGovernorConfig):
        self.cfg = cfg
        self.scale = 1.0
        self._temp_ema: float | None = None
        self._temp_prev: float | None = None
        self._cell_v_ema: float | None = None
        self._pack_v_ema: float | None = None

    def reset(self) -> None:
        self.scale = 1.0
        self._temp_ema = None
        self._temp_prev = None
        self._cell_v_ema = None
        self._pack_v_ema = None

    def _ema(self, old: float | None, value: float, dt: float, tau: float) -> float:
        if old is None:
            return value
        alpha = _clamp(dt / max(tau + dt, 1e-6), 0.0, 1.0)
        return old + alpha * (value - old)

    def step(self, dt: float, temp_max_c: float, cell_v_min: float, pack_v: float):
        self._temp_ema = self._ema(self._temp_ema, temp_max_c, dt, self.cfg.temp_filter_tau_s)
        self._cell_v_ema = self._ema(self._cell_v_ema, cell_v_min, dt, self.cfg.cell_filter_tau_s)
        self._pack_v_ema = self._ema(self._pack_v_ema, pack_v, dt, self.cfg.pack_filter_tau_s)
        if self._temp_prev is None:
            temp_rate = 0.0
        else:
            temp_rate = (self._temp_ema - self._temp_prev) / max(dt, 1e-6)
        self._temp_prev = self._temp_ema

        temp_pred = self._temp_ema + self.cfg.temp_prediction_horizon_s * temp_rate
        temp_den = max(self.cfg.temp_crit_c - self.cfg.temp_warn_c, 1e-6)
        volt_den = max(self.cfg.cell_warn_v - self.cfg.cell_scale_stop_v, 1e-6)
        s_temp = _clamp((self.cfg.temp_crit_c - temp_pred) / temp_den, 0.0, 1.0)
        s_volt = _clamp((self._cell_v_ema - self.cfg.cell_scale_stop_v) / volt_den, 0.0, 1.0) ** 2.0
        target_scale = min(s_temp, s_volt)
        max_drop = self.cfg.max_scale_drop_per_s * dt
        max_rise = self.cfg.max_scale_rise_per_s * dt
        if target_scale < self.scale:
            self.scale = max(target_scale, self.scale - max_drop)
        else:
            self.scale = min(target_scale, self.scale + max_rise)
        self.scale = _clamp(self.scale, 0.0, 1.0)

        yaw_scale = self.scale ** max(self.cfg.yaw_exponent, 1.0)
        hard_stop_temp = self._temp_ema >= self.cfg.temp_stop_c
        hard_stop_pack = pack_v <= self.cfg.pack_hard_stop_v
        hard_stop_cell = cell_v_min <= self.cfg.cell_hard_stop_v
        hard_stop = hard_stop_temp or hard_stop_pack or hard_stop_cell
        dbg = {
            "hard_stop_temp": float(hard_stop_temp),
            "hard_stop_pack": float(hard_stop_pack),
            "hard_stop_cell": float(hard_stop_cell),
        }
        return float(self.scale), float(yaw_scale), bool(hard_stop), dbg


def _load_governor_classes():
    try:
        from replay_utils import GovernorConfig, ThermalVoltageGovernor

        return GovernorConfig, ThermalVoltageGovernor
    except Exception as exc:
        print(
            f"[WARN] replay_utils import failed ({exc}); using local governor fallback.",
            file=sys.stderr,
        )
        return _FallbackGovernorConfig, _FallbackThermalVoltageGovernor


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return float(default)
    return float(val)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline governor eval from 50Hz log CSV.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--governor", action="store_true", default=False)
    parser.add_argument("--temp_warn_c", type=float, default=65.0)
    parser.add_argument("--temp_crit_c", type=float, default=70.0)
    parser.add_argument("--temp_stop_c", type=float, default=75.0)
    parser.add_argument("--cell_warn_v", type=float, default=3.20)
    parser.add_argument("--cell_stop_v", type=float, default=3.05)
    parser.add_argument("--cell_hard_v", type=float, default=3.00)
    parser.add_argument("--pack_stop_v", type=float, default=24.5)
    parser.add_argument("--temp_pred_horizon_s", type=float, default=1.0)
    parser.add_argument("--temp_tau_s", type=float, default=1.0)
    parser.add_argument("--volt_tau_s", type=float, default=0.2)
    parser.add_argument("--yaw_exp", type=float, default=1.5)
    args = parser.parse_args()

    gov = None
    if args.governor:
        GovernorConfig, ThermalVoltageGovernor = _load_governor_classes()
        cfg = GovernorConfig(
            temp_warn_c=float(args.temp_warn_c),
            temp_crit_c=float(args.temp_crit_c),
            temp_stop_c=float(args.temp_stop_c),
            cell_warn_v=float(args.cell_warn_v),
            cell_scale_stop_v=float(args.cell_stop_v),
            cell_hard_stop_v=float(args.cell_hard_v),
            pack_hard_stop_v=float(args.pack_stop_v),
            temp_prediction_horizon_s=float(args.temp_pred_horizon_s),
            temp_filter_tau_s=float(args.temp_tau_s),
            cell_filter_tau_s=float(args.volt_tau_s),
            pack_filter_tau_s=float(args.volt_tau_s),
            yaw_exponent=float(args.yaw_exp),
        )
        gov = ThermalVoltageGovernor(cfg)
        gov.reset()

    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | bool | str]] = []
    hard_stop_reason = ""
    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            temp = _f(r, "temp_max_c", 25.0)
            vpack = _f(r, "vpack_v", 33.6)
            vcell = _f(r, "vcell_min_v", 4.2)
            vx = _f(r, "vx", 0.0)
            vy = _f(r, "vy", 0.0)
            wz = _f(r, "yaw_rate", 0.0)

            if args.governor:
                assert gov is not None
                s_lin, s_yaw, hard_stop, dbg = gov.step(float(args.dt), temp, vcell, vpack)
            else:
                s_lin, s_yaw, hard_stop = 1.0, 1.0, False
                dbg = {"hard_stop_temp": 0.0, "hard_stop_pack": 0.0, "hard_stop_cell": 0.0}

            if hard_stop:
                tags: list[str] = []
                if float(dbg.get("hard_stop_temp", 0.0)) > 0.5:
                    tags.append("temp")
                if float(dbg.get("hard_stop_pack", 0.0)) > 0.5:
                    tags.append("pack")
                if float(dbg.get("hard_stop_cell", 0.0)) > 0.5:
                    tags.append("cell")
                hard_stop_reason = "governor_hard_stop_" + ("+".join(tags) if len(tags) > 0 else "unknown")

            rows.append(
                {
                    "temp_max_c": temp,
                    "vpack_v": vpack,
                    "vcell_min_v": vcell,
                    "vx_cmd_raw": vx,
                    "vy_cmd_raw": vy,
                    "wz_cmd_raw": wz,
                    "vx_cmd_exec": vx * s_lin,
                    "vy_cmd_exec": vy * s_lin,
                    "wz_cmd_exec": wz * s_yaw,
                    "scale_lin": s_lin,
                    "scale_yaw": s_yaw,
                    "hard_stop": hard_stop,
                }
            )
            if hard_stop:
                break

    if len(rows) == 0:
        output_json.write_text(json.dumps({"steps": 0, "error": "no rows"}, indent=2))
        print(f"[WARN] Empty rows. Wrote: {output_json}")
        return 0

    steps = len(rows)
    dt = float(args.dt)
    temp_over_warn = sum(1 for x in rows if float(x["temp_max_c"]) >= float(args.temp_warn_c))
    vpack_min = min(float(x["vpack_v"]) for x in rows)
    vcell_min = min(float(x["vcell_min_v"]) for x in rows)
    mean_scale_lin = sum(float(x["scale_lin"]) for x in rows) / steps
    time_scale_lt_0p9 = sum(1 for x in rows if float(x["scale_lin"]) < 0.9) * dt
    hard_stop = any(bool(x["hard_stop"]) for x in rows)

    summary = {
        "config": {
            "governor": bool(args.governor),
            "dt": dt,
            "temp_warn_c": float(args.temp_warn_c),
            "temp_crit_c": float(args.temp_crit_c),
            "temp_stop_c": float(args.temp_stop_c),
            "cell_warn_v": float(args.cell_warn_v),
            "cell_stop_v": float(args.cell_stop_v),
            "cell_hard_v": float(args.cell_hard_v),
            "pack_stop_v": float(args.pack_stop_v),
            "temp_pred_horizon_s": float(args.temp_pred_horizon_s),
            "temp_tau_s": float(args.temp_tau_s),
            "volt_tau_s": float(args.volt_tau_s),
            "yaw_exp": float(args.yaw_exp),
        },
        "metrics": {
            "steps": steps,
            "duration_s": steps * dt,
            "time_temp_over_warn_s": temp_over_warn * dt,
            "vpack_min_v": vpack_min,
            "vcell_min_v": vcell_min,
            "mean_scale_lin": mean_scale_lin,
            "time_scale_lt_0p9_s": time_scale_lt_0p9,
            "hard_stop": bool(hard_stop),
            "hard_stop_reason": hard_stop_reason,
        },
    }
    output_json.write_text(json.dumps(summary, indent=2))
    print(f"[DONE] Wrote summary: {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
