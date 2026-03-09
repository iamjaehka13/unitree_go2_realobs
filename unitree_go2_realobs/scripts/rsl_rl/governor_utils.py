from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class ReplaySchedule:
    dt: float
    t: np.ndarray
    commands: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.commands.shape[0])

    @property
    def duration_s(self) -> float:
        if self.num_steps == 0:
            return 0.0
        return float(self.t[-1] + self.dt)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _to_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for '{name}': {value}") from exc


def _column(row: dict[str, str], names: list[str], required: bool = True) -> float | None:
    for key in names:
        if key in row and row[key] not in (None, ""):
            return _to_float(row[key], key)
    if required:
        raise ValueError(f"Missing one of required columns: {names}")
    return None


def _parse_csv(path: Path, default_dt: float) -> ReplaySchedule:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise ValueError(f"Replay CSV is empty: {path}")

    vx = np.array([_column(r, ["vx", "lin_vel_x", "cmd_vx"]) for r in rows], dtype=np.float64)
    vy = np.array([_column(r, ["vy", "lin_vel_y", "cmd_vy"], required=False) or 0.0 for r in rows], dtype=np.float64)
    wz = np.array(
        [_column(r, ["yaw_rate", "wz", "ang_vel_z", "cmd_wz"]) for r in rows],
        dtype=np.float64,
    )

    has_t = any(("t" in r and r["t"] not in (None, "")) for r in rows)
    if has_t:
        t_raw = np.array([_column(r, ["t", "time_s"]) for r in rows], dtype=np.float64)
        if np.any(np.diff(t_raw) <= 0.0):
            raise ValueError("Replay CSV time column must be strictly increasing.")
        dt = float(default_dt)
        grid = np.arange(0.0, t_raw[-1] + 1e-9, dt, dtype=np.float64)
        vx_u = np.interp(grid, t_raw, vx)
        vy_u = np.interp(grid, t_raw, vy)
        wz_u = np.interp(grid, t_raw, wz)
        cmd = np.stack((vx_u, vy_u, wz_u), axis=1)
        return ReplaySchedule(dt=dt, t=grid, commands=cmd)

    dt = float(default_dt)
    t = np.arange(0.0, len(rows) * dt, dt, dtype=np.float64)
    cmd = np.stack((vx, vy, wz), axis=1)
    return ReplaySchedule(dt=dt, t=t, commands=cmd)


def _expand_segments(
    segments: list[dict[str, Any]],
    defaults: dict[str, float],
) -> list[tuple[float, float, float, float]]:
    expanded: list[tuple[float, float, float, float]] = []
    for seg in segments:
        repeat = int(seg.get("repeat", 1))
        local_defaults = {
            "vx": float(defaults.get("vx", 0.0)),
            "vy": float(defaults.get("vy", 0.0)),
            "yaw_rate": float(defaults.get("yaw_rate", 0.0)),
        }
        if "defaults" in seg and isinstance(seg["defaults"], dict):
            for key in ("vx", "vy", "yaw_rate"):
                if key in seg["defaults"]:
                    local_defaults[key] = _to_float(seg["defaults"][key], key)

        for _ in range(repeat):
            if "sequence" in seg:
                nested = seg["sequence"]
                if not isinstance(nested, list):
                    raise ValueError("Each 'sequence' must be a list of segments.")
                expanded.extend(_expand_segments(nested, local_defaults))
                continue

            duration = _to_float(seg.get("duration_s"), "duration_s")
            if duration <= 0.0:
                raise ValueError("Segment duration_s must be > 0.")
            vx = _to_float(seg.get("vx", local_defaults["vx"]), "vx")
            vy = _to_float(seg.get("vy", local_defaults["vy"]), "vy")
            yaw_rate = _to_float(seg.get("yaw_rate", local_defaults["yaw_rate"]), "yaw_rate")
            expanded.append((duration, vx, vy, yaw_rate))
    return expanded


def _parse_structured(path: Path, default_dt: float) -> ReplaySchedule:
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load YAML replay files. Install with `pip install pyyaml`."
            )
        with path.open("r") as f:
            payload = yaml.safe_load(f)
    else:
        with path.open("r") as f:
            payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Replay file must contain a dict/object at top level.")

    dt = float(payload.get("dt", default_dt))
    defaults = payload.get("defaults", {}) or {}
    if not isinstance(defaults, dict):
        raise ValueError("`defaults` must be an object/dict.")

    segments = payload.get("segments", payload.get("sequence"))
    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError("Replay file must define non-empty `segments` (or `sequence`).")

    expanded = _expand_segments(segments, defaults)
    cmd_rows: list[tuple[float, float, float]] = []
    for duration, vx, vy, yaw_rate in expanded:
        steps = max(1, int(round(duration / dt)))
        cmd_rows.extend([(vx, vy, yaw_rate)] * steps)

    if len(cmd_rows) == 0:
        raise ValueError("Expanded replay command is empty.")

    cmd = np.array(cmd_rows, dtype=np.float64)
    total_override = payload.get("total_duration_s", None)
    if total_override is not None:
        target_steps = max(1, int(round(float(total_override) / dt)))
        if target_steps < cmd.shape[0]:
            cmd = cmd[:target_steps]
        elif target_steps > cmd.shape[0]:
            pad = np.repeat(cmd[-1:, :], repeats=target_steps - cmd.shape[0], axis=0)
            cmd = np.concatenate([cmd, pad], axis=0)

    t = np.arange(0.0, cmd.shape[0] * dt, dt, dtype=np.float64)
    return ReplaySchedule(dt=dt, t=t, commands=cmd)


def load_replay_schedule(command_file: str, default_dt: float = 0.02) -> ReplaySchedule:
    path = Path(command_file)
    if not path.exists():
        raise FileNotFoundError(f"Replay command file not found: {command_file}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _parse_csv(path, default_dt=default_dt)
    if suffix in (".json", ".yaml", ".yml"):
        return _parse_structured(path, default_dt=default_dt)
    raise ValueError(f"Unsupported replay file extension: {suffix}")


@dataclass
class GovernorConfig:
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


class ThermalVoltageGovernor:
    """Continuous replay governor driven by temperature and cell-voltage margin."""

    def __init__(self, cfg: GovernorConfig):
        self.cfg = cfg
        self.scale = 1.0
        self._temp_ema: float | None = None
        self._temp_prev: float | None = None
        self._cell_v_ema: float | None = None
        self._pack_v_ema: float | None = None

    def reset(self):
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

    def step(
        self, dt: float, temp_max_c: float, cell_v_min: float, pack_v: float
    ) -> tuple[float, float, bool, dict[str, float]]:
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

        debug = {
            "temp_ema_c": float(self._temp_ema),
            "temp_pred_c": float(temp_pred),
            "temp_rate_cps": float(temp_rate),
            "cell_v_ema": float(self._cell_v_ema),
            "pack_v_ema": float(self._pack_v_ema),
            "s_temp": float(s_temp),
            "s_volt": float(s_volt),
            "s_target": float(target_scale),
            "hard_stop_temp": float(hard_stop_temp),
            "hard_stop_pack": float(hard_stop_pack),
            "hard_stop_cell": float(hard_stop_cell),
        }
        return float(self.scale), float(yaw_scale), bool(hard_stop), debug


def leg_joint_names(leg: str) -> list[str]:
    leg_norm = leg.strip().upper()
    if leg_norm not in {"FR", "FL", "RR", "RL"}:
        raise ValueError(f"Unsupported leg tag: {leg}")
    return [f"{leg_norm}_hip_joint", f"{leg_norm}_thigh_joint", f"{leg_norm}_calf_joint"]
