#!/usr/bin/env python3
from __future__ import annotations

"""
Convert raw 500Hz real logs to a 50Hz replay-friendly CSV.

Expected input columns (minimum):
- timestamp: ts/time_s/time/t
- command: vx_cmd|vx|cmd_vx, vy_cmd|vy|cmd_vy, wz_cmd|yaw_rate|wz|cmd_wz
- temperature group (one of):
  - temp_m1..temp_m12
  - temp_0..temp_11
  - FR_hip_temp..RL_calf_temp
- pack voltage: vpack|vpack_v|power_v|battery_voltage
- cell voltage group (one of):
  - vcell_0..vcell_7
  - cell_vol_0..cell_vol_7
  - bms_cell_vol_0..bms_cell_vol_7

Output columns:
- t, vx, vy, yaw_rate, temp_max_c, vpack_v, vcell_min_v
"""

import argparse
import csv
from pathlib import Path


_GO2_TEMP_COLUMNS = [
    "FR_hip_temp",
    "FR_thigh_temp",
    "FR_calf_temp",
    "FL_hip_temp",
    "FL_thigh_temp",
    "FL_calf_temp",
    "RR_hip_temp",
    "RR_thigh_temp",
    "RR_calf_temp",
    "RL_hip_temp",
    "RL_thigh_temp",
    "RL_calf_temp",
]


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    if val in ("", None):
        return float(default)
    return float(val)


def _pick_one(fieldnames: list[str], candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in fieldnames:
            return c
    raise ValueError(f"Missing required '{label}' column. Tried: {candidates}")


def _pick_group(fieldnames: list[str], groups: list[list[str]], label: str) -> list[str]:
    for g in groups:
        if all(c in fieldnames for c in g):
            return g
    pretty = "; ".join(str(g) for g in groups)
    raise ValueError(f"Missing required '{label}' group. Tried: {pretty}")


def _window_start(ts: float, window_s: float) -> int:
    return int(ts // window_s)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raw 500Hz log -> 50Hz replay CSV converter.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--window_ms", type=int, default=20, help="Aggregation window in ms (default: 20ms -> 50Hz).")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    window_s = float(args.window_ms) / 1000.0
    if window_s <= 0:
        raise ValueError("window_ms must be > 0.")

    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        ts_col = _pick_one(fieldnames, ["ts", "time_s", "time", "t"], label="timestamp")
        vx_col = _pick_one(fieldnames, ["vx_cmd", "vx", "cmd_vx"], label="vx_cmd")
        vy_col = _pick_one(fieldnames, ["vy_cmd", "vy", "cmd_vy"], label="vy_cmd")
        wz_col = _pick_one(fieldnames, ["wz_cmd", "yaw_rate", "wz", "cmd_wz"], label="wz_cmd")
        vpack_col = _pick_one(fieldnames, ["vpack", "vpack_v", "power_v", "battery_voltage"], label="vpack")
        temp_cols = _pick_group(
            fieldnames,
            groups=[
                [f"temp_m{i}" for i in range(1, 13)],
                [f"temp_{i}" for i in range(12)],
                _GO2_TEMP_COLUMNS,
            ],
            label="motor temperatures (12)",
        )
        cell_cols = _pick_group(
            fieldnames,
            groups=[
                [f"vcell_{i}" for i in range(8)],
                [f"cell_vol_{i}" for i in range(8)],
                [f"bms_cell_vol_{i}" for i in range(8)],
            ],
            label="cell voltages (8)",
        )

        rows = list(reader)

    if len(rows) == 0:
        raise ValueError("Input CSV is empty.")

    # Group by fixed windows.
    groups: dict[int, list[dict[str, str]]] = {}
    for r in rows:
        ts = _f(r, ts_col)
        gid = _window_start(ts, window_s)
        groups.setdefault(gid, []).append(r)

    gids = sorted(groups.keys())
    out_rows: list[dict[str, float]] = []
    t0 = _f(groups[gids[0]][0], ts_col)

    for gid in gids:
        g = groups[gid]
        # Fixed rules:
        # - temp_max_c: max in window
        # - vcell_min_v: min in window
        # - vpack_v: last in window
        # - command: last in window
        temp_max_c = max(max(_f(r, c) for c in temp_cols) for r in g)
        vcell_min_v = min(min(_f(r, c, default=4.2) for c in cell_cols) for r in g)
        r_last = g[-1]
        ts_last = _f(r_last, ts_col)

        out_rows.append(
            {
                "t": ts_last - t0,
                "vx": _f(r_last, vx_col),
                "vy": _f(r_last, vy_col),
                "yaw_rate": _f(r_last, wz_col),
                "temp_max_c": temp_max_c,
                "vpack_v": _f(r_last, vpack_col),
                "vcell_min_v": vcell_min_v,
            }
        )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["t", "vx", "vy", "yaw_rate", "temp_max_c", "vpack_v", "vcell_min_v"],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        "[INFO] Column mapping: "
        f"ts={ts_col}, vx={vx_col}, vy={vy_col}, wz={wz_col}, vpack={vpack_col}, "
        f"temp={temp_cols[0]}..{temp_cols[-1]}, cell={cell_cols[0]}..{cell_cols[-1]}"
    )
    print(f"[DONE] Wrote {len(out_rows)} rows to {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
