#!/usr/bin/env python3
from __future__ import annotations

"""
Log Go2 LowState directly via unitree_sdk2_python (CycloneDDS), without ROS2.

This logger writes a replay-friendly raw CSV that is compatible with
`log_to_replay_csv.py` column matching:
  - timestamp: `ts`
  - commands: `vx_cmd`, `vy_cmd`, `wz_cmd` (static/default unless overridden)
  - temps: `temp_m1..temp_m12`, `temp_0..temp_11`, joint-name temps
  - pack voltage: `vpack_v`
  - cell voltages: `bms_cell_vol_0..bms_cell_vol_7` (V)

Usage example:
  python3 unitree_go2_realobs/scripts/real/log_collector_sdk2py.py \
    --interface enp3s0 \
    --output_csv ./go2_full_log_sdk2py.csv \
    --log_hz 500
"""

import argparse
import csv
import signal
import sys
import time
from pathlib import Path
from threading import Lock


JOINT_NAMES = [
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
]


def _cell_raw_to_volt(raw: int) -> float:
    # Unitree messages may expose mV integers; keep robust conversion.
    if raw <= 0:
        return 0.0
    if raw > 100:
        return float(raw) * 1e-3
    return float(raw)


def _safe_list_get(x, idx: int, default: float = 0.0) -> float:
    try:
        return float(x[idx])
    except Exception:
        return float(default)


class _LowStateCsvLogger:
    def __init__(
        self,
        writer: csv.writer,
        cmd_vx: float,
        cmd_vy: float,
        cmd_wz: float,
        cmd_source_id: str,
        log_hz: float,
        flush_every_n: int,
        print_every_s: float,
    ):
        self.writer = writer
        self.cmd_vx = float(cmd_vx)
        self.cmd_vy = float(cmd_vy)
        self.cmd_wz = float(cmd_wz)
        self.cmd_source_id = str(cmd_source_id)
        self.log_period = (1.0 / float(log_hz)) if float(log_hz) > 0.0 else 0.0
        self.flush_every_n = max(int(flush_every_n), 1)
        self.print_every_s = max(float(print_every_s), 0.1)

        self._start_mono = time.monotonic()
        self._last_logged_mono = 0.0
        self._last_print_mono = self._start_mono
        self._rows = 0
        self._callbacks = 0
        self._gated = 0
        self._lock = Lock()

    @property
    def rows(self) -> int:
        return int(self._rows)

    @property
    def callbacks(self) -> int:
        return int(self._callbacks)

    @property
    def gated(self) -> int:
        return int(self._gated)

    def on_lowstate(self, msg) -> None:
        now_mono = time.monotonic()
        now_wall = time.time()
        t_rel = now_mono - self._start_mono

        with self._lock:
            self._callbacks += 1
            if self.log_period > 0.0 and (now_mono - self._last_logged_mono) < self.log_period:
                self._gated += 1
                return
            self._last_logged_mono = now_mono

            imu = getattr(msg, "imu_state", None)
            bms = getattr(msg, "bms_state", None)
            motor_state = getattr(msg, "motor_state", [])
            foot_force = getattr(msg, "foot_force", [])
            foot_force_est = getattr(msg, "foot_force_est", [])

            row: list[object] = []

            # Core timing / command
            row += [
                float(now_wall),                     # ts
                float(t_rel),                        # time_s
                int(getattr(msg, "tick", 0)),        # tick
                int(getattr(msg, "level_flag", 0)),  # level_flag
                int(getattr(msg, "bit_flag", 0)),    # bit_flag
                self.cmd_vx,                         # vx_cmd
                self.cmd_vy,                         # vy_cmd
                self.cmd_wz,                         # wz_cmd
                self.cmd_source_id,                  # cmd_source_id
            ]

            # IMU
            quat = getattr(imu, "quaternion", [0.0, 0.0, 0.0, 1.0]) if imu is not None else [0.0, 0.0, 0.0, 1.0]
            gyro = getattr(imu, "gyroscope", [0.0, 0.0, 0.0]) if imu is not None else [0.0, 0.0, 0.0]
            acc = getattr(imu, "accelerometer", [0.0, 0.0, 0.0]) if imu is not None else [0.0, 0.0, 0.0]
            rpy = getattr(imu, "rpy", [0.0, 0.0, 0.0]) if imu is not None else [0.0, 0.0, 0.0]
            imu_temp = int(getattr(imu, "temperature", 0)) if imu is not None else 0
            row += [
                _safe_list_get(quat, 0), _safe_list_get(quat, 1), _safe_list_get(quat, 2), _safe_list_get(quat, 3),
                _safe_list_get(gyro, 0), _safe_list_get(gyro, 1), _safe_list_get(gyro, 2),
                _safe_list_get(acc, 0), _safe_list_get(acc, 1), _safe_list_get(acc, 2),
                _safe_list_get(rpy, 0), _safe_list_get(rpy, 1), _safe_list_get(rpy, 2),
                imu_temp,
            ]

            # Power/BMS core
            row += [
                float(getattr(msg, "power_v", 0.0)),   # vpack_v
                float(getattr(msg, "power_a", 0.0)),   # ibatt_a
                int(getattr(bms, "soc", 0)) if bms is not None else 0,
                int(getattr(bms, "status", 0)) if bms is not None else 0,
                int(getattr(bms, "current", 0)) if bms is not None else 0,
                int(getattr(bms, "cycle", 0)) if bms is not None else 0,
            ]

            # 8-cell voltage (V) + raw
            cell_vol = getattr(bms, "cell_vol", []) if bms is not None else []
            for i in range(8):
                raw_i = int(_safe_list_get(cell_vol, i, 0.0))
                row.append(_cell_raw_to_volt(raw_i))  # bms_cell_vol_i
            for i in range(8):
                raw_i = int(_safe_list_get(cell_vol, i, 0.0))
                row.append(raw_i)                     # bms_cell_raw_i

            # Motor channels (12 DoF)
            motor_temp_12: list[float] = []
            for i in range(12):
                m = motor_state[i] if i < len(motor_state) else None
                if m is None:
                    mode = 0
                    q = 0.0
                    dq = 0.0
                    ddq = 0.0
                    tau = 0.0
                    temp = 0
                    lost = 0
                else:
                    mode = int(getattr(m, "mode", 0))
                    q = float(getattr(m, "q", 0.0))
                    dq = float(getattr(m, "dq", 0.0))
                    ddq = float(getattr(m, "ddq", 0.0))
                    tau = float(getattr(m, "tau_est", 0.0))
                    temp = int(getattr(m, "temperature", 0))
                    lost = int(getattr(m, "lost", 0))
                row += [mode, q, dq, ddq, tau, temp, lost]
                motor_temp_12.append(float(temp))

            # Temperature aliases for downstream tools.
            # temp_m1..12
            for i in range(12):
                row.append(motor_temp_12[i])
            # temp_0..11
            for i in range(12):
                row.append(motor_temp_12[i])
            # FR_hip_temp..RL_calf_temp
            for i in range(12):
                row.append(motor_temp_12[i])

            # Foot/contact channels
            for i in range(4):
                row.append(int(_safe_list_get(foot_force, i, 0.0)))
            for i in range(4):
                row.append(int(_safe_list_get(foot_force_est, i, 0.0)))

            self.writer.writerow(row)
            self._rows += 1

            if (self._rows % self.flush_every_n) == 0:
                # csv file handle flush is managed by outer loop via writer's file.
                pass

            if (now_mono - self._last_print_mono) >= self.print_every_s:
                hz = self._rows / max(t_rel, 1e-6)
                print(
                    f"[LOG] rows={self._rows} callbacks={self._callbacks} "
                    f"gate_drop={self._gated} avg_hz={hz:.1f} "
                    f"vpack={float(getattr(msg, 'power_v', 0.0)):.2f} "
                    f"temp_max={max(motor_temp_12) if len(motor_temp_12) > 0 else 0.0:.1f}"
                )
                self._last_print_mono = now_mono


def _build_header() -> list[str]:
    header: list[str] = []
    header += ["ts", "time_s", "tick", "level_flag", "bit_flag"]
    header += ["vx_cmd", "vy_cmd", "wz_cmd", "cmd_source_id"]

    header += [
        "imu_qw", "imu_qx", "imu_qy", "imu_qz",
        "imu_gyro_x", "imu_gyro_y", "imu_gyro_z",
        "imu_acc_x", "imu_acc_y", "imu_acc_z",
        "imu_rpy_x", "imu_rpy_y", "imu_rpy_z",
        "imu_temp",
    ]

    header += ["vpack_v", "ibatt_a", "bms_soc", "bms_status", "bms_current", "bms_cycle"]

    for i in range(8):
        header.append(f"bms_cell_vol_{i}")
    for i in range(8):
        header.append(f"bms_cell_raw_{i}")

    for name in JOINT_NAMES:
        header += [
            f"{name}_mode",
            f"{name}_q",
            f"{name}_dq",
            f"{name}_ddq",
            f"{name}_tau_est",
            f"{name}_temp",
            f"{name}_lost",
        ]

    for i in range(1, 13):
        header.append(f"temp_m{i}")
    for i in range(12):
        header.append(f"temp_{i}")
    for name in JOINT_NAMES:
        header.append(f"{name}_temp_alias")

    for i in range(4):
        header.append(f"foot_force_{i}")
    for i in range(4):
        header.append(f"foot_force_est_{i}")

    return header


def _try_import_sdk2py(sdk2py_root: str | None):
    if sdk2py_root is not None and sdk2py_root.strip() != "":
        sys.path.insert(0, str(Path(sdk2py_root).expanduser().resolve()))

    try:
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    except Exception as exc:
        raise RuntimeError(
            "Failed to import unitree_sdk2py. Install with `pip install -e <unitree_sdk2_python>` "
            "or pass --sdk2py_root <path_to_unitree_sdk2_python>."
        ) from exc

    return ChannelSubscriber, ChannelFactoryInitialize, LowState_


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 raw logger via unitree_sdk2_python (no ROS2 required).")
    parser.add_argument("--interface", type=str, required=True, help="Robot network interface (e.g., enp3s0).")
    parser.add_argument("--output_csv", type=str, default="./go2_full_log_sdk2py.csv")
    parser.add_argument("--log_hz", type=float, default=500.0, help="Target logging rate. 0 = log every callback.")
    parser.add_argument("--duration_s", type=float, default=0.0, help="Run duration. <=0 means until Ctrl+C.")
    parser.add_argument("--queue_len", type=int, default=64, help="DDS callback queue length.")
    parser.add_argument("--flush_every_n", type=int, default=100)
    parser.add_argument("--print_every_s", type=float, default=1.0)
    parser.add_argument("--cmd_vx", type=float, default=0.0, help="Static vx command metadata in log.")
    parser.add_argument("--cmd_vy", type=float, default=0.0, help="Static vy command metadata in log.")
    parser.add_argument("--cmd_wz", type=float, default=0.0, help="Static wz command metadata in log.")
    parser.add_argument("--cmd_source_id", type=str, default="unknown", help="Metadata tag: teleop/manual/replay.")
    parser.add_argument(
        "--sdk2py_root",
        type=str,
        default="",
        help="Path to cloned unitree_sdk2_python repo root (optional if already pip-installed).",
    )
    args = parser.parse_args()

    ChannelSubscriber, ChannelFactoryInitialize, LowState_ = _try_import_sdk2py(args.sdk2py_root)

    out_path = Path(args.output_csv).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = _build_header()
    running = True

    def _sig_handler(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print("[INFO] Initializing DDS channel factory...")
    ChannelFactoryInitialize(0, args.interface)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()

        logger = _LowStateCsvLogger(
            writer=writer,
            cmd_vx=float(args.cmd_vx),
            cmd_vy=float(args.cmd_vy),
            cmd_wz=float(args.cmd_wz),
            cmd_source_id=str(args.cmd_source_id),
            log_hz=float(args.log_hz),
            flush_every_n=int(args.flush_every_n),
            print_every_s=float(args.print_every_s),
        )

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(logger.on_lowstate, int(args.queue_len))

        print(f"[INFO] Logging started -> {out_path}")
        print(
            f"[INFO] interface={args.interface}, log_hz={args.log_hz}, "
            f"duration_s={args.duration_s}, queue_len={args.queue_len}"
        )

        start_mono = time.monotonic()
        try:
            while running:
                time.sleep(0.2)
                if logger.rows > 0 and (logger.rows % max(int(args.flush_every_n), 1) == 0):
                    f.flush()
                if float(args.duration_s) > 0.0 and (time.monotonic() - start_mono) >= float(args.duration_s):
                    running = False
        finally:
            try:
                sub.Close()
            except Exception:
                pass
            f.flush()

    elapsed = max(time.monotonic() - start_mono, 1e-6)
    print(
        f"[DONE] rows={logger.rows}, callbacks={logger.callbacks}, gated={logger.gated}, "
        f"elapsed_s={elapsed:.2f}, avg_log_hz={logger.rows/elapsed:.2f}"
    )
    print(f"[DONE] CSV saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
