#!/usr/bin/env python3
from __future__ import annotations

"""
Convert a rosbag2 directory containing `unitree_go/msg/LowState` into:

1) raw replay-friendly CSV compatible with `log_to_replay_csv.py`
2) optional 50 Hz replay CSV used by offline governor evaluation

Requires the optional `rosbags` dependency:

  python3 -m pip install --user rosbags
"""

import argparse
import csv
import re
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path


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

LOWSTATE_TYPENAME = "unitree_go/msg/LowState"

_BMS_STATE_MSG = """uint8 version_high
uint8 version_low
uint8 status
uint8 soc
int32 current
uint16 cycle
uint8[2] bq_ntc
uint8[2] mcu_ntc
uint16[15] cell_vol
"""

_IMU_STATE_MSG = """float32[4] quaternion
float32[3] gyroscope
float32[3] accelerometer
float32[3] rpy
uint8 temperature
"""

_MOTOR_STATE_MSG = """uint8 mode
float32 q
float32 dq
float32 ddq
float32 tau_est
float32 q_raw
float32 dq_raw
float32 ddq_raw
uint8 temperature
uint32 lost
uint32[2] reserve
"""

_LOWSTATE_MSG = """uint8[2] head
uint8 level_flag
uint8 frame_reserve
uint32[2] sn
uint32[2] version
uint16 bandwidth
unitree_go/msg/IMUState imu_state
unitree_go/msg/MotorState[20] motor_state
unitree_go/msg/BmsState bms_state
int16[4] foot_force
int16[4] foot_force_est
uint32 tick
uint8[40] wireless_remote
uint8 bit_flag
float32 adc_reel
uint8 temperature_ntc1
uint8 temperature_ntc2
float32 power_v
float32 power_a
uint16[4] fan_frequency
uint32 reserve
uint32 crc
"""

_CMD_PATH_RE = re.compile(
    r"cmd[_-]?x(?P<vx>[-+]?\d+(?:\.\d+)?)"
    r",?y(?P<vy>[-+]?\d+(?:\.\d+)?)"
    r",?z(?P<wz>[-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


@dataclass
class ReplayBucket:
    gid: int
    t_s: float
    vx: float
    vy: float
    yaw_rate: float
    temp_max_c: float
    vpack_v: float
    vcell_min_v: float

    def update(
        self,
        *,
        t_s: float,
        vx: float,
        vy: float,
        yaw_rate: float,
        temp_max_c: float,
        vpack_v: float,
        vcell_min_v: float,
    ) -> None:
        self.t_s = float(t_s)
        self.vx = float(vx)
        self.vy = float(vy)
        self.yaw_rate = float(yaw_rate)
        self.temp_max_c = max(float(self.temp_max_c), float(temp_max_c))
        self.vpack_v = float(vpack_v)
        self.vcell_min_v = min(float(self.vcell_min_v), float(vcell_min_v))

    def to_row(self) -> dict[str, float]:
        return {
            "t": float(self.t_s),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "yaw_rate": float(self.yaw_rate),
            "temp_max_c": float(self.temp_max_c),
            "vpack_v": float(self.vpack_v),
            "vcell_min_v": float(self.vcell_min_v),
        }


def _cell_raw_to_volt(raw: int) -> float:
    if raw <= 0:
        return 0.0
    if raw > 100:
        return float(raw) * 1e-3
    return float(raw)


def _safe_seq_value(seq, idx: int, default: float = 0.0):
    try:
        return seq[idx]
    except Exception:
        return default


def _build_raw_header() -> list[str]:
    header: list[str] = []
    header += ["ts", "time_s", "tick", "level_flag", "bit_flag"]
    header += ["vx_cmd", "vy_cmd", "wz_cmd", "cmd_source_id"]

    header += [
        "imu_qw",
        "imu_qx",
        "imu_qy",
        "imu_qz",
        "imu_gyro_x",
        "imu_gyro_y",
        "imu_gyro_z",
        "imu_acc_x",
        "imu_acc_y",
        "imu_acc_z",
        "imu_rpy_x",
        "imu_rpy_y",
        "imu_rpy_z",
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


def _load_rosbags():
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    except Exception as exc:
        raise RuntimeError(
            "Missing optional dependency `rosbags`. "
            "Install with `python3 -m pip install --user rosbags`."
        ) from exc

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    for name, msgdef in (
        ("unitree_go/msg/BmsState", _BMS_STATE_MSG),
        ("unitree_go/msg/IMUState", _IMU_STATE_MSG),
        ("unitree_go/msg/MotorState", _MOTOR_STATE_MSG),
        (LOWSTATE_TYPENAME, _LOWSTATE_MSG),
    ):
        typestore.register(get_types_from_msg(msgdef, name))
    return Reader, typestore


def _resolve_bag_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".db3":
            raise FileNotFoundError(f"Expected a rosbag directory or .db3 file, got: {path}")
        path = path.parent

    metadata = path / "metadata.yaml"
    if metadata.exists():
        return path

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    candidates = sorted(child for child in path.iterdir() if child.is_dir() and (child / "metadata.yaml").exists())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise FileNotFoundError(
            f"Multiple rosbag directories found under {path}. Pass the exact one containing metadata.yaml."
        )
    raise FileNotFoundError(f"No metadata.yaml found under {path}")


def _infer_commands_from_path(path: Path) -> tuple[float, float, float] | None:
    match = _CMD_PATH_RE.search(str(path))
    if not match:
        lowered = str(path).lower()
        if "stand" in lowered:
            return (0.0, 0.0, 0.0)
        return None
    return (
        float(match.group("vx")),
        float(match.group("vy")),
        float(match.group("wz")),
    )


def _resolve_commands(
    bag_dir: Path,
    *,
    cmd_vx: float | None,
    cmd_vy: float | None,
    cmd_wz: float | None,
    infer_from_path: bool,
) -> tuple[float, float, float, str]:
    inferred = _infer_commands_from_path(bag_dir) if infer_from_path else None
    vx = float(cmd_vx) if cmd_vx is not None else float(inferred[0]) if inferred else 0.0
    vy = float(cmd_vy) if cmd_vy is not None else float(inferred[1]) if inferred else 0.0
    wz = float(cmd_wz) if cmd_wz is not None else float(inferred[2]) if inferred else 0.0
    source = "path_inferred" if inferred and cmd_vx is None and cmd_vy is None and cmd_wz is None else "rosbag2"
    return vx, vy, wz, source


def _build_raw_row(
    *,
    timestamp_ns: int,
    rel_time_s: float,
    msg,
    cmd_vx: float,
    cmd_vy: float,
    cmd_wz: float,
    cmd_source_id: str,
) -> list[object]:
    imu = msg.imu_state
    bms = msg.bms_state
    motors = msg.motor_state
    foot_force = msg.foot_force
    foot_force_est = msg.foot_force_est

    row: list[object] = [
        float(timestamp_ns) * 1e-9,
        float(rel_time_s),
        int(msg.tick),
        int(msg.level_flag),
        int(msg.bit_flag),
        float(cmd_vx),
        float(cmd_vy),
        float(cmd_wz),
        cmd_source_id,
    ]

    row += [
        float(_safe_seq_value(imu.quaternion, 0, 0.0)),
        float(_safe_seq_value(imu.quaternion, 1, 0.0)),
        float(_safe_seq_value(imu.quaternion, 2, 0.0)),
        float(_safe_seq_value(imu.quaternion, 3, 1.0)),
        float(_safe_seq_value(imu.gyroscope, 0, 0.0)),
        float(_safe_seq_value(imu.gyroscope, 1, 0.0)),
        float(_safe_seq_value(imu.gyroscope, 2, 0.0)),
        float(_safe_seq_value(imu.accelerometer, 0, 0.0)),
        float(_safe_seq_value(imu.accelerometer, 1, 0.0)),
        float(_safe_seq_value(imu.accelerometer, 2, 0.0)),
        float(_safe_seq_value(imu.rpy, 0, 0.0)),
        float(_safe_seq_value(imu.rpy, 1, 0.0)),
        float(_safe_seq_value(imu.rpy, 2, 0.0)),
        int(imu.temperature),
    ]

    row += [
        float(msg.power_v),
        float(msg.power_a),
        int(bms.soc),
        int(bms.status),
        int(bms.current),
        int(bms.cycle),
    ]

    cell_raw_8: list[int] = []
    for i in range(8):
        raw_i = int(_safe_seq_value(bms.cell_vol, i, 0))
        row.append(_cell_raw_to_volt(raw_i))
        cell_raw_8.append(raw_i)
    for raw_i in cell_raw_8:
        row.append(int(raw_i))

    motor_temp_12: list[float] = []
    for i in range(12):
        motor = _safe_seq_value(motors, i, None)
        if motor is None:
            mode = 0
            q = 0.0
            dq = 0.0
            ddq = 0.0
            tau_est = 0.0
            temp = 0
            lost = 0
        else:
            mode = int(motor.mode)
            q = float(motor.q)
            dq = float(motor.dq)
            ddq = float(motor.ddq)
            tau_est = float(motor.tau_est)
            temp = int(motor.temperature)
            lost = int(motor.lost)
        row += [mode, q, dq, ddq, tau_est, temp, lost]
        motor_temp_12.append(float(temp))

    for temp in motor_temp_12:
        row.append(float(temp))
    for temp in motor_temp_12:
        row.append(float(temp))
    for temp in motor_temp_12:
        row.append(float(temp))

    for i in range(4):
        row.append(int(_safe_seq_value(foot_force, i, 0)))
    for i in range(4):
        row.append(int(_safe_seq_value(foot_force_est, i, 0)))

    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert rosbag2 LowState logs to raw/replay CSV.")
    parser.add_argument("--input_bag", type=str, required=True, help="rosbag directory or a .db3 file path.")
    parser.add_argument("--topic", type=str, default="/lowstate")
    parser.add_argument("--output_raw_csv", type=str, default="", help="Optional raw CSV output path.")
    parser.add_argument("--output_replay_csv", type=str, default="", help="Optional 50Hz replay CSV output path.")
    parser.add_argument("--window_ms", type=int, default=20, help="Replay aggregation window in ms.")
    parser.add_argument("--cmd_vx", type=float, default=None, help="Command metadata for exported rows.")
    parser.add_argument("--cmd_vy", type=float, default=None, help="Command metadata for exported rows.")
    parser.add_argument("--cmd_wz", type=float, default=None, help="Command metadata for exported rows.")
    parser.add_argument(
        "--infer_command_from_path",
        action="store_true",
        default=False,
        help="Infer cmd_vx/cmd_vy/cmd_wz from directory names like cmd_x1,y0,z-1.",
    )
    parser.add_argument("--progress_every", type=int, default=50000)
    parser.add_argument("--max_messages", type=int, default=0, help="Debug limit. <=0 means all messages.")
    args = parser.parse_args()

    if not args.output_raw_csv and not args.output_replay_csv:
        raise ValueError("At least one of --output_raw_csv or --output_replay_csv must be set.")
    if int(args.window_ms) <= 0:
        raise ValueError("--window_ms must be > 0.")

    bag_dir = _resolve_bag_dir(args.input_bag)
    Reader, typestore = _load_rosbags()
    cmd_vx, cmd_vy, cmd_wz, cmd_source_id = _resolve_commands(
        bag_dir,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_wz=args.cmd_wz,
        infer_from_path=bool(args.infer_command_from_path),
    )

    raw_path = Path(args.output_raw_csv).expanduser().resolve() if args.output_raw_csv else None
    replay_path = Path(args.output_replay_csv).expanduser().resolve() if args.output_replay_csv else None
    if raw_path:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
    if replay_path:
        replay_path.parent.mkdir(parents=True, exist_ok=True)

    raw_writer = None
    replay_writer = None
    current_bucket: ReplayBucket | None = None
    t0_ns: int | None = None
    message_count = 0
    replay_rows = 0
    last_timestamp_ns: int | None = None
    window_ns = int(args.window_ms) * 1_000_000

    with ExitStack() as stack:
        if raw_path:
            raw_file = stack.enter_context(raw_path.open("w", newline=""))
            raw_writer = csv.writer(raw_file)
            raw_writer.writerow(_build_raw_header())

        if replay_path:
            replay_file = stack.enter_context(replay_path.open("w", newline=""))
            replay_writer = csv.DictWriter(
                replay_file,
                fieldnames=["t", "vx", "vy", "yaw_rate", "temp_max_c", "vpack_v", "vcell_min_v"],
            )
            replay_writer.writeheader()

        with Reader(bag_dir) as reader:
            connections = [c for c in reader.connections if c.topic == args.topic and c.msgtype == LOWSTATE_TYPENAME]
            if not connections:
                available = sorted((c.topic, c.msgtype) for c in reader.connections)
                raise RuntimeError(
                    f"No {LOWSTATE_TYPENAME} messages found on topic {args.topic}. Available: {available}"
                )

            for connection, timestamp_ns, rawdata in reader.messages(connections=connections):
                message_count += 1
                if args.max_messages > 0 and message_count > int(args.max_messages):
                    break

                if t0_ns is None:
                    t0_ns = int(timestamp_ns)
                last_timestamp_ns = int(timestamp_ns)
                rel_time_s = (int(timestamp_ns) - int(t0_ns)) * 1e-9
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)

                if raw_writer is not None:
                    raw_writer.writerow(
                        _build_raw_row(
                            timestamp_ns=int(timestamp_ns),
                            rel_time_s=rel_time_s,
                            msg=msg,
                            cmd_vx=cmd_vx,
                            cmd_vy=cmd_vy,
                            cmd_wz=cmd_wz,
                            cmd_source_id=cmd_source_id,
                        )
                    )

                if replay_writer is not None:
                    motor_temp_12 = [float(_safe_seq_value(msg.motor_state, i, None).temperature) for i in range(12)]
                    vcell_min_v = min(
                        _cell_raw_to_volt(int(_safe_seq_value(msg.bms_state.cell_vol, i, 0))) for i in range(8)
                    )
                    bucket_gid = (int(timestamp_ns) - int(t0_ns)) // window_ns
                    temp_max_c = max(motor_temp_12) if motor_temp_12 else 0.0

                    if current_bucket is None or int(bucket_gid) != int(current_bucket.gid):
                        if current_bucket is not None:
                            replay_writer.writerow(current_bucket.to_row())
                            replay_rows += 1
                        current_bucket = ReplayBucket(
                            gid=int(bucket_gid),
                            t_s=float(rel_time_s),
                            vx=float(cmd_vx),
                            vy=float(cmd_vy),
                            yaw_rate=float(cmd_wz),
                            temp_max_c=float(temp_max_c),
                            vpack_v=float(msg.power_v),
                            vcell_min_v=float(vcell_min_v),
                        )
                    else:
                        current_bucket.update(
                            t_s=float(rel_time_s),
                            vx=float(cmd_vx),
                            vy=float(cmd_vy),
                            yaw_rate=float(cmd_wz),
                            temp_max_c=float(temp_max_c),
                            vpack_v=float(msg.power_v),
                            vcell_min_v=float(vcell_min_v),
                        )

                if int(args.progress_every) > 0 and (message_count % int(args.progress_every)) == 0:
                    print(
                        f"[INFO] processed={message_count} "
                        f"rel_t_s={rel_time_s:.2f} power_v={float(msg.power_v):.2f}"
                    )

        if replay_writer is not None and current_bucket is not None:
            replay_writer.writerow(current_bucket.to_row())
            replay_rows += 1

    if message_count == 0:
        raise RuntimeError(f"No messages were read from {bag_dir} on topic {args.topic}.")

    duration_s = ((int(last_timestamp_ns) - int(t0_ns)) * 1e-9) if last_timestamp_ns is not None and t0_ns is not None else 0.0
    print(f"[DONE] bag_dir={bag_dir}")
    print(f"[DONE] messages={message_count}, duration_s={duration_s:.3f}")
    print(f"[DONE] cmd_vx={cmd_vx}, cmd_vy={cmd_vy}, cmd_wz={cmd_wz}, cmd_source_id={cmd_source_id}")
    if raw_path:
        print(f"[DONE] raw_csv={raw_path}")
    if replay_path:
        print(f"[DONE] replay_csv={replay_path}, replay_rows={replay_rows}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
