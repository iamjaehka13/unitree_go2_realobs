#!/usr/bin/env python3
from __future__ import annotations

"""
Run a 50Hz thermal-voltage governor from real Go2 packets via UDP bridge.

Expected bridge packets (CSV over UDP):
  ts,temp_max_c,vpack_v,vcell_min_v,wz_actual,estop,mode

Command packets sent to bridge (CSV over UDP):
  vx,vy,wz,stop
"""

import argparse
import csv
import json
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _load_governor_classes():
    try:
        from replay_utils import GovernorConfig, ThermalVoltageGovernor

        return GovernorConfig, ThermalVoltageGovernor
    except Exception as exc:  # pragma: no cover - fallback for minimal envs
        print(
            f"[WARN] replay_utils import failed ({exc}); using local governor fallback.",
            file=sys.stderr,
        )
        return _FallbackGovernorConfig, _FallbackThermalVoltageGovernor


def _load_replay_helpers():
    try:
        from replay_utils import load_replay_schedule
    except Exception as exc:
        raise RuntimeError(
            "Failed to import replay schedule loader. Install dependencies for "
            "`unitree_go2_phm/scripts/rsl_rl/replay_utils.py` (notably numpy, optional pyyaml)."
        ) from exc
    return load_replay_schedule


@dataclass
class Packet500Hz:
    ts: float
    temp_12: list[float]
    vpack: float
    vcell_8: list[float]
    vx_cmd: float
    vy_cmd: float
    wz_cmd: float
    wz_actual: float = 0.0
    estop: int = 0
    mode: int = 0


def _parse_state_packet(raw: bytes) -> Packet500Hz | None:
    # ts,temp_max_c,vpack_v,vcell_min_v,wz_actual,estop,mode
    text = raw.decode("utf-8", errors="ignore").strip()
    if text == "":
        return None
    cols = text.split(",")
    if len(cols) < 7:
        return None
    try:
        ts = float(cols[0])
        temp_max_c = float(cols[1])
        vpack_v = float(cols[2])
        vcell_min_v = float(cols[3])
        wz_actual = float(cols[4])
        estop = int(float(cols[5]))
        mode = int(float(cols[6]))
    except ValueError:
        return None
    return Packet500Hz(
        ts=ts,
        temp_12=[temp_max_c] * 12,
        vpack=vpack_v,
        vcell_8=[vcell_min_v] * 8,
        vx_cmd=0.0,
        vy_cmd=0.0,
        wz_cmd=0.0,
        wz_actual=wz_actual,
        estop=estop,
        mode=mode,
    )


def aggregate_window(window_packets: list[Packet500Hz]) -> dict[str, float]:
    if len(window_packets) == 0:
        return {
            "temp_max_c": 25.0,
            "vpack_v": 33.6,
            "vcell_min_v": 4.2,
            "vx_cmd_raw": 0.0,
            "vy_cmd_raw": 0.0,
            "wz_cmd_raw": 0.0,
            "wz_actual": 0.0,
            "estop": 0,
            "mode": 0,
        }

    p_last = window_packets[-1]
    temp_max_c = max(max(p.temp_12) for p in window_packets)
    vcell_min_v = min(min(p.vcell_8) for p in window_packets)
    vpack_v = p_last.vpack
    return {
        "temp_max_c": float(temp_max_c),
        "vpack_v": float(vpack_v),
        "vcell_min_v": float(vcell_min_v),
        "vx_cmd_raw": float(p_last.vx_cmd),
        "vy_cmd_raw": float(p_last.vy_cmd),
        "wz_cmd_raw": float(p_last.wz_cmd),
        "wz_actual": float(p_last.wz_actual),
        "estop": int(p_last.estop),
        "mode": int(p_last.mode),
    }


def classify_stop_reason(hard_stop: bool, dbg: dict[str, float], estop: int) -> str:
    if estop > 0:
        return "estop"
    if not hard_stop:
        return ""
    tags: list[str] = []
    if float(dbg.get("hard_stop_temp", 0.0)) > 0.5:
        tags.append("temp")
    if float(dbg.get("hard_stop_pack", 0.0)) > 0.5:
        tags.append("pack")
    if float(dbg.get("hard_stop_cell", 0.0)) > 0.5:
        tags.append("cell")
    if len(tags) == 0:
        return "governor_hard_stop_unknown"
    return "governor_hard_stop_" + "+".join(tags)


def write_summary(
    rows: list[dict[str, Any]],
    out_json: Path,
    dt: float,
    temp_warn_c: float,
    stop_reason_override: str = "",
) -> None:
    if len(rows) == 0:
        summary = {
            "num_steps": 0,
            "duration_s": 0.0,
            "hard_stop": False,
            "stop_reasons": ({stop_reason_override: 1} if stop_reason_override != "" else {}),
        }
        out_json.write_text(json.dumps(summary, indent=2))
        return

    over_warn = sum(1 for r in rows if float(r["temp_max_c"]) >= float(temp_warn_c))
    vpack_min = min(float(r["vpack_v"]) for r in rows)
    vcell_min = min(float(r["vcell_min_v"]) for r in rows)
    scale_mean = sum(float(r["scale_lin"]) for r in rows) / max(len(rows), 1)
    hard_stop = any(bool(r["hard_stop"]) for r in rows)
    reasons: dict[str, int] = {}
    for r in rows:
        reason = str(r.get("stop_reason", ""))
        if reason == "":
            continue
        reasons[reason] = reasons.get(reason, 0) + 1

    summary = {
        "num_steps": len(rows),
        "duration_s": float(len(rows) * dt),
        "time_temp_over_warn_s": float(over_warn * dt),
        "vpack_min_v": float(vpack_min),
        "vcell_min_v": float(vcell_min),
        "mean_scale_lin": float(scale_mean),
        "hard_stop": bool(hard_stop),
        "stop_reasons": reasons,
    }
    out_json.write_text(json.dumps(summary, indent=2))


def _build_governor(args: argparse.Namespace) -> Any:
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
    return gov


def _command_from_schedule(
    elapsed_s: float,
    schedule: Any | None,
    hold_last_cmd: bool,
    default_cmd: tuple[float, float, float],
) -> tuple[float, float, float]:
    if schedule is None:
        return default_cmd
    idx = int(max(0.0, elapsed_s) / max(float(schedule.dt), 1e-6))
    if idx >= int(schedule.num_steps):
        if hold_last_cmd and int(schedule.num_steps) > 0:
            idx = int(schedule.num_steps) - 1
        else:
            return (0.0, 0.0, 0.0)
    cmd = schedule.commands[idx]
    return float(cmd[0]), float(cmd[1]), float(cmd[2])


def _send_bridge_cmd(sock: socket.socket, host: str, port: int, vx: float, vy: float, wz: float, stop: bool) -> None:
    payload = f"{vx:.6f},{vy:.6f},{wz:.6f},{1 if stop else 0}".encode("utf-8")
    sock.sendto(payload, (host, port))


def _poll_state_packets(sock: socket.socket) -> list[Packet500Hz]:
    packets: list[Packet500Hz] = []
    while True:
        try:
            raw, _addr = sock.recvfrom(4096)
        except BlockingIOError:
            break
        pkt = _parse_state_packet(raw)
        if pkt is not None:
            packets.append(pkt)
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(description="Live governor runner via SDK2 UDP bridge.")
    parser.add_argument("--out_dir", type=str, default="./real_runs")
    parser.add_argument("--duration_s", type=float, default=120.0)
    parser.add_argument("--loop_hz", type=float, default=50.0)
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
    parser.add_argument("--state_host", type=str, default="127.0.0.1")
    parser.add_argument("--state_port", type=int, default=17001)
    parser.add_argument("--cmd_host", type=str, default="127.0.0.1")
    parser.add_argument("--cmd_port", type=int, default=17002)
    parser.add_argument("--state_timeout_s", type=float, default=1.0)
    parser.add_argument("--command_file", type=str, default="", help="Replay command file (.yaml/.json/.csv)")
    parser.add_argument("--replay_dt", type=float, default=0.02)
    parser.add_argument("--default_vx", type=float, default=0.0)
    parser.add_argument("--default_vy", type=float, default=0.0)
    parser.add_argument("--default_wz", type=float, default=0.0)
    parser.add_argument("--no_hold_last_cmd", action="store_true", default=False)
    parser.add_argument(
        "--bridge_cmd",
        type=str,
        default="",
        help="Optional command to launch bridge process (example: './go2_udp_bridge enp3s0').",
    )
    parser.add_argument("--bridge_startup_wait_s", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step_csv = out_dir / "steps.csv"
    summary_json = out_dir / "summary.json"

    dt = 1.0 / max(float(args.loop_hz), 1e-6)
    governor = _build_governor(args)

    schedule = None
    if args.command_file.strip() != "":
        load_replay_schedule = _load_replay_helpers()
        schedule = load_replay_schedule(args.command_file, default_dt=float(args.replay_dt))

    state_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_sock.setblocking(False)
    state_sock.bind((args.state_host, int(args.state_port)))

    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.setblocking(False)

    bridge_proc: subprocess.Popen[str] | None = None
    if args.bridge_cmd.strip() != "":
        bridge_proc = subprocess.Popen(shlex.split(args.bridge_cmd))
        time.sleep(max(float(args.bridge_startup_wait_s), 0.0))

    rows: list[dict[str, Any]] = []
    started = time.time()
    next_tick = started
    last_state_rx = started
    last_packet: Packet500Hz | None = None
    received_any_state = False
    hold_last_cmd = not bool(args.no_hold_last_cmd)
    default_cmd = (float(args.default_vx), float(args.default_vy), float(args.default_wz))
    stop_reason_override = ""

    try:
        with step_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "t_s",
                    "temp_max_c",
                    "vpack_v",
                    "vcell_min_v",
                    "vx_cmd_raw",
                    "vy_cmd_raw",
                    "wz_cmd_raw",
                    "vx_cmd_exec",
                    "vy_cmd_exec",
                    "wz_cmd_exec",
                    "scale_lin",
                    "scale_yaw",
                    "hard_stop",
                    "stop_reason",
                    "estop",
                    "mode",
                ],
            )
            writer.writeheader()

            while True:
                now = time.time()
                elapsed = now - started
                if elapsed >= float(args.duration_s):
                    break

                packets = _poll_state_packets(state_sock)
                if len(packets) > 0:
                    last_state_rx = now
                    last_packet = packets[-1]
                    received_any_state = True
                elif last_packet is not None:
                    packets = [last_packet]

                if now - last_state_rx > float(args.state_timeout_s):
                    stop_reason_override = "state_timeout"
                    break

                if not received_any_state:
                    # Never drive before the first valid bridge state arrives.
                    _send_bridge_cmd(
                        sock=cmd_sock,
                        host=args.cmd_host,
                        port=int(args.cmd_port),
                        vx=0.0,
                        vy=0.0,
                        wz=0.0,
                        stop=True,
                    )
                    next_tick += dt
                    sleep_s = next_tick - time.time()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    continue

                vx_ref, vy_ref, wz_ref = _command_from_schedule(
                    elapsed_s=elapsed,
                    schedule=schedule,
                    hold_last_cmd=hold_last_cmd,
                    default_cmd=default_cmd,
                )
                for p in packets:
                    p.vx_cmd = vx_ref
                    p.vy_cmd = vy_ref
                    p.wz_cmd = wz_ref

                agg = aggregate_window(packets)
                scale_lin, scale_yaw, hard_stop, dbg = governor.step(
                    dt=dt,
                    temp_max_c=float(agg["temp_max_c"]),
                    cell_v_min=float(agg["vcell_min_v"]),
                    pack_v=float(agg["vpack_v"]),
                )
                vx_exec = float(agg["vx_cmd_raw"]) * scale_lin
                vy_exec = float(agg["vy_cmd_raw"]) * scale_lin
                wz_exec = float(agg["wz_cmd_raw"]) * scale_yaw
                stop_reason = classify_stop_reason(hard_stop, dbg, int(agg["estop"]))

                _send_bridge_cmd(
                    sock=cmd_sock,
                    host=args.cmd_host,
                    port=int(args.cmd_port),
                    vx=vx_exec,
                    vy=vy_exec,
                    wz=wz_exec,
                    stop=(stop_reason != ""),
                )

                row = {
                    "t_s": round(elapsed, 4),
                    "temp_max_c": round(float(agg["temp_max_c"]), 4),
                    "vpack_v": round(float(agg["vpack_v"]), 4),
                    "vcell_min_v": round(float(agg["vcell_min_v"]), 4),
                    "vx_cmd_raw": round(float(agg["vx_cmd_raw"]), 4),
                    "vy_cmd_raw": round(float(agg["vy_cmd_raw"]), 4),
                    "wz_cmd_raw": round(float(agg["wz_cmd_raw"]), 4),
                    "vx_cmd_exec": round(float(vx_exec), 4),
                    "vy_cmd_exec": round(float(vy_exec), 4),
                    "wz_cmd_exec": round(float(wz_exec), 4),
                    "scale_lin": round(float(scale_lin), 4),
                    "scale_yaw": round(float(scale_yaw), 4),
                    "hard_stop": int(bool(hard_stop)),
                    "stop_reason": stop_reason,
                    "estop": int(agg["estop"]),
                    "mode": int(agg["mode"]),
                }
                writer.writerow(row)
                rows.append(row)

                if stop_reason != "":
                    break

                next_tick += dt
                sleep_s = next_tick - time.time()
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        # Always send a final stop packet.
        for _ in range(3):
            try:
                _send_bridge_cmd(
                    sock=cmd_sock,
                    host=args.cmd_host,
                    port=int(args.cmd_port),
                    vx=0.0,
                    vy=0.0,
                    wz=0.0,
                    stop=True,
                )
            except OSError:
                pass
            time.sleep(0.01)
        state_sock.close()
        cmd_sock.close()
        if bridge_proc is not None:
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()

    if stop_reason_override != "" and len(rows) > 0 and rows[-1]["stop_reason"] == "":
        rows[-1]["stop_reason"] = stop_reason_override

    write_summary(
        rows,
        summary_json,
        dt=dt,
        temp_warn_c=float(args.temp_warn_c),
        stop_reason_override=stop_reason_override,
    )
    print(f"[DONE] Step log: {step_csv}")
    print(f"[DONE] Summary : {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
