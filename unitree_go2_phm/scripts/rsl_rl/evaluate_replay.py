"""Replay-based PHM evaluation with optional governor and fault injection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from replay_utils import (
    GovernorConfig,
    ReplaySchedule,
    ThermalVoltageGovernor,
    leg_joint_names,
    load_replay_schedule,
)


parser = argparse.ArgumentParser(description="Replay evaluation with PHM governor/fault injection.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--command_file", type=str, required=True, help="Replay command file (.csv/.json/.yaml)")
parser.add_argument("--output_dir", type=str, default="./replay_results", help="Output directory")
parser.add_argument("--num_trials", type=int, default=3, help="Number of repeated trials")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (recommended: 1)")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point")
parser.add_argument("--replay_dt", type=float, default=0.02, help="Replay command dt if source file is not explicit")
parser.add_argument("--max_steps", type=int, default=None, help="Optional step cap per trial")
parser.add_argument("--soc_init", type=float, default=None, help="Optional initial SOC override [0,1]")
parser.add_argument("--real-time", action="store_true", default=False, help="Sleep to match env step dt")

parser.add_argument("--governor", action="store_true", default=False, help="Enable thermal/voltage governor")
parser.add_argument("--temp_warn_c", type=float, default=65.0)
parser.add_argument("--temp_crit_c", type=float, default=70.0)
parser.add_argument("--temp_stop_c", type=float, default=75.0)
parser.add_argument("--cell_warn_v", type=float, default=3.20)
parser.add_argument("--cell_stop_v", type=float, default=3.05, help="Cell-voltage scale stop point (soft)")
parser.add_argument("--cell_hard_stop_v", type=float, default=3.00, help="Cell-voltage hard stop threshold")
parser.add_argument("--pack_stop_v", type=float, default=24.5, help="Pack-voltage hard stop threshold")
parser.add_argument("--temp_pred_horizon_s", type=float, default=1.0)
parser.add_argument("--temp_tau_s", type=float, default=1.0)
parser.add_argument("--volt_tau_s", type=float, default=0.2)
parser.add_argument("--yaw_exp", type=float, default=1.5)
parser.add_argument(
    "--temp_signal",
    type=str,
    default="auto",
    choices=["auto", "case", "coil"],
    help="Temperature signal for governor: case preferred, auto falls back to coil.",
)
parser.add_argument(
    "--coil_to_case_delta_c",
    type=float,
    default=5.0,
    help="When temp_signal=auto and case temperature is unavailable, estimate case as coil-delta.",
)
parser.add_argument(
    "--risk_factor_fixed",
    type=float,
    default=1.0,
    help="Force risk_factor command to a constant value for deterministic replay (if command exists).",
)

parser.add_argument("--fault_leg", type=str, default="", help="One of FR/FL/RR/RL. Empty disables fault injection")
parser.add_argument("--fault_kp_scale", type=float, default=0.6, help="Kp scale for fault-injected leg joints")
parser.add_argument("--fault_kd_scale", type=float, default=1.0, help="Kd scale for fault-injected leg joints")
parser.add_argument(
    "--fault_start_s",
    type=float,
    default=0.0,
    help="Apply fault injection starting at this replay time (seconds).",
)

cli_args.add_rsl_rl_args(parser, include_checkpoint=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import unitree_go2_phm.tasks  # noqa: F401
from unitree_go2_phm.tasks.manager_based.unitree_go2_phm.phm.utils import compute_battery_voltage


def _resample_schedule(schedule: ReplaySchedule, target_dt: float) -> ReplaySchedule:
    if abs(schedule.dt - target_dt) < 1e-9:
        return schedule
    target_t = np.arange(0.0, schedule.duration_s + 1e-9, target_dt, dtype=np.float64)
    x = np.interp(target_t, schedule.t, schedule.commands[:, 0])
    y = np.interp(target_t, schedule.t, schedule.commands[:, 1])
    z = np.interp(target_t, schedule.t, schedule.commands[:, 2])
    commands = np.stack((x, y, z), axis=1)
    return ReplaySchedule(dt=target_dt, t=target_t, commands=commands)


def _set_velocity_command(base_env, vx: float, vy: float, yaw_rate: float):
    cmd_buf = base_env.command_manager.get_command("base_velocity")
    cmd_buf[:, 0] = float(vx)
    cmd_buf[:, 1] = float(vy)
    cmd_buf[:, 2] = float(yaw_rate)


def _set_risk_factor(base_env, risk_value: float):
    try:
        cmd = base_env.command_manager.get_command("risk_factor")
    except Exception:
        return
    if cmd is None:
        return
    cmd[...] = float(risk_value)


def _pack_voltage(base_env) -> float:
    phm = base_env.phm_state
    v = phm.battery_voltage_true if hasattr(phm, "battery_voltage_true") else phm.battery_voltage
    return float(v[0].detach().cpu().item())


def _cell_min_voltage(base_env) -> float:
    phm = base_env.phm_state
    for name in ("cell_voltage", "cell_voltages", "bms_cell_voltage", "bms_cell_voltages"):
        if not hasattr(phm, name):
            continue
        val = getattr(phm, name)
        if not isinstance(val, torch.Tensor):
            continue
        if val.ndim >= 2:
            return float(torch.min(val[0]).detach().cpu().item())
        if val.ndim == 1 and int(val.shape[0]) == int(base_env.num_envs):
            return float(val[0].detach().cpu().item())
    # Fallback for simulation states without explicit per-cell data.
    return _pack_voltage(base_env) / 8.0


def _max_joint_temp_from_tensor(value: torch.Tensor, env_index: int = 0) -> float:
    if value.ndim >= 2:
        return float(torch.max(value[env_index]).detach().cpu().item())
    if value.ndim == 1 and int(value.shape[0]) > env_index:
        return float(value[env_index].detach().cpu().item())
    return float(torch.max(value).detach().cpu().item())


def _temperature_for_governor(base_env, signal_mode: str, coil_to_case_delta_c: float) -> tuple[float, str]:
    phm = base_env.phm_state
    mode = signal_mode.strip().lower()

    def _try_case() -> tuple[float, str] | None:
        case_names = (
            "motor_case_temp",
            "case_temp",
            "motor_temp_case",
            "housing_temp",
            "motor_housing_temp",
        )
        for name in case_names:
            if not hasattr(phm, name):
                continue
            val = getattr(phm, name)
            if isinstance(val, torch.Tensor):
                return _max_joint_temp_from_tensor(val, env_index=0), name
        return None

    def _coil_value() -> float:
        if not hasattr(phm, "coil_temp"):
            raise RuntimeError("PHM state does not expose coil_temp.")
        val = getattr(phm, "coil_temp")
        if not isinstance(val, torch.Tensor):
            raise RuntimeError("PHM state coil_temp is not a tensor.")
        return _max_joint_temp_from_tensor(val, env_index=0)

    if mode == "coil":
        return _coil_value(), "coil_temp"

    case_hit = _try_case()
    if case_hit is not None:
        return case_hit

    if mode == "case":
        raise RuntimeError("Requested temp_signal=case but no case-temperature tensor was found in phm_state.")

    # auto fallback: estimate case-like value from coil hotspot.
    coil = _coil_value()
    case_est = coil - float(coil_to_case_delta_c)
    return float(case_est), "coil_temp_fallback"


def _set_initial_soc(base_env, soc_value: float):
    soc = float(max(0.0, min(1.0, soc_value)))
    phm = base_env.phm_state
    phm.soc[:] = soc
    v = compute_battery_voltage(phm.soc, torch.zeros_like(phm.soc))
    if hasattr(phm, "battery_voltage_true"):
        phm.battery_voltage_true[:] = v
    phm.battery_voltage[:] = v
    if hasattr(phm, "min_voltage_log"):
        phm.min_voltage_log[:] = v
    if hasattr(phm, "bms_voltage_pred"):
        phm.bms_voltage_pred[:] = v


def _apply_fault(base_env, fault_leg: str, kp_scale: float, kd_scale: float):
    if hasattr(base_env, "clear_external_fault_profile"):
        base_env.clear_external_fault_profile()
    if fault_leg.strip() == "":
        return
    if not hasattr(base_env, "set_external_fault_profile"):
        raise RuntimeError("Environment does not expose set_external_fault_profile().")
    base_env.set_external_fault_profile(
        joint_names=leg_joint_names(fault_leg),
        kp_scale=float(kp_scale),
        kd_scale=float(kd_scale),
    )


def _write_step_csv(path: Path, rows: list[dict[str, Any]]):
    if len(rows) == 0:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _obs_from_reset_output(reset_output):
    """Normalize Gymnasium reset output across API versions."""
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _safe_reset_recurrent(policy_nn, dones: torch.Tensor):
    """Reset recurrent state only when the policy exposes a reset method."""
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(dones)


def _summarize_rows(
    rows: list[dict[str, float]],
    dt: float,
    completed_script: bool,
    stop_reason: str,
    temp_warn_c: float,
) -> dict[str, Any]:
    if len(rows) == 0:
        return {
            "steps": 0,
            "duration_s": 0.0,
            "completed_script": False,
            "stop_reason": stop_reason or "empty",
        }

    yaw_err_exec = np.array([abs(r["wz_actual"] - r["wz_cmd_exec"]) for r in rows], dtype=np.float64)
    yaw_err_raw = np.array([abs(r["wz_actual"] - r["wz_cmd_raw"]) for r in rows], dtype=np.float64)
    temp = np.array([r["temp_max_c"] for r in rows], dtype=np.float64)
    vpack = np.array([r["vpack_v"] for r in rows], dtype=np.float64)
    vcell = np.array([r["vcell_min_v"] for r in rows], dtype=np.float64)
    power = np.array([r["power_w"] for r in rows], dtype=np.float64)
    scale = np.array([r["scale_lin"] for r in rows], dtype=np.float64)

    return {
        "steps": int(len(rows)),
        "duration_s": float(len(rows) * dt),
        "completed_script": bool(completed_script),
        "stop_reason": stop_reason,
        "yaw_mae_exec": float(np.mean(yaw_err_exec)),
        "yaw_mae_raw": float(np.mean(yaw_err_raw)),
        "temp_max_c": float(np.max(temp)),
        "time_temp_over_warn_s": float(np.sum(temp >= float(temp_warn_c)) * dt),
        "vpack_min_v": float(np.min(vpack)),
        "vcell_min_v": float(np.min(vcell)),
        "energy_j": float(np.sum(power) * dt),
        "mean_power_w": float(np.mean(power)),
        "mean_scale_lin": float(np.mean(scale)),
        "time_scale_lt_0p9_s": float(np.sum(scale < 0.9) * dt),
    }


def _aggregate_trial_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if len(summaries) == 0:
        return {}

    aggregate: dict[str, Any] = {"num_trials": len(summaries)}
    numeric_keys: list[str] = []
    for key, value in summaries[0].items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float, np.floating)):
            numeric_keys.append(key)
    for key in numeric_keys:
        values = np.array([float(s[key]) for s in summaries], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    completed = [bool(s.get("completed_script", False)) for s in summaries]
    aggregate["completed_rate"] = float(np.mean(np.array(completed, dtype=np.float64)))
    reasons: dict[str, int] = {}
    for s in summaries:
        reason = str(s.get("stop_reason", ""))
        reasons[reason] = reasons.get(reason, 0) + 1
    aggregate["stop_reasons"] = reasons
    return aggregate


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.seed = int(args_cli.seed)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    if env.num_envs != 1:
        raise ValueError(
            f"Replay evaluation requires num_envs=1 for unambiguous metrics, got num_envs={env.num_envs}."
        )

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=base_env.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    schedule = load_replay_schedule(args_cli.command_file, default_dt=float(args_cli.replay_dt))
    schedule = _resample_schedule(schedule, target_dt=float(base_env.step_dt))
    if args_cli.max_steps is not None and args_cli.max_steps > 0:
        clip_steps = min(int(args_cli.max_steps), schedule.num_steps)
        schedule = ReplaySchedule(
            dt=schedule.dt,
            t=schedule.t[:clip_steps],
            commands=schedule.commands[:clip_steps, :],
        )

    governor = None
    if args_cli.governor:
        governor = ThermalVoltageGovernor(
            GovernorConfig(
                temp_warn_c=float(args_cli.temp_warn_c),
                temp_crit_c=float(args_cli.temp_crit_c),
                temp_stop_c=float(args_cli.temp_stop_c),
                cell_warn_v=float(args_cli.cell_warn_v),
                cell_scale_stop_v=float(args_cli.cell_stop_v),
                cell_hard_stop_v=float(args_cli.cell_hard_stop_v),
                pack_hard_stop_v=float(args_cli.pack_stop_v),
                temp_prediction_horizon_s=float(args_cli.temp_pred_horizon_s),
                temp_filter_tau_s=float(args_cli.temp_tau_s),
                cell_filter_tau_s=float(args_cli.volt_tau_s),
                pack_filter_tau_s=float(args_cli.volt_tau_s),
                yaw_exponent=float(args_cli.yaw_exp),
            )
        )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"replay_{run_stamp}_{Path(args_cli.command_file).stem}_{'gov' if args_cli.governor else 'nogov'}"
    run_dir = Path(args_cli.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_source_seen: set[str] = set()

    trial_summaries: list[dict[str, Any]] = []
    for trial_idx in range(int(args_cli.num_trials)):
        obs = _obs_from_reset_output(env.reset())
        if args_cli.soc_init is not None:
            _set_initial_soc(base_env, float(args_cli.soc_init))
        _apply_fault(base_env=base_env, fault_leg="", kp_scale=1.0, kd_scale=1.0)
        fault_enabled = args_cli.fault_leg.strip() != ""
        fault_applied = False
        _set_risk_factor(base_env, float(args_cli.risk_factor_fixed))
        if governor is not None:
            governor.reset()

        rows: list[dict[str, Any]] = []
        stop_reason = ""
        completed_script = True
        for i in range(schedule.num_steps):
            loop_t0 = time.time()
            t_now = float(i * base_env.step_dt)

            if (
                fault_enabled
                and (not fault_applied)
                and (t_now >= float(args_cli.fault_start_s))
            ):
                _apply_fault(
                    base_env=base_env,
                    fault_leg=args_cli.fault_leg,
                    kp_scale=float(args_cli.fault_kp_scale),
                    kd_scale=float(args_cli.fault_kd_scale),
                )
                fault_applied = True

            raw_vx, raw_vy, raw_wz = schedule.commands[i]
            temp_now, temp_src_now = _temperature_for_governor(
                base_env=base_env,
                signal_mode=str(args_cli.temp_signal),
                coil_to_case_delta_c=float(args_cli.coil_to_case_delta_c),
            )
            temp_source_seen.add(temp_src_now)
            vpack_now = _pack_voltage(base_env)
            cell_v_now = _cell_min_voltage(base_env)

            if governor is not None:
                scale_lin, scale_yaw, hard_stop, gov_dbg = governor.step(
                    dt=float(base_env.step_dt),
                    temp_max_c=temp_now,
                    cell_v_min=cell_v_now,
                    pack_v=vpack_now,
                )
            else:
                scale_lin, scale_yaw, hard_stop = 1.0, 1.0, False
                gov_dbg = {
                    "temp_ema_c": temp_now,
                    "temp_pred_c": temp_now,
                    "temp_rate_cps": 0.0,
                    "cell_v_ema": cell_v_now,
                    "pack_v_ema": vpack_now,
                    "s_temp": 1.0,
                    "s_volt": 1.0,
                    "s_target": 1.0,
                }

            cmd_vx = float(raw_vx * scale_lin)
            cmd_vy = float(raw_vy * scale_lin)
            cmd_wz = float(raw_wz * scale_yaw)
            _set_velocity_command(base_env, cmd_vx, cmd_vy, cmd_wz)
            _set_risk_factor(base_env, float(args_cli.risk_factor_fixed))

            obs = env.get_observations()
            with torch.no_grad():
                action = policy(obs)
            action = action.clone()
            obs, _, dones, _ = env.step(action)
            _safe_reset_recurrent(policy_nn, dones)

            wz_actual = float(base_env.scene["robot"].data.root_ang_vel_b[0, 2].detach().cpu().item())
            temp_after, temp_src_after = _temperature_for_governor(
                base_env=base_env,
                signal_mode=str(args_cli.temp_signal),
                coil_to_case_delta_c=float(args_cli.coil_to_case_delta_c),
            )
            temp_source_seen.add(temp_src_after)
            vpack_after = _pack_voltage(base_env)
            cell_v_after = _cell_min_voltage(base_env)
            power_w = float(torch.sum(base_env.phm_state.avg_power_log[0]).detach().cpu().item())

            rows.append(
                {
                    "step": i,
                    "t_s": t_now,
                    "vx_cmd_raw": float(raw_vx),
                    "vy_cmd_raw": float(raw_vy),
                    "wz_cmd_raw": float(raw_wz),
                    "vx_cmd_exec": cmd_vx,
                    "vy_cmd_exec": cmd_vy,
                    "wz_cmd_exec": cmd_wz,
                    "fault_active": float(fault_applied),
                    "wz_actual": wz_actual,
                    "temp_max_c": temp_after,
                    "temp_source": temp_src_after,
                    "vpack_v": vpack_after,
                    "vcell_min_v": cell_v_after,
                    "power_w": power_w,
                    "scale_lin": float(scale_lin),
                    "scale_yaw": float(scale_yaw),
                    "gov_temp_ema_c": float(gov_dbg["temp_ema_c"]),
                    "gov_temp_pred_c": float(gov_dbg["temp_pred_c"]),
                    "gov_cell_ema_v": float(gov_dbg["cell_v_ema"]),
                    "gov_pack_ema_v": float(gov_dbg["pack_v_ema"]),
                    "gov_s_temp": float(gov_dbg["s_temp"]),
                    "gov_s_volt": float(gov_dbg["s_volt"]),
                    "gov_stop_temp": float(gov_dbg.get("hard_stop_temp", 0.0)),
                    "gov_stop_pack": float(gov_dbg.get("hard_stop_pack", 0.0)),
                    "gov_stop_cell": float(gov_dbg.get("hard_stop_cell", 0.0)),
                }
            )

            if bool(dones[0].detach().cpu().item()):
                completed_script = False
                stop_reason = "env_terminated"
                break
            if hard_stop:
                completed_script = False
                stop_tags = []
                if float(gov_dbg.get("hard_stop_temp", 0.0)) > 0.5:
                    stop_tags.append("temp")
                if float(gov_dbg.get("hard_stop_pack", 0.0)) > 0.5:
                    stop_tags.append("pack")
                if float(gov_dbg.get("hard_stop_cell", 0.0)) > 0.5:
                    stop_tags.append("cell")
                stop_reason = "governor_hard_stop_" + ("+".join(stop_tags) if stop_tags else "unknown")
                break

            if args_cli.real_time:
                sleep_t = float(base_env.step_dt) - (time.time() - loop_t0)
                if sleep_t > 0.0:
                    time.sleep(sleep_t)

        if stop_reason == "" and completed_script:
            stop_reason = "completed"

        trial_summary = _summarize_rows(
            rows=rows,
            dt=float(base_env.step_dt),
            completed_script=completed_script,
            stop_reason=stop_reason,
            temp_warn_c=float(args_cli.temp_warn_c),
        )
        trial_summary["trial_index"] = trial_idx
        trial_summaries.append(trial_summary)
        _write_step_csv(run_dir / f"trial_{trial_idx:02d}_steps.csv", rows)

        print(
            f"[Trial {trial_idx:02d}] completed={trial_summary['completed_script']} "
            f"yaw_mae={trial_summary.get('yaw_mae_exec', float('nan')):.4f} "
            f"Tmax={trial_summary.get('temp_max_c', float('nan')):.2f}C "
            f"Vmin={trial_summary.get('vpack_min_v', float('nan')):.2f}V "
            f"E={trial_summary.get('energy_j', float('nan')):.2f}J "
            f"reason={trial_summary.get('stop_reason', '')}"
        )

    output = {
        "timestamp": run_stamp,
        "task": args_cli.task,
        "checkpoint": args_cli.checkpoint,
        "command_file": args_cli.command_file,
        "schedule": {
            "dt": schedule.dt,
            "num_steps": schedule.num_steps,
            "duration_s": schedule.duration_s,
        },
        "config": {
            "num_trials": args_cli.num_trials,
            "governor": bool(args_cli.governor),
            "fault_leg": args_cli.fault_leg,
            "fault_kp_scale": args_cli.fault_kp_scale,
            "fault_kd_scale": args_cli.fault_kd_scale,
            "fault_start_s": args_cli.fault_start_s,
            "soc_init": args_cli.soc_init,
            "temp_warn_c": args_cli.temp_warn_c,
            "temp_crit_c": args_cli.temp_crit_c,
            "temp_stop_c": args_cli.temp_stop_c,
            "temp_signal": args_cli.temp_signal,
            "coil_to_case_delta_c": args_cli.coil_to_case_delta_c,
            "cell_warn_v": args_cli.cell_warn_v,
            "cell_stop_v": args_cli.cell_stop_v,
            "cell_hard_stop_v": args_cli.cell_hard_stop_v,
            "pack_stop_v": args_cli.pack_stop_v,
            "temp_pred_horizon_s": args_cli.temp_pred_horizon_s,
            "temp_tau_s": args_cli.temp_tau_s,
            "volt_tau_s": args_cli.volt_tau_s,
            "yaw_exp": args_cli.yaw_exp,
            "risk_factor_fixed": args_cli.risk_factor_fixed,
        },
        "temperature_sources_seen": sorted(list(temp_source_seen)),
        "per_trial": trial_summaries,
        "aggregate": _aggregate_trial_summaries(trial_summaries),
    }
    summary_path = run_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"[DONE] Replay evaluation results saved to: {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
