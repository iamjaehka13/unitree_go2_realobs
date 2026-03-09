# =============================================================================
# evaluate.py — MotorDeg Evaluation / Reporting Script
# =============================================================================
# Runs a trained policy under controlled degradation scenarios and exports
# locomotion, safety, and governor diagnostics for reports and paper tables.
#
# Usage:
#   python evaluate.py --task Unitree-Go2-Strategic-v1 \
#       --checkpoint /path/to/model_3000.pt \
#       --num_envs 512 --num_episodes 100 \
#       --output_dir ./eval_results/realobs_eval \
#       --headless
# =============================================================================

"""Evaluation entrypoint.

AppLauncher must be initialized before importing Isaac task modules.
"""

import argparse
import sys

from isaaclab.app import AppLauncher
from paper_b_runtime import apply_paper_b_runtime_overrides

parser = argparse.ArgumentParser(description="Evaluate trained policy under degradation scenarios.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
parser.add_argument("--num_envs", type=int, default=512, help="Number of parallel envs")
parser.add_argument("--num_episodes", type=int, default=100, help="Min episodes per scenario to collect")
parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory for CSV files")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Agent config entry point"
)
parser.add_argument(
    "--critical_governor_enable",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Override critical command governor enable state for Paper B evaluation.",
)
parser.add_argument(
    "--paper_b_obs_ablation",
    "--realobs_obs_ablation",
    dest="paper_b_obs_ablation",
    type=str,
    default="none",
    choices=["none", "no_voltage", "no_thermal", "no_vibration"],
    help="Paper B observation-channel ablation.",
)
parser.add_argument(
    "--paper_b_sensor_preset",
    "--realobs_sensor_preset",
    dest="paper_b_sensor_preset",
    type=str,
    default="full",
    choices=["full", "ideal", "voltage_only", "encoder_transport"],
    help="Paper B sensor-realism preset.",
)
parser.add_argument(
    "--eval_fault_mode",
    type=str,
    default="from_env",
    choices=["from_env", "single_motor_random", "single_motor_fixed", "all_motors_random"],
    help=(
        "Fault injection mode for scenario evaluation. "
        "'from_env' keeps env cfg setting; other values override it."
    ),
)
parser.add_argument(
    "--eval_fault_motor_id",
    type=int,
    default=-1,
    help="If >=0, force single_motor_fixed evaluation on this motor id (0-11).",
)
parser.add_argument(
    "--paper_protocol_strict",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Enforce paper protocol for evaluation: single_motor_fixed + valid motor id. "
        "Use --no-paper-protocol-strict only for exploratory/debug runs."
    ),
)
parser.add_argument(
    "--eval_cmd_profile",
    type=str,
    default="from_env",
    choices=["from_env", "stand", "forced_walk", "forced_walk_then_zero", "target_random"],
    help=(
        "Override evaluation command distribution. "
        "'from_env' keeps current task cfg; others force fixed command profiles."
    ),
)
parser.add_argument(
    "--eval_protocol_mode",
    type=str,
    default="combined",
    choices=["combined", "locomotion_only", "safety_only"],
    help=(
        "Evaluation protocol split. "
        "'combined': fresh/used/aged/critical, "
        "'locomotion_only': fresh/used/aged, "
        "'safety_only': critical."
    ),
)
parser.add_argument(
    "--eval_safety_cmd_profile",
    type=str,
    default="stand",
    choices=["from_env", "stand", "forced_walk", "forced_walk_then_zero", "target_random"],
    help=(
        "Command profile used for safety_only protocol when --eval_cmd_profile=from_env. "
        "Default is stand (safe-stop focused)."
    ),
)
parser.add_argument(
    "--eval_forced_walk_lin_x_min",
    type=float,
    default=0.3,
    help="Forced-walk profile override: minimum forward speed command (m/s).",
)
parser.add_argument(
    "--eval_forced_walk_lin_x_max",
    type=float,
    default=0.8,
    help="Forced-walk profile override: maximum forward speed command (m/s).",
)
parser.add_argument(
    "--eval_forced_walk_ang_z_min",
    type=float,
    default=-0.5,
    help="Forced-walk profile override: minimum yaw-rate command (rad/s).",
)
parser.add_argument(
    "--eval_forced_walk_ang_z_max",
    type=float,
    default=0.5,
    help="Forced-walk profile override: maximum yaw-rate command (rad/s).",
)
parser.add_argument(
    "--eval_forced_walk_ramp_s",
    type=float,
    default=0.0,
    help=(
        "Forced-walk debug: if >0, linearly ramp commanded velocity from zero to target "
        "over this duration (seconds) after each reset."
    ),
)
parser.add_argument(
    "--eval_forced_walk_then_zero_walk_s",
    type=float,
    default=6.0,
    help=(
        "Forced-walk-then-zero profile: duration of forced-walk phase before "
        "command is hard-set to zero (seconds) for recovery/unlatch evaluation."
    ),
)
parser.add_argument(
    "--eval_disable_velocity_curriculum",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Disable velocity command curriculum during evaluation for fixed-distribution comparisons."
    ),
)
parser.add_argument(
    "--eval_nonzero_cmd_threshold",
    type=float,
    default=0.05,
    help="Threshold (m/s) for non-zero command ratio in walk-vs-stand diagnostics.",
)
parser.add_argument(
    "--eval_stand_cmd_threshold",
    type=float,
    default=0.01,
    help="Threshold (m/s) for stand command ratio in walk-vs-stand diagnostics.",
)
parser.add_argument(
    "--eval_safe_stop_lin_vel_max",
    type=float,
    default=0.05,
    help="Critical safety metric: max |v_xy| (m/s) to consider a step as safe-stop state.",
)
parser.add_argument(
    "--eval_safe_stop_ang_vel_max",
    type=float,
    default=0.10,
    help="Critical safety metric: max |w_z| (rad/s) to consider a step as safe-stop state.",
)
parser.add_argument(
    "--eval_safe_stop_hold_s",
    type=float,
    default=1.0,
    help="Critical safety metric: required continuous safe-stop duration (seconds).",
)
parser.add_argument(
    "--eval_safe_stop_require_pose",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "If enabled, safe-stop requires pose stability in addition to velocity thresholds "
        "(roll/pitch and base height)."
    ),
)
parser.add_argument(
    "--eval_safe_stop_max_tilt_rad",
    type=float,
    default=0.35,
    help="Safe-stop pose threshold: max absolute roll/pitch (rad).",
)
parser.add_argument(
    "--eval_safe_stop_min_height_m",
    type=float,
    default=0.28,
    help="Safe-stop pose threshold: minimum base height (m).",
)
parser.add_argument(
    "--eval_oracle_safe_stop_steps",
    type=int,
    default=0,
    help=(
        "Oracle debug: if >0, ignore policy actions and force zero-action for the first N steps "
        "after each episode reset."
    ),
)
parser.add_argument(
    "--eval_crit_cmd_delta_active_eps",
    type=float,
    default=1e-3,
    help=(
        "Step is counted as active governor command intervention if "
        "||cmd_eff-cmd_raw|| > this epsilon."
    ),
)
parser.add_argument(
    "--eval_cls_stand_actual_max",
    type=float,
    default=0.05,
    help="Episode classification threshold: max mean actual speed (m/s) for stand.",
)
parser.add_argument(
    "--eval_cls_stand_ratio_min",
    type=float,
    default=0.80,
    help="Episode classification threshold: min stand_cmd_ratio for stand.",
)
parser.add_argument(
    "--eval_cls_stand_cmd_ang_abs_max",
    type=float,
    default=0.15,
    help="Episode classification threshold: max mean |cmd yaw rate| (rad/s) for stand.",
)
parser.add_argument(
    "--eval_cls_walk_cmd_min",
    type=float,
    default=0.04,
    help="Episode classification threshold: min mean commanded speed (m/s) for walk.",
)
parser.add_argument(
    "--eval_cls_walk_nz_ratio_min",
    type=float,
    default=0.60,
    help="Episode classification threshold: min nonzero_cmd_ratio for walk.",
)
parser.add_argument(
    "--eval_cls_walk_progress_min",
    type=float,
    default=0.05,
    help="Episode classification threshold: min forward progress distance (m) for walk.",
)
parser.add_argument(
    "--eval_cls_walk_progress_speed_min",
    type=float,
    default=0.03,
    help="Episode classification threshold: min mean forward progress speed (m/s) for walk.",
)
parser.add_argument(
    "--eval_cls_walk_progress_ratio_min",
    type=float,
    default=0.45,
    help="Episode classification threshold: min progress_speed/cmd_speed ratio for walk.",
)
parser.add_argument(
    "--eval_contact_force_threshold",
    type=float,
    default=15.0,
    help="Contact force threshold (N) for gait/contact diagnostics.",
)
parser.add_argument(
    "--eval_debug_print_body_names",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print robot body names and resolved foot bodies for gait diagnostics.",
)
parser.add_argument(
    "--eval_foot_body_names",
    type=str,
    default="",
    help=(
        "Comma-separated foot body names for gait diagnostics "
        "(e.g., 'FL_foot,FR_foot,RL_foot,RR_foot'). Empty = auto-detect."
    ),
)
parser.add_argument(
    "--eval_dump_timeseries",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Dump per-step diagnostic timeseries CSV for one selected env/episode.",
)
parser.add_argument(
    "--eval_dump_timeseries_scenario",
    type=str,
    default="critical",
    choices=["fresh", "used", "aged", "critical"],
    help="Scenario name for timeseries dumping.",
)
parser.add_argument(
    "--eval_dump_timeseries_env_id",
    type=int,
    default=0,
    help="Environment index to dump timeseries from.",
)
parser.add_argument(
    "--eval_dump_timeseries_max_steps",
    type=int,
    default=2000,
    help="Maximum steps to dump for one selected episode.",
)

# local imports
import cli_args  # isort: skip
# Keep evaluate.py's required --checkpoint argument as the single source.
cli_args.add_rsl_rl_args(parser, include_checkpoint=False)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set_int("/log/channels/omni.physx.tensors.plugin/level", 1)

"""The remaining imports depend on AppLauncher/Hydra argv being configured above."""

import csv
import gymnasium as gym
import os
import json
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_go2_realobs.tasks  # noqa: F401
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.paper_b_task_contract import (
    summarize_paper_b_task_cfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.mdp.realobs_contract import (
    resolve_realobs_case_temperature_tensor,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.motor_deg.interface import (
    case_proxy_safe_coil_max_for_reset,
    thermal_termination_params_from_cfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.motor_deg.utils import compute_battery_voltage

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Evaluation Scenario Presets
# =============================================================================
SCENARIOS = {
    "fresh": {
        "description": "Brand new robot, no degradation",
        "fatigue_range": (0.0, 0.0),
        "health_margin": (1.0, 1.0),
        "temp_range": (25.0, 25.0),
        "soc_range": (1.0, 1.0),
    },
    "used": {
        "description": "Moderate wear, SOC 80-100%",
        "fatigue_range": (0.1, 0.3),
        "health_margin": (0.30, 0.55),
        "temp_range": (25.0, 35.0),
        "soc_range": (0.8, 1.0),
    },
    "aged": {
        "description": "Significant wear, warm motors, SOC 40-80%",
        "fatigue_range": (0.4, 0.7),
        "health_margin": (0.10, 0.25),
        "temp_range": (35.0, 55.0),
        "soc_range": (0.4, 0.8),
    },
    "critical": {
        "description": "Near end-of-life, hot motors, low battery",
        "fatigue_range": (0.7, 0.95),
        "health_margin": (0.02, 0.10),
        "temp_range": (65.0, 85.0),
        "soc_range": (0.1, 0.3),
    },
}


@dataclass
class _RewardTermSpec:
    name: str
    func: Any
    weight: float
    params: dict


def _extract_reward_term_specs(base_env) -> list[_RewardTermSpec]:
    """Extract active reward terms from env cfg in a robust, order-preserving way."""
    reward_cfg = getattr(getattr(base_env, "cfg", None), "rewards", None)
    if reward_cfg is None:
        return []

    if hasattr(reward_cfg, "__dataclass_fields__"):
        candidate_names = list(reward_cfg.__dataclass_fields__.keys())
    else:
        candidate_names = [k for k in vars(reward_cfg).keys() if not k.startswith("_")]

    terms: list[_RewardTermSpec] = []
    for name in candidate_names:
        term = getattr(reward_cfg, name, None)
        if term is None:
            continue
        if not hasattr(term, "func") or not hasattr(term, "weight"):
            continue

        try:
            weight = float(term.weight)
        except Exception:
            continue

        if abs(weight) <= 1e-12:
            # Skip inactive terms to keep breakdown focused.
            continue

        params = getattr(term, "params", None)
        if params is None:
            params = {}
        if not isinstance(params, dict):
            continue

        terms.append(_RewardTermSpec(name=name, func=term.func, weight=weight, params=params))

    return terms


class _RewardBreakdownCollector:
    """
    Recompute per-term reward contributions for analysis.

    The collector auto-detects whether manager-level reward scaling includes dt,
    by comparing reconstructed reward vs environment reward buffer at runtime.
    """

    def __init__(self, base_env):
        self.base_env = base_env
        self.step_dt = float(getattr(base_env, "step_dt", 1.0))
        self.term_specs = _extract_reward_term_specs(base_env)
        self.term_names = [t.name for t in self.term_specs]

        self._use_dt_scaling: bool | None = None
        self._decision_mae_no_dt: float | None = None
        self._decision_mae_with_dt: float | None = None
        self.last_reconstruction_mae: float | None = None

    @property
    def use_dt_scaling(self) -> bool | None:
        return self._use_dt_scaling

    def _as_reward_vector(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(x, device=self.base_env.device, dtype=torch.float32)
        if y.ndim == 0:
            y = y.repeat(self.base_env.num_envs)
        if y.ndim > 1:
            if y.shape[-1] == 1:
                y = y.squeeze(-1)
            else:
                y = torch.mean(y, dim=tuple(range(1, y.ndim)))
        return y

    def compute_step_contributions(self, reward_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        if len(self.term_specs) == 0:
            return {}

        reward_vec = self._as_reward_vector(reward_vec)
        contrib_no_dt: dict[str, torch.Tensor] = {}
        sum_no_dt = torch.zeros_like(reward_vec)
        sum_with_dt = torch.zeros_like(reward_vec)

        for spec in self.term_specs:
            raw = spec.func(self.base_env, **spec.params)
            raw = self._as_reward_vector(raw)
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)

            weighted = raw * float(spec.weight)
            contrib_no_dt[spec.name] = weighted
            sum_no_dt = sum_no_dt + weighted
            sum_with_dt = sum_with_dt + (weighted * self.step_dt)

        if self._use_dt_scaling is None:
            mae_no_dt = torch.mean(torch.abs(sum_no_dt - reward_vec)).item()
            mae_with_dt = torch.mean(torch.abs(sum_with_dt - reward_vec)).item()
            self._decision_mae_no_dt = float(mae_no_dt)
            self._decision_mae_with_dt = float(mae_with_dt)
            self._use_dt_scaling = mae_with_dt <= mae_no_dt

        if self._use_dt_scaling:
            contrib = {k: v * self.step_dt for k, v in contrib_no_dt.items()}
            recon = sum_with_dt
        else:
            contrib = contrib_no_dt
            recon = sum_no_dt

        self.last_reconstruction_mae = float(torch.mean(torch.abs(recon - reward_vec)).item())
        return contrib

    def meta(self) -> dict[str, Any]:
        return {
            "terms": list(self.term_names),
            "dt_scaled": bool(self._use_dt_scaling) if self._use_dt_scaling is not None else None,
            "decision_mae_no_dt": self._decision_mae_no_dt,
            "decision_mae_with_dt": self._decision_mae_with_dt,
            "last_reconstruction_mae": self.last_reconstruction_mae,
        }


def _thermal_failure_params(base_env) -> tuple[float | None, bool, float]:
    """Read thermal termination settings from env cfg (None when disabled)."""
    return thermal_termination_params_from_cfg(getattr(base_env, "cfg", None))


def _case_temperature_tensor(deg_state):
    for name in ("motor_case_temp", "case_temp", "motor_temp_case", "housing_temp", "motor_housing_temp"):
        if hasattr(deg_state, name):
            val = getattr(deg_state, name)
            if isinstance(val, torch.Tensor):
                return val
    return None


def _temperature_metric_semantics_with_source(base_env) -> tuple[str, str]:
    """Resolve temperature metric semantics and where that decision came from."""
    cfg = getattr(base_env, "cfg", None)
    explicit = str(getattr(cfg, "temperature_metric_semantics", "")).strip().lower()
    if explicit in {"case_proxy", "case", "case_like"}:
        return "case_proxy", "cfg.temperature_metric_semantics"
    if explicit in {"coil_hotspot", "coil"}:
        return "coil_hotspot", "cfg.temperature_metric_semantics"

    # Backward compatibility: infer from thermal termination params.
    term_cfg = getattr(getattr(base_env, "cfg", None), "terminations", None)
    thermal_failure = getattr(term_cfg, "thermal_failure", None) if term_cfg is not None else None
    params = getattr(thermal_failure, "params", None)
    _, use_case_proxy, _ = _thermal_failure_params(base_env)
    if isinstance(params, dict):
        source = "terminations.thermal_failure.params.use_case_proxy"
    else:
        source = "default_fallback(coil_hotspot)"
    semantics = "case_proxy" if use_case_proxy else "coil_hotspot"
    return semantics, source


def _temperature_metric_semantics(base_env) -> str:
    """Resolve which temperature channel should be reported/evaluated."""
    semantics, _ = _temperature_metric_semantics_with_source(base_env)
    return semantics


def _fault_injection_params(base_env, num_motors: int, strict: bool = False) -> tuple[str, int]:
    """Read/reset-safe fault-injection mode from env cfg."""
    cfg = getattr(base_env, "cfg", None)
    mode = str(getattr(cfg, "motor_deg_fault_injection_mode", "single_motor_random")).strip().lower()
    if mode not in {"all_motors_random", "single_motor_random", "single_motor_fixed"}:
        if strict:
            raise ValueError(
                f"Invalid motor_deg_fault_injection_mode='{mode}'. "
                "Expected one of: all_motors_random, single_motor_random, single_motor_fixed."
            )
        mode = "single_motor_random"

    fixed_id = int(getattr(cfg, "motor_deg_fault_fixed_motor_id", -1))
    if mode == "single_motor_fixed" and not (0 <= fixed_id < int(num_motors)):
        if strict:
            raise ValueError(
                f"Invalid motor_deg_fault_fixed_motor_id={fixed_id} for single_motor_fixed mode "
                f"(expected 0..{int(num_motors)-1})."
            )
        mode = "single_motor_random"
        fixed_id = -1
    return mode, fixed_id


def _apply_eval_command_profile(
    env_cfg,
    profile: str,
    forced_walk_lin_x: tuple[float, float] = (0.3, 0.8),
    forced_walk_ang_z: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, Any]:
    """Optionally override command distribution for evaluation-only runs."""
    profile = str(profile).strip().lower()
    info: dict[str, Any] = {"profile": profile, "applied": False}
    if profile == "from_env":
        return info

    commands_cfg = getattr(env_cfg, "commands", None)
    base_vel_cfg = getattr(commands_cfg, "base_velocity", None) if commands_cfg is not None else None
    ranges = getattr(base_vel_cfg, "ranges", None) if base_vel_cfg is not None else None
    if ranges is None:
        info["warning"] = "base_velocity ranges not found; command profile override skipped."
        return info

    def _to_pair(x, fallback):
        try:
            a, b = x
            return float(a), float(b)
        except Exception:
            return float(fallback[0]), float(fallback[1])

    if profile == "stand":
        lin_x = (0.0, 0.0)
        lin_y = (0.0, 0.0)
        ang_z = (0.0, 0.0)
        rel_standing = 1.0
    elif profile in {"forced_walk", "forced_walk_then_zero"}:
        lin_x = (float(forced_walk_lin_x[0]), float(forced_walk_lin_x[1]))
        lin_y = (0.0, 0.0)
        ang_z = (float(forced_walk_ang_z[0]), float(forced_walk_ang_z[1]))
        rel_standing = 0.0
    elif profile == "target_random":
        lin_x = _to_pair(getattr(env_cfg, "velocity_cmd_target_lin_vel_x", (-1.0, 1.0)), (-1.0, 1.0))
        lin_y = _to_pair(getattr(env_cfg, "velocity_cmd_target_lin_vel_y", (-0.4, 0.4)), (-0.4, 0.4))
        ang_z = _to_pair(getattr(env_cfg, "velocity_cmd_target_ang_vel_z", (-1.0, 1.0)), (-1.0, 1.0))
        rel_standing = 0.0
    else:
        info["warning"] = f"unknown eval_cmd_profile={profile}; override skipped."
        return info

    if hasattr(ranges, "lin_vel_x"):
        ranges.lin_vel_x = lin_x
    if hasattr(ranges, "lin_vel_y"):
        ranges.lin_vel_y = lin_y
    if hasattr(ranges, "ang_vel_z"):
        ranges.ang_vel_z = ang_z
    if hasattr(base_vel_cfg, "rel_standing_envs"):
        base_vel_cfg.rel_standing_envs = float(rel_standing)

    info.update(
        {
            "applied": True,
            "lin_vel_x": tuple(getattr(ranges, "lin_vel_x", lin_x)),
            "lin_vel_y": tuple(getattr(ranges, "lin_vel_y", lin_y)),
            "ang_vel_z": tuple(getattr(ranges, "ang_vel_z", ang_z)),
            "rel_standing_envs": float(getattr(base_vel_cfg, "rel_standing_envs", rel_standing)),
        }
    )
    return info


def _try_enable_contact_sensors(env_cfg) -> str | None:
    """Best-effort contact sensor enable across Isaac Lab config variants."""
    candidates = [
        ("scene", "robot", "activate_contact_sensors"),
        ("scene", "robot", "spawn", "activate_contact_sensors"),
    ]
    for path in candidates:
        obj = env_cfg
        ok = True
        for part in path[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if not ok or not hasattr(obj, path[-1]):
            continue
        try:
            setattr(obj, path[-1], True)
            return ".".join(path)
        except Exception:
            continue
    return None


def _temperature_tensor_for_eval(base_env) -> torch.Tensor:
    deg_state = base_env.motor_deg_state
    _, _, coil_to_case_delta_c = _thermal_failure_params(base_env)
    temp_semantics = _temperature_metric_semantics(base_env)
    if temp_semantics == "case_proxy":
        case_temp, _ = resolve_realobs_case_temperature_tensor(
            base_env,
            coil_to_case_delta_c=coil_to_case_delta_c,
        )
        if case_temp is not None:
            return case_temp
    return deg_state.coil_temp


def _sync_long_term_buffer_after_scenario(base_env, env_ids: torch.Tensor):
    """Reset long-term snapshots so slope/trend observations match injected MotorDeg state."""
    if env_ids.numel() == 0:
        return

    ltb = getattr(base_env, "motor_deg_long_term_buffer", None)
    if ltb is None:
        return

    fatigue_now = base_env.motor_deg_state.fatigue_index[env_ids]

    if hasattr(ltb, "fatigue_snapshots"):
        ltb.fatigue_snapshots[env_ids] = 0.0
        ltb.fatigue_snapshots[env_ids, 0, :] = fatigue_now
    if hasattr(ltb, "snapshot_index"):
        ltb.snapshot_index[env_ids] = 1
    if hasattr(ltb, "fill_count"):
        ltb.fill_count[env_ids] = 1
    if hasattr(ltb, "is_buffer_filled"):
        ltb.is_buffer_filled[env_ids] = False
    if hasattr(ltb, "step_timer"):
        ltb.step_timer[env_ids] = 1
    if hasattr(ltb, "thermal_overload_duration"):
        ltb.thermal_overload_duration[env_ids] = 0.0


def _sync_runtime_state_after_scenario(base_env, env_ids: torch.Tensor):
    """Synchronize derived runtime state after direct MotorDeg state injection."""
    if env_ids.numel() == 0:
        return

    deg_state = base_env.motor_deg_state

    # 1) Electrical state sync
    soc = deg_state.soc[env_ids]
    zero_load = torch.zeros_like(soc)
    true_voltage = compute_battery_voltage(soc, zero_load)
    true_voltage = torch.nan_to_num(true_voltage, nan=25.0).clamp(0.0, 60.0)

    if hasattr(deg_state, "battery_voltage_true"):
        deg_state.battery_voltage_true[env_ids] = true_voltage

    sensor_bias = deg_state.voltage_sensor_bias[env_ids] if hasattr(deg_state, "voltage_sensor_bias") else torch.zeros_like(true_voltage)
    observed_voltage = torch.clamp(true_voltage + sensor_bias, 0.0, 60.0)
    deg_state.battery_voltage[env_ids] = observed_voltage

    if hasattr(deg_state, "min_voltage_log"):
        deg_state.min_voltage_log[env_ids] = true_voltage

    # 2) Always refresh BMS prediction cache.
    # Brownout may use this cache or measured channels depending on env config.
    if hasattr(base_env, "_predict_instant_voltage_ivp"):
        predicted_all = base_env._predict_instant_voltage_ivp(use_noisy_state=True, use_nominal_model=True)
        predicted_voltage = predicted_all[env_ids]
    else:
        predicted_voltage = observed_voltage

    if hasattr(deg_state, "bms_voltage_pred"):
        deg_state.bms_voltage_pred[env_ids] = predicted_voltage

    # 3) Brownout state sync (use same thresholds as env.step)
    if hasattr(deg_state, "brownout_latched") and hasattr(deg_state, "brownout_scale"):
        def _scalar_attr(name: str, default: float) -> float:
            value = getattr(base_env, name, default)
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu().item())
            return float(value)

        brownout_enter_v = _scalar_attr("_brownout_enter_v", 24.5)
        brownout_recover_v = _scalar_attr("_brownout_recover_v", 25.0)
        brownout_scale_low = _scalar_attr("_const_brownout_scale_low", 0.5)
        brownout_scale_high = _scalar_attr("_const_brownout_scale_high", 1.0)
        brownout_source = str(getattr(base_env, "_brownout_voltage_source", "bms_pred")).strip().lower()

        if brownout_source == "true_voltage" and hasattr(deg_state, "battery_voltage_true"):
            brownout_voltage = deg_state.battery_voltage_true[env_ids]
        elif brownout_source == "sensor_voltage" and hasattr(deg_state, "battery_voltage"):
            brownout_voltage = deg_state.battery_voltage[env_ids]
        else:
            brownout_voltage = predicted_voltage

        is_low_voltage = brownout_voltage < brownout_enter_v
        is_recovered = brownout_voltage > brownout_recover_v
        current_latch = deg_state.brownout_latched[env_ids]

        new_latch = torch.where(is_low_voltage, torch.ones_like(current_latch), current_latch)
        new_latch = torch.where(is_recovered, torch.zeros_like(new_latch), new_latch)
        deg_state.brownout_latched[env_ids] = new_latch

        deg_state.brownout_scale[env_ids] = torch.where(
            new_latch,
            torch.full_like(predicted_voltage, brownout_scale_low),
            torch.full_like(predicted_voltage, brownout_scale_high),
        )

    # 4) Align long-term MotorDeg buffer with directly injected scenario state.
    _sync_long_term_buffer_after_scenario(base_env, env_ids)

    # 5) Apply degradation and effort limits immediately
    if hasattr(base_env, "_apply_physical_degradation"):
        base_env._apply_physical_degradation(env_ids=env_ids)

    if hasattr(base_env, "_compute_thermal_limits"):
        thermal_limits = base_env._compute_thermal_limits(env_ids=env_ids)
        if hasattr(deg_state, "brownout_scale"):
            final_limits = thermal_limits * deg_state.brownout_scale[env_ids].unsqueeze(-1)
        else:
            final_limits = thermal_limits

        base_env.robot.data.joint_effort_limits[env_ids] = final_limits
        if hasattr(base_env.robot, "write_joint_effort_limit_to_sim"):
            base_env.robot.write_joint_effort_limit_to_sim(final_limits, env_ids=env_ids)


def apply_scenario(env, scenario_name: str, env_ids: torch.Tensor):
    """Force-inject a specific degradation scenario for controlled evaluation."""
    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    scenario = dict(SCENARIOS[scenario_name])
    device = base_env.device
    env_ids = env_ids.to(device=device, dtype=torch.long)
    n = len(env_ids)
    num_motors = base_env.motor_deg_state.fatigue_index.shape[1]
    fault_mode, fixed_motor_id = _fault_injection_params(base_env, num_motors=num_motors)
    ambient_temp = float(SCENARIOS["fresh"]["temp_range"][0])
    # Keep runtime scenario-id aligned with injected scenario so critical-only
    # control logic (e.g., critical command governor) activates as intended.
    if hasattr(base_env, "_motor_deg_scenario_id"):
        scenario_id_map = {"fresh": 1, "used": 2, "aged": 3, "critical": 4}
        scenario_id = int(scenario_id_map.get(str(scenario_name).strip().lower(), 0))
        if str(scenario_name).strip().lower() == "critical":
            scenario_id = int(getattr(base_env, "_motor_deg_scenario_id_critical", scenario_id))
        base_env._motor_deg_scenario_id[env_ids] = int(scenario_id)

    fmin, fmax = scenario["fatigue_range"]
    mmin, mmax = scenario["health_margin"]

    tmin, tmax = scenario["temp_range"]
    threshold_temp, use_case_proxy, coil_to_case_delta_c = _thermal_failure_params(base_env)
    # RealObs thermal termination may use case-proxy 72C; reduce injected critical/aged
    # case-equivalent temperature to avoid immediate-start failures in evaluation.
    if use_case_proxy and threshold_temp is not None:
        safe_coil_tmax = case_proxy_safe_coil_max_for_reset(
            threshold_temp=threshold_temp,
            coil_to_case_delta_c=coil_to_case_delta_c,
        )
        if scenario_name == "critical":
            case_tmax = max(30.0, threshold_temp - 2.0)
            case_tmin = max(25.0, case_tmax - 12.0)
            tmin = case_tmin + coil_to_case_delta_c
            tmax = case_tmax + coil_to_case_delta_c
            tmax = max(30.0, min(tmax, safe_coil_tmax))
            tmin = min(tmin, tmax)
        elif scenario_name == "aged":
            case_tmax = max(30.0, threshold_temp - 8.0)
            case_tmin = max(25.0, case_tmax - 14.0)
            tmin = case_tmin + coil_to_case_delta_c
            tmax = case_tmax + coil_to_case_delta_c
            tmax = max(30.0, min(tmax, safe_coil_tmax))
            tmin = min(tmin, tmax)

    fatigue = torch.zeros((n, num_motors), device=device, dtype=torch.float32)
    health = torch.ones((n, num_motors), device=device, dtype=torch.float32)
    temp = torch.full((n, num_motors), ambient_temp, device=device, dtype=torch.float32)
    fault_mask = torch.zeros((n, num_motors), device=device, dtype=torch.float32)
    fault_motor = torch.full((n,), -1, device=device, dtype=torch.long)

    if scenario_name != "fresh":
        if fault_mode == "all_motors_random":
            fatigue = torch.rand((n, num_motors), device=device) * (fmax - fmin) + fmin
            margin = torch.rand((n, num_motors), device=device) * (mmax - mmin) + mmin
            health = torch.clamp(fatigue + margin, max=1.0)
            temp = torch.rand((n, num_motors), device=device) * (tmax - tmin) + tmin
            fault_mask.fill_(1.0)
        else:
            row_idx = torch.arange(n, device=device, dtype=torch.long)
            if fault_mode == "single_motor_fixed" and (0 <= fixed_motor_id < int(num_motors)):
                motor_idx = torch.full((n,), int(fixed_motor_id), device=device, dtype=torch.long)
            else:
                motor_idx = torch.randint(0, int(num_motors), (n,), device=device)

            sampled_fatigue = torch.rand((n,), device=device) * (fmax - fmin) + fmin
            sampled_margin = torch.rand((n,), device=device) * (mmax - mmin) + mmin
            sampled_temp = torch.rand((n,), device=device) * (tmax - tmin) + tmin

            fatigue[row_idx, motor_idx] = sampled_fatigue
            health[row_idx, motor_idx] = torch.clamp(sampled_fatigue + sampled_margin, max=1.0)
            temp[row_idx, motor_idx] = sampled_temp
            fault_mask[row_idx, motor_idx] = 1.0
            fault_motor = motor_idx

    base_env.motor_deg_state.fatigue_index[env_ids] = fatigue
    base_env.motor_deg_state.motor_health_capacity[env_ids] = health
    base_env.motor_deg_state.coil_temp[env_ids] = temp
    if hasattr(base_env.motor_deg_state, "fault_mask"):
        base_env.motor_deg_state.fault_mask[env_ids] = fault_mask
    if hasattr(base_env.motor_deg_state, "fault_motor_id"):
        base_env.motor_deg_state.fault_motor_id[env_ids] = fault_motor
    case_temp_tensor = _case_temperature_tensor(base_env.motor_deg_state)
    if case_temp_tensor is not None:
        if use_case_proxy:
            delta = torch.full((n, num_motors), float(coil_to_case_delta_c), device=device)
        else:
            if scenario_name == "fresh":
                dmin, dmax = 0.0, 1.0
            elif scenario_name == "used":
                dmin, dmax = 1.0, 3.5
            elif scenario_name == "aged":
                dmin, dmax = 2.5, 6.5
            else:
                dmin, dmax = 3.5, 8.0
            delta = torch.rand((n, num_motors), device=device) * (dmax - dmin) + dmin
        case_temp = torch.clamp(base_env.motor_deg_state.coil_temp[env_ids] - delta, min=25.0)
        case_temp_tensor[env_ids] = case_temp

    if hasattr(base_env.motor_deg_state, "temp_derivative"):
        base_env.motor_deg_state.temp_derivative[env_ids] = 0.0
    if hasattr(base_env.motor_deg_state, "case_temp_derivative"):
        base_env.motor_deg_state.case_temp_derivative[env_ids] = 0.0

    smin, smax = scenario["soc_range"]
    soc = torch.rand(n, device=device) * (smax - smin) + smin
    base_env.motor_deg_state.soc[env_ids] = soc

    # Controlled evaluation: neutralize reset-time hidden random biases.
    if hasattr(base_env.motor_deg_state, "friction_bias"):
        base_env.motor_deg_state.friction_bias[env_ids] = 1.0
    if hasattr(base_env.motor_deg_state, "voltage_sensor_bias"):
        base_env.motor_deg_state.voltage_sensor_bias[env_ids] = 0.0
    if hasattr(base_env.motor_deg_state, "encoder_offset"):
        base_env.motor_deg_state.encoder_offset[env_ids] = 0.0

    _sync_runtime_state_after_scenario(base_env, env_ids)


def collect_step_metrics(env) -> dict:
    """Extract per-step metrics from environment state."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    deg_state = unwrapped.motor_deg_state

    # Velocity tracking error
    cmd = unwrapped.command_manager.get_command("base_velocity")[:, :2]
    actual_vel = robot.data.root_lin_vel_b[:, :2]
    tracking_error = torch.norm(cmd - actual_vel, dim=1)

    # Angular velocity tracking
    cmd_ang = unwrapped.command_manager.get_command("base_velocity")[:, 2]
    actual_ang = robot.data.root_ang_vel_b[:, 2]
    ang_tracking_error = torch.abs(cmd_ang - actual_ang)

    temp_for_eval = _temperature_tensor_for_eval(unwrapped)

    metrics = {
        "tracking_error_xy": tracking_error.cpu(),
        "tracking_error_ang": ang_tracking_error.cpu(),
        "avg_temp": torch.mean(temp_for_eval, dim=1).cpu(),
        "max_temp": torch.max(temp_for_eval, dim=1)[0].cpu(),
        "avg_fatigue": torch.mean(deg_state.fatigue_index, dim=1).cpu(),
        "max_fatigue": torch.max(deg_state.fatigue_index, dim=1)[0].cpu(),
        "soc": deg_state.soc.cpu(),
        "battery_voltage": (
            deg_state.battery_voltage_true.cpu()
            if hasattr(deg_state, "battery_voltage_true")
            else deg_state.battery_voltage.cpu()
        ),
        "total_power": torch.sum(deg_state.avg_power_log, dim=1).cpu(),
        "max_saturation": torch.max(deg_state.torque_saturation, dim=1)[0].cpu(),
    }

    # SOH (State of Health)
    soh = deg_state.motor_health_capacity - deg_state.fatigue_index
    metrics["min_soh"] = torch.min(soh, dim=1)[0].cpu()

    return metrics


def collect_cmd_motion_metrics(env, cmd_name: str = "base_velocity") -> dict:
    """
    Collect command and motion evidence on pre-step state.

    Using pre-step tensors avoids reset-mixing artifacts for envs that will be done
    in the subsequent env.step().
    """
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    cmd = unwrapped.command_manager.get_command(cmd_name)
    cmd_xy = cmd[:, :2]
    cmd_speed_xy = torch.norm(cmd_xy, dim=1)
    cmd_ang_abs = torch.abs(cmd[:, 2])

    vel_xy = robot.data.root_lin_vel_b[:, :2]
    actual_speed_xy = torch.norm(vel_xy, dim=1)
    actual_ang_abs = torch.abs(robot.data.root_ang_vel_b[:, 2])

    cmd_dir = cmd_xy / (cmd_speed_xy.unsqueeze(-1) + 1e-6)
    progress_speed = torch.sum(vel_xy * cmd_dir, dim=1)
    progress_speed = torch.where(cmd_speed_xy > 1e-3, progress_speed, torch.zeros_like(progress_speed))

    return {
        "cmd_speed_xy": cmd_speed_xy.detach().cpu(),
        "cmd_ang_abs": cmd_ang_abs.detach().cpu(),
        "actual_speed_xy": actual_speed_xy.detach().cpu(),
        "actual_ang_abs": actual_ang_abs.detach().cpu(),
        "progress_speed_along_cmd": progress_speed.detach().cpu(),
    }


def _quat_wxyz_to_roll_pitch(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert quaternion (w,x,y,z) to roll/pitch (rad)."""
    q = torch.as_tensor(quat_wxyz, dtype=torch.float32)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = torch.clamp(sinp, min=-1.0, max=1.0)
    pitch = torch.asin(sinp)
    return roll, pitch


def _parse_foot_body_names(raw_names: str) -> list[str] | None:
    raw = str(raw_names).strip()
    if raw == "":
        return None
    names = [x.strip() for x in raw.split(",") if x.strip()]
    return names if len(names) > 0 else None


def _get_robot_body_names(base_env) -> list[str] | None:
    robot = base_env.scene["robot"]
    data = getattr(robot, "data", None)
    if data is not None:
        for key in ("body_names", "_body_names", "link_names", "_link_names"):
            names = getattr(data, key, None)
            if names is not None:
                try:
                    return list(names)
                except Exception:
                    pass
    for key in ("body_names", "_body_names", "link_names", "_link_names"):
        names = getattr(robot, key, None)
        if names is not None:
            try:
                return list(names)
            except Exception:
                pass
    return None


def _resolve_eval_foot_body_indices(
    base_env,
    foot_body_names: list[str] | None,
    debug_print: bool = False,
) -> tuple[torch.Tensor | None, list[str], str]:
    body_names = _get_robot_body_names(base_env)
    if body_names is None:
        return None, [], "body_names_unavailable"

    if debug_print:
        print(f"[DEBUG] robot body_names (count={len(body_names)}): {body_names}")

    resolved_names: list[str] = []
    if foot_body_names is None:
        auto_names = [n for n in body_names if ("foot" in n.lower() or "toe" in n.lower())]
        if len(auto_names) < 4:
            return None, [], "auto_detect_failed"
        # Keep original body order and pick first 4 for deterministic behavior.
        resolved_names = auto_names[:4]
    else:
        for name in foot_body_names:
            if name in body_names:
                resolved_names.append(name)
                continue
            low = name.lower()
            matches = [bn for bn in body_names if low in bn.lower()]
            if len(matches) == 1:
                resolved_names.append(matches[0])
            else:
                return None, [], f"cannot_resolve:{name}"

    indices = [body_names.index(n) for n in resolved_names]
    idx_tensor = torch.tensor(indices, device=base_env.device, dtype=torch.long)
    if debug_print:
        print(f"[DEBUG] resolved foot bodies: {resolved_names} -> ids={indices}")
    return idx_tensor, resolved_names, "ok"


def _get_contact_forces_tensor(base_env) -> torch.Tensor | None:
    robot_data = base_env.scene["robot"].data
    for key in ("net_contact_forces_w", "contact_forces_w", "net_contact_forces", "contact_forces"):
        val = getattr(robot_data, key, None)
        if isinstance(val, torch.Tensor) and val.ndim >= 3 and val.shape[-1] >= 3:
            return val
    return None


def _get_body_lin_vel_tensor(base_env) -> torch.Tensor | None:
    robot_data = base_env.scene["robot"].data
    for key in ("body_lin_vel_w", "body_lin_vel", "link_lin_vel_w", "link_lin_vel"):
        val = getattr(robot_data, key, None)
        if isinstance(val, torch.Tensor) and val.ndim >= 3 and val.shape[-1] >= 3:
            return val
    body_state = getattr(robot_data, "body_state_w", None)
    if isinstance(body_state, torch.Tensor) and body_state.ndim >= 3 and body_state.shape[-1] >= 10:
        # Common Isaac format: [pos(3), quat(4), lin_vel(3), ang_vel(3), ...]
        return body_state[..., 7:10]
    return None


def collect_gait_contact_metrics(
    env,
    foot_body_ids: torch.Tensor | None,
    contact_force_threshold: float,
) -> dict[str, Any]:
    base_env = env.unwrapped
    if foot_body_ids is None or int(foot_body_ids.numel()) == 0:
        return {"available": False}

    contact_forces = _get_contact_forces_tensor(base_env)
    body_lin_vel = _get_body_lin_vel_tensor(base_env)
    if contact_forces is None or body_lin_vel is None:
        return {"available": False}

    num_bodies = int(contact_forces.shape[1])
    if int(torch.max(foot_body_ids).item()) >= num_bodies:
        return {"available": False}

    fz = torch.abs(contact_forces[:, foot_body_ids, 2])
    contact = fz > float(contact_force_threshold)
    foot_vel_xy = body_lin_vel[:, foot_body_ids, :2]
    foot_speed_xy = torch.norm(foot_vel_xy, dim=-1)
    return {
        "available": True,
        "contact": contact.detach().cpu(),
        "foot_speed_xy": foot_speed_xy.detach().cpu(),
    }


def _terminal_scalar(terminal_metrics: dict, terminal_lookup: dict[int, int], env_idx: int, key: str) -> float | None:
    """Read terminal snapshot scalar for one environment; return None if unavailable."""
    pos = terminal_lookup.get(env_idx)
    if pos is None:
        return None
    val = terminal_metrics.get(key, None)
    if val is None:
        return None
    return float(val[pos].detach().cpu().item())


def _reset_recurrent_state(policy_nn, num_envs: int, device: torch.device | str):
    """Reset recurrent hidden state across all envs, if the policy supports it."""
    if not hasattr(policy_nn, "reset"):
        return
    done_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    policy_nn.reset(done_mask)


def _safe_reset_recurrent(policy_nn, dones: torch.Tensor):
    """Reset recurrent state only when the policy exposes a reset method."""
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(dones)


def _obs_from_reset_output(reset_output):
    """Normalize Gymnasium reset output across API versions."""
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def run_evaluation(
    env,
    policy,
    policy_nn,
    scenario_name: str,
    num_target_episodes: int,
    temp_metric_semantics: str = "coil_hotspot",
    nonzero_cmd_threshold: float = 0.05,
    stand_cmd_threshold: float = 0.01,
    contact_force_threshold: float = 15.0,
    eval_foot_body_names: list[str] | None = None,
    eval_debug_print_body_names: bool = False,
    cls_stand_actual_max: float = 0.05,
    cls_stand_ratio_min: float = 0.80,
    cls_stand_cmd_ang_abs_max: float = 0.15,
    cls_walk_cmd_min: float = 0.04,
    cls_walk_nz_ratio_min: float = 0.60,
    cls_walk_progress_min: float = 0.05,
    cls_walk_progress_speed_min: float = 0.03,
    cls_walk_progress_ratio_min: float = 0.45,
    cmd_profile_effective: str = "from_env",
    forced_walk_ramp_s: float = 0.0,
    forced_walk_then_zero_walk_s: float = 6.0,
    safe_stop_lin_vel_max: float = 0.05,
    safe_stop_ang_vel_max: float = 0.10,
    safe_stop_hold_s: float = 1.0,
    safe_stop_require_pose: bool = False,
    safe_stop_max_tilt_rad: float = 0.35,
    safe_stop_min_height_m: float = 0.28,
    oracle_safe_stop_steps: int = 0,
    crit_cmd_delta_active_eps: float = 1e-3,
    thermal_threshold_c: float | None = None,
    dump_timeseries: bool = False,
    dump_timeseries_scenario: str = "critical",
    dump_timeseries_env_id: int = 0,
    dump_timeseries_max_steps: int = 2000,
    dump_timeseries_output_dir: str = "./eval_results",
) -> dict:
    """Run evaluation for one scenario, collecting episode-level statistics."""
    print(f"\n{'='*60}")
    print(f"  Evaluating scenario: {scenario_name}")
    print(f"  Description: {SCENARIOS[scenario_name]['description']}")
    print(f"  Target episodes: {num_target_episodes}")
    print(f"{'='*60}")

    episode_metrics = defaultdict(list)
    # Per-env episode step counters
    num_envs = env.num_envs
    base_env = env.unwrapped
    ep_step_counter = torch.zeros(num_envs, dtype=torch.long)
    ep_tracking_errors = [[] for _ in range(num_envs)]
    ep_ang_errors = [[] for _ in range(num_envs)]
    ep_power_history = [[] for _ in range(num_envs)]
    ep_total_reward = torch.zeros(num_envs, dtype=torch.float32)
    dt = float(base_env.step_dt)

    # Episode command-motion evidence (walk vs stand diagnostics).
    ep_cmd_speed_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_cmd_ang_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_actual_speed_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_path_length = torch.zeros(num_envs, dtype=torch.float32)
    ep_progress_signed = torch.zeros(num_envs, dtype=torch.float32)
    ep_progress_forward = torch.zeros(num_envs, dtype=torch.float32)
    ep_nonzero_cmd = torch.zeros(num_envs, dtype=torch.float32)
    ep_stand_cmd = torch.zeros(num_envs, dtype=torch.float32)
    ep_energy_cum_j = torch.zeros(num_envs, dtype=torch.float32)
    ep_peak_temp_c = torch.full((num_envs,), -float("inf"), dtype=torch.float32)
    ep_peak_saturation = torch.zeros(num_envs, dtype=torch.float32)
    ep_safe_stop_consec_steps = torch.zeros(num_envs, dtype=torch.long)
    ep_safe_stop_reached = torch.zeros(num_envs, dtype=torch.bool)
    ep_time_to_safe_stop_s = torch.full((num_envs,), float("nan"), dtype=torch.float32)
    ep_energy_to_safe_stop_j = torch.full((num_envs,), float("nan"), dtype=torch.float32)
    num_joints = int(base_env.motor_deg_state.torque_saturation.shape[1])
    ep_sat_joint_over_0p95_steps = torch.zeros((num_envs, num_joints), dtype=torch.float32)
    ep_sat_joint_over_1p00_steps = torch.zeros((num_envs, num_joints), dtype=torch.float32)
    ep_sat_any_over_1p00_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_sat_any_over_thr_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_sat_window_ratio_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_latched_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_action_delta_latched_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_action_delta_latched_max = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_action_transition_delta_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_action_transition_delta_max = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_cmd_delta_latched_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_cmd_delta_latched_max = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_post_unlatch_ramp_active_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_cmd_delta_active_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_cmd_delta_active_latched_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_sat_ratio_fallback_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_governor_mode_steps = torch.zeros((num_envs, 5), dtype=torch.float32)
    ep_crit_latch_event_count = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_latch_duration_steps_sum = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_latch_open_run_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_crit_prev_latched = torch.zeros(num_envs, dtype=torch.bool)
    ep_crit_first_latch_step = torch.full((num_envs,), -1, dtype=torch.long)
    ep_crit_first_unlatch_step = torch.full((num_envs,), -1, dtype=torch.long)
    ep_crit_first_unlatch_after_zero_step = torch.full((num_envs,), -1, dtype=torch.long)
    ep_crit_sat_ratio_step_history = [[] for _ in range(num_envs)]
    ep_crit_sat_ratio_step_history_latched = [[] for _ in range(num_envs)]
    ep_crit_post_unlatch_sat_history = [[] for _ in range(num_envs)]
    ep_max_saturation_step_history = [[] for _ in range(num_envs)]
    ep_cmd_target = torch.zeros((num_envs, 3), device=base_env.device, dtype=torch.float32)
    governor_mode_names = ("none", "v_cap", "stand", "stop_latch", "other")

    observed_cmd_speed_sum = 0.0
    observed_cmd_speed_count = 0
    observed_cmd_speed_max = 0.0

    # Gait/contact evidence buffers (best-effort; stay robust when unavailable).
    foot_body_ids, resolved_foot_names, foot_resolve_status = _resolve_eval_foot_body_indices(
        env.unwrapped,
        eval_foot_body_names,
        debug_print=eval_debug_print_body_names,
    )
    num_feet = int(foot_body_ids.numel()) if foot_body_ids is not None else 0
    ep_gait_valid_steps = torch.zeros(num_envs, dtype=torch.float32)
    ep_gait_quad_support_steps = torch.zeros(num_envs, dtype=torch.float32)
    if num_feet > 0:
        ep_gait_contact_steps = torch.zeros((num_envs, num_feet), dtype=torch.float32)
        ep_gait_touchdown_count = torch.zeros((num_envs, num_feet), dtype=torch.float32)
        ep_gait_slip_distance = torch.zeros((num_envs, num_feet), dtype=torch.float32)
        prev_contact = torch.zeros((num_envs, num_feet), dtype=torch.bool)
        gait_pre_init = collect_gait_contact_metrics(
            env,
            foot_body_ids=foot_body_ids,
            contact_force_threshold=float(contact_force_threshold),
        )
        if bool(gait_pre_init.get("available", False)):
            prev_contact = gait_pre_init["contact"].clone()
    else:
        ep_gait_contact_steps = None
        ep_gait_touchdown_count = None
        ep_gait_slip_distance = None
        prev_contact = None

    reward_breakdown = _RewardBreakdownCollector(env.unwrapped)
    ep_term_signed = {name: torch.zeros(num_envs, dtype=torch.float32) for name in reward_breakdown.term_names}
    ep_term_abs = {name: torch.zeros(num_envs, dtype=torch.float32) for name in reward_breakdown.term_names}

    # Optional per-step diagnostic dump for one selected env/episode.
    ts_enabled = bool(dump_timeseries) and str(scenario_name) == str(dump_timeseries_scenario)
    ts_env = int(dump_timeseries_env_id)
    ts_max_steps = max(int(dump_timeseries_max_steps), 1)
    ts_rows: list[dict[str, Any]] = []
    ts_dumped = False
    ts_path: str | None = None
    ts_meta_path: str | None = None
    if ts_enabled:
        if not (0 <= ts_env < int(num_envs)):
            print(
                f"[WARN] Timeseries dump disabled: env_id={ts_env} out of range for num_envs={num_envs}."
            )
            ts_enabled = False
        else:
            print(
                f"[INFO] Timeseries dump enabled for scenario={scenario_name}, env_id={ts_env}, "
                f"max_steps={ts_max_steps}."
            )

    def _flush_timeseries(reason: str):
        nonlocal ts_dumped, ts_path, ts_meta_path
        if (not ts_enabled) or ts_dumped or (len(ts_rows) == 0):
            return
        os.makedirs(dump_timeseries_output_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"timeseries_{scenario_name}_env{ts_env}_{stamp}"
        ts_path = os.path.join(dump_timeseries_output_dir, f"{base_name}.csv")
        with open(ts_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(ts_rows[0].keys()))
            writer.writeheader()
            writer.writerows(ts_rows)

        ts_meta_path = os.path.join(dump_timeseries_output_dir, f"{base_name}_meta.json")
        meta = {
            "scenario": str(scenario_name),
            "env_id": int(ts_env),
            "rows": int(len(ts_rows)),
            "dt": float(dt),
            "reason": str(reason),
            "csv_path": str(ts_path),
        }
        with open(ts_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        ts_dumped = True
        print(f"[DONE] Timeseries CSV saved to: {ts_path}")
        print(f"[DONE] Timeseries meta saved to: {ts_meta_path}")

    # Episode-level gait/motion classification thresholds (debug-exported for reproducibility).
    stand_cmd_max = max(0.02, 2.0 * float(stand_cmd_threshold))
    stand_actual_max = float(cls_stand_actual_max)
    stand_ratio_min = float(cls_stand_ratio_min)
    stand_cmd_ang_abs_max = float(cls_stand_cmd_ang_abs_max)
    walk_cmd_min = float(cls_walk_cmd_min)
    walk_nz_ratio_min = float(cls_walk_nz_ratio_min)
    walk_progress_min = float(cls_walk_progress_min)
    walk_progress_speed_min = float(cls_walk_progress_speed_min)
    walk_progress_ratio_min = float(cls_walk_progress_ratio_min)
    post_unlatch_ignore_s = 0.5
    post_unlatch_window_s = 2.0
    post_unlatch_ignore_steps = max(int(round(post_unlatch_ignore_s / max(dt, 1e-6))), 0)
    post_unlatch_window_steps = max(int(round(post_unlatch_window_s / max(dt, 1e-6))), 1)
    safe_stop_lin_vel_max = float(max(safe_stop_lin_vel_max, 0.0))
    safe_stop_ang_vel_max = float(max(safe_stop_ang_vel_max, 0.0))
    safe_stop_hold_s = float(max(safe_stop_hold_s, dt))
    safe_stop_hold_steps = max(int(round(safe_stop_hold_s / max(dt, 1e-6))), 1)
    safe_stop_require_pose = bool(safe_stop_require_pose)
    safe_stop_max_tilt_rad = float(max(safe_stop_max_tilt_rad, 0.0))
    safe_stop_min_height_m = float(max(safe_stop_min_height_m, 0.0))
    oracle_safe_stop_steps = max(int(oracle_safe_stop_steps), 0)
    forced_walk_then_zero_walk_s = float(max(forced_walk_then_zero_walk_s, dt))
    cmd_profile_lower = str(cmd_profile_effective).strip().lower()
    forced_walk_ramp_s = float(max(forced_walk_ramp_s, 0.0))
    forced_walk_ramp_enabled = bool(
        forced_walk_ramp_s > 1e-6 and cmd_profile_lower in {"forced_walk", "forced_walk_then_zero"}
    )
    forced_walk_then_zero_enabled = bool(cmd_profile_lower == "forced_walk_then_zero")
    if forced_walk_ramp_enabled:
        print(
            f"[INFO] Forced-walk ramp enabled: ramp_s={forced_walk_ramp_s:.2f}s "
            f"(cmd_profile={cmd_profile_effective})."
        )
    if forced_walk_then_zero_enabled:
        print(
            f"[INFO] Forced-walk-then-zero enabled: walk_s={forced_walk_then_zero_walk_s:.2f}s "
            f"(cmd_profile={cmd_profile_effective})."
        )
    if oracle_safe_stop_steps > 0:
        print(
            f"[INFO] Oracle safe-stop enabled: forcing zero-action for first "
            f"{oracle_safe_stop_steps} step(s) after reset."
        )

    def _capture_cmd_targets(env_ids: torch.Tensor):
        if (not (forced_walk_ramp_enabled or forced_walk_then_zero_enabled)) or env_ids.numel() == 0:
            return
        cmd_now = base_env.command_manager.get_command("base_velocity")[env_ids]
        ep_cmd_target[env_ids] = cmd_now[:, :3]

    def _apply_eval_cmd_override_per_step():
        if not (forced_walk_ramp_enabled or forced_walk_then_zero_enabled):
            return
        cmd_buf = base_env.command_manager.get_command("base_velocity")
        step_alpha = torch.ones(num_envs, device=base_env.device, dtype=torch.float32)
        if forced_walk_ramp_enabled:
            step_alpha = (
                (ep_step_counter.to(torch.float32).to(device=base_env.device) + 1.0)
                * dt
                / max(forced_walk_ramp_s, dt)
            ).clamp(0.0, 1.0)
        cmd_step = ep_cmd_target * step_alpha.unsqueeze(-1)
        if forced_walk_then_zero_enabled:
            step_time_s = (
                (ep_step_counter.to(torch.float32).to(device=base_env.device) + 1.0) * dt
            )
            walk_mask = (step_time_s <= forced_walk_then_zero_walk_s).unsqueeze(-1)
            cmd_step = torch.where(walk_mask, cmd_step, torch.zeros_like(cmd_step))
        cmd_buf[:, :3] = cmd_step

    completed_episodes = 0
    total_steps = 0
    max_steps = num_target_episodes * 1500  # Safety: prevent infinite loop
    last_reported_episodes = 0
    warned_sat_ratio_fallback = False

    # Reset and inject scenario.
    # This prevents hidden-state/episode carry-over across scenarios.
    obs = _obs_from_reset_output(env.reset())
    _reset_recurrent_state(policy_nn, num_envs=env.num_envs, device=env.unwrapped.device)
    all_env_ids = torch.arange(num_envs, device=env.unwrapped.device)
    apply_scenario(env, scenario_name, all_env_ids)
    # Ensure first action uses observations consistent with injected scenario.
    obs = env.get_observations()
    _capture_cmd_targets(all_env_ids)

    while completed_episodes < num_target_episodes and total_steps < max_steps:
        _apply_eval_cmd_override_per_step()

        # Pre-step command-motion capture (avoids done/reset mixing from env.step()).
        pre = collect_cmd_motion_metrics(env, cmd_name="base_velocity")
        cmd_speed_xy = pre["cmd_speed_xy"]
        cmd_ang_abs = pre["cmd_ang_abs"]
        actual_speed_xy = pre["actual_speed_xy"]
        actual_ang_abs = pre["actual_ang_abs"]
        progress_speed = pre["progress_speed_along_cmd"]

        ep_cmd_speed_sum += cmd_speed_xy
        ep_cmd_ang_sum += cmd_ang_abs
        ep_actual_speed_sum += actual_speed_xy
        ep_path_length += actual_speed_xy * dt
        ep_progress_signed += progress_speed * dt
        ep_progress_forward += torch.clamp(progress_speed, min=0.0) * dt
        ep_nonzero_cmd += (cmd_speed_xy > float(nonzero_cmd_threshold)).to(torch.float32)
        ep_stand_cmd += (cmd_speed_xy < float(stand_cmd_threshold)).to(torch.float32)
        safe_stop_pose_mask = torch.ones_like(actual_speed_xy, dtype=torch.bool)
        if safe_stop_require_pose:
            robot_data = base_env.scene["robot"].data
            if hasattr(robot_data, "root_quat_w"):
                roll_all, pitch_all = _quat_wxyz_to_roll_pitch(robot_data.root_quat_w)
                pose_tilt_ok = (
                    (torch.abs(roll_all) <= safe_stop_max_tilt_rad)
                    & (torch.abs(pitch_all) <= safe_stop_max_tilt_rad)
                ).detach().cpu()
                safe_stop_pose_mask &= pose_tilt_ok
            if hasattr(robot_data, "root_pos_w"):
                root_h_ok = (robot_data.root_pos_w[:, 2] >= safe_stop_min_height_m).detach().cpu()
                safe_stop_pose_mask &= root_h_ok
        safe_stop_step_mask = (
            (actual_speed_xy <= safe_stop_lin_vel_max)
            & (actual_ang_abs <= safe_stop_ang_vel_max)
            & safe_stop_pose_mask
        )
        ep_safe_stop_consec_steps = torch.where(
            safe_stop_step_mask,
            ep_safe_stop_consec_steps + 1,
            torch.zeros_like(ep_safe_stop_consec_steps),
        )
        newly_safe = (~ep_safe_stop_reached) & (ep_safe_stop_consec_steps >= safe_stop_hold_steps)
        if torch.any(newly_safe):
            ep_safe_stop_reached[newly_safe] = True
            ep_time_to_safe_stop_s[newly_safe] = (
                ep_step_counter[newly_safe].to(torch.float32) + 1.0
            ) * dt
            ep_energy_to_safe_stop_j[newly_safe] = ep_energy_cum_j[newly_safe]

        observed_cmd_speed_sum += float(torch.sum(cmd_speed_xy).item())
        observed_cmd_speed_count += int(cmd_speed_xy.numel())
        observed_cmd_speed_max = max(observed_cmd_speed_max, float(torch.max(cmd_speed_xy).item()))
        ts_contact_pre = None
        ts_foot_speed_pre = None

        sat_step = base_env.motor_deg_state.torque_saturation.detach().cpu().to(torch.float32)
        ep_sat_joint_over_0p95_steps += (sat_step > 0.95).to(torch.float32)
        ep_sat_joint_over_1p00_steps += (sat_step > 1.00).to(torch.float32)
        ep_sat_any_over_1p00_steps += torch.any(sat_step > 1.00, dim=1).to(torch.float32)

        # Pre-step gait/contact capture (best-effort).
        if num_feet > 0 and ep_gait_contact_steps is not None and prev_contact is not None:
            gait = collect_gait_contact_metrics(
                env,
                foot_body_ids=foot_body_ids,
                contact_force_threshold=float(contact_force_threshold),
            )
            if bool(gait.get("available", False)):
                contact = gait["contact"]
                foot_speed_xy = gait["foot_speed_xy"]
                touchdown = (~prev_contact) & contact
                ep_gait_touchdown_count += touchdown.to(torch.float32)
                ep_gait_contact_steps += contact.to(torch.float32)
                ep_gait_quad_support_steps += torch.all(contact, dim=1).to(torch.float32)
                ep_gait_slip_distance += contact.to(torch.float32) * foot_speed_xy * dt
                ep_gait_valid_steps += 1.0
                prev_contact = contact
                if ts_enabled and (not ts_dumped):
                    ts_contact_pre = contact[ts_env].tolist()
                    ts_foot_speed_pre = foot_speed_xy[ts_env].tolist()

        ts_row_pre: dict[str, Any] | None = None
        if ts_enabled and (not ts_dumped) and (len(ts_rows) < ts_max_steps):
            robot = base_env.scene["robot"]
            robot_data = robot.data
            deg_state = base_env.motor_deg_state
            env_i = int(ts_env)

            cmd_now = base_env.command_manager.get_command("base_velocity")[env_i]
            root_vel_xy = robot_data.root_lin_vel_b[env_i, :2]
            root_ang_z = robot_data.root_ang_vel_b[env_i, 2]
            root_pos_z = robot_data.root_pos_w[env_i, 2] if hasattr(robot_data, "root_pos_w") else torch.tensor(float("nan"))
            root_quat = robot_data.root_quat_w[env_i] if hasattr(robot_data, "root_quat_w") else torch.tensor([1.0, 0.0, 0.0, 0.0])
            roll, pitch = _quat_wxyz_to_roll_pitch(root_quat)

            temp_tensor = _temperature_tensor_for_eval(base_env)[env_i]
            sat_tensor = deg_state.torque_saturation[env_i]
            power_w = float(torch.sum(deg_state.avg_power_log[env_i]).detach().cpu().item())

            ts_row_pre = {
                "global_step": int(total_steps),
                "episode_step": int(ep_step_counter[env_i].item()),
                "t_s": float(ep_step_counter[env_i].item() * dt),
                "cmd_vx": float(cmd_now[0].detach().cpu().item()),
                "cmd_vy": float(cmd_now[1].detach().cpu().item()),
                "cmd_wz": float(cmd_now[2].detach().cpu().item()),
                "base_speed_xy": float(torch.norm(root_vel_xy).detach().cpu().item()),
                "base_wz_abs": float(torch.abs(root_ang_z).detach().cpu().item()),
                "base_roll_rad": float(roll.detach().cpu().item()),
                "base_pitch_rad": float(pitch.detach().cpu().item()),
                "base_height_m": float(root_pos_z.detach().cpu().item()),
                "temp_avg_c": float(torch.mean(temp_tensor).detach().cpu().item()),
                "temp_max_c": float(torch.max(temp_tensor).detach().cpu().item()),
                "sat_mean": float(torch.mean(sat_tensor).detach().cpu().item()),
                "sat_max": float(torch.max(sat_tensor).detach().cpu().item()),
                "sat_over_0p95_count": int(torch.sum(sat_tensor > 0.95).detach().cpu().item()),
                "sat_over_1p00_count": int(torch.sum(sat_tensor > 1.00).detach().cpu().item()),
                "power_w": float(power_w),
            }
            sat_vals = sat_tensor.detach().cpu().to(torch.float32).tolist()
            for j, sat_v in enumerate(sat_vals):
                ts_row_pre[f"sat_j{j:02d}"] = float(sat_v)
            if ts_contact_pre is not None and len(ts_contact_pre) > 0:
                contact_count = 0
                for k, v in enumerate(ts_contact_pre):
                    foot_name = resolved_foot_names[k] if k < len(resolved_foot_names) else f"foot_{k}"
                    safe_name = "".join(c if (c.isalnum() or c == "_") else "_" for c in foot_name)
                    ts_row_pre[f"contact_{safe_name}"] = int(bool(v))
                    if ts_foot_speed_pre is not None and k < len(ts_foot_speed_pre):
                        ts_row_pre[f"foot_speed_xy_{safe_name}"] = float(ts_foot_speed_pre[k])
                    contact_count += int(bool(v))
                ts_row_pre["contact_count"] = int(contact_count)
            else:
                ts_row_pre["contact_count"] = int(-1)

        # NOTE:
        # Keep env.step() out of torch.inference_mode(). Otherwise internal simulator
        # buffers can become inference tensors and later fail on reset() with
        # "Inplace update to inference tensor outside InferenceMode".
        with torch.no_grad():
            actions = policy(obs)
        # Defensive clone: if policy wrapper returns inference tensors internally,
        # convert to a regular tensor before passing into the environment.
        actions = actions.clone()
        if oracle_safe_stop_steps > 0:
            oracle_mask = ep_step_counter < oracle_safe_stop_steps
            if bool(torch.any(oracle_mask).item()):
                actions[oracle_mask] = 0.0
        obs, rewards, dones, _ = env.step(actions)
        dones = torch.as_tensor(dones, device=env.unwrapped.device, dtype=torch.bool)
        if dones.ndim > 1:
            dones = torch.any(dones, dim=tuple(range(1, dones.ndim)))
        _safe_reset_recurrent(policy_nn, dones)
        if ts_row_pre is not None and ts_enabled and (not ts_dumped):
            ts_row_pre["done_after_step"] = int(bool(dones[ts_env].detach().cpu().item()))
            ts_rows.append(ts_row_pre)
            if len(ts_rows) >= ts_max_steps:
                _flush_timeseries(reason="max_steps_reached")

        # Collect step metrics
        step_metrics = collect_step_metrics(env)
        step_reward = reward_breakdown._as_reward_vector(rewards).detach().cpu()
        ep_total_reward += step_reward

        step_term_contrib = reward_breakdown.compute_step_contributions(reward_vec=rewards)
        done_mask_cpu = dones.detach().cpu()
        for term_name, term_val in step_term_contrib.items():
            term_cpu = term_val.detach().cpu().to(torch.float32)
            # done envs are already reset inside env.step(); final-step term values come from terminal snapshot.
            term_cpu[done_mask_cpu] = 0.0
            ep_term_signed[term_name] += term_cpu
            ep_term_abs[term_name] += torch.abs(term_cpu)

        terminal_metrics = getattr(env.unwrapped, "_last_terminal_metrics", {})
        terminal_ids = getattr(env.unwrapped, "_last_terminal_env_ids", None)
        terminal_lookup: dict[int, int] = {}
        if isinstance(terminal_ids, torch.Tensor) and terminal_ids.numel() > 0 and isinstance(terminal_metrics, dict):
            terminal_lookup = {int(env_id.item()): i for i, env_id in enumerate(terminal_ids.detach().cpu())}

        # Governor per-step diagnostics (episode-level mean decomposition).
        crit_sat_any_step = torch.zeros(num_envs, dtype=torch.float32)
        crit_sat_ratio_step = torch.zeros(num_envs, dtype=torch.float32)
        latched_step_mask = torch.zeros(num_envs, dtype=torch.float32)
        action_delta_step = torch.zeros(num_envs, dtype=torch.float32)
        action_transition_delta_step = torch.zeros(num_envs, dtype=torch.float32)
        cmd_delta_step = torch.zeros(num_envs, dtype=torch.float32)
        post_unlatch_ramp_active_step = torch.zeros(num_envs, dtype=torch.float32)
        governor_mode_step = torch.zeros(num_envs, dtype=torch.long)
        raw_crit_any = getattr(base_env, "_crit_sat_any", None)
        raw_crit_ratio = None
        raw_crit_ratio_legacy = getattr(base_env, "_crit_sat_ratio", None)
        raw_governor_mode = getattr(base_env, "_crit_governor_mode_step", None)
        sat_latch_obj = getattr(base_env, "_crit_sat_latch", None)
        if sat_latch_obj is not None and hasattr(sat_latch_obj, "ratio"):
            raw_crit_ratio = sat_latch_obj.ratio
        if isinstance(raw_crit_ratio, torch.Tensor) and isinstance(raw_crit_ratio_legacy, torch.Tensor):
            if raw_crit_ratio.shape == raw_crit_ratio_legacy.shape:
                raw_crit_ratio = torch.maximum(raw_crit_ratio, raw_crit_ratio_legacy)
        elif raw_crit_ratio is None:
            raw_crit_ratio = raw_crit_ratio_legacy
        if isinstance(raw_crit_any, torch.Tensor) and raw_crit_any.ndim == 1 and raw_crit_any.shape[0] == num_envs:
            crit_sat_any_step = raw_crit_any.detach().cpu().to(torch.float32)
        if isinstance(raw_crit_ratio, torch.Tensor) and raw_crit_ratio.ndim == 1 and raw_crit_ratio.shape[0] == num_envs:
            crit_sat_ratio_step = raw_crit_ratio.detach().cpu().to(torch.float32)
        if (
            isinstance(raw_governor_mode, torch.Tensor)
            and raw_governor_mode.ndim == 1
            and raw_governor_mode.shape[0] == num_envs
        ):
            governor_mode_step = raw_governor_mode.detach().cpu().to(torch.long).clamp_(0, 4)
        # Done envs are reset inside env.step(); restore terminal-step values from snapshot.
        if len(terminal_lookup) > 0:
            term_any_vec = terminal_metrics.get("crit/sat_any_over_thr_step", None)
            term_ratio_vec = terminal_metrics.get("crit/sat_any_over_thr_ratio", None)
            term_latched_vec = terminal_metrics.get("crit/is_latched", None)
            term_action_delta_vec = terminal_metrics.get("crit/action_delta_norm_step", None)
            term_action_transition_delta_vec = terminal_metrics.get("crit/action_transition_delta_norm_step", None)
            term_cmd_delta_vec = terminal_metrics.get("crit/cmd_delta_norm_step", None)
            term_mode_vec = terminal_metrics.get("crit/governor_mode_step", None)
            term_post_unlatch_ramp_active_vec = terminal_metrics.get(
                "crit/post_unlatch_action_ramp_active_step", None
            )
            for env_idx, pos in terminal_lookup.items():
                if isinstance(term_any_vec, torch.Tensor):
                    crit_sat_any_step[env_idx] = float(term_any_vec[pos].detach().cpu().item())
                if isinstance(term_ratio_vec, torch.Tensor):
                    crit_sat_ratio_step[env_idx] = float(term_ratio_vec[pos].detach().cpu().item())
                if isinstance(term_latched_vec, torch.Tensor):
                    latched_step_mask[env_idx] = float(term_latched_vec[pos].detach().cpu().item())
                if isinstance(term_action_delta_vec, torch.Tensor):
                    action_delta_step[env_idx] = float(term_action_delta_vec[pos].detach().cpu().item())
                if isinstance(term_action_transition_delta_vec, torch.Tensor):
                    action_transition_delta_step[env_idx] = float(
                        term_action_transition_delta_vec[pos].detach().cpu().item()
                    )
                if isinstance(term_cmd_delta_vec, torch.Tensor):
                    cmd_delta_step[env_idx] = float(term_cmd_delta_vec[pos].detach().cpu().item())
                if isinstance(term_mode_vec, torch.Tensor):
                    governor_mode_step[env_idx] = int(term_mode_vec[pos].detach().cpu().item())
                if isinstance(term_post_unlatch_ramp_active_vec, torch.Tensor):
                    post_unlatch_ramp_active_step[env_idx] = float(
                        term_post_unlatch_ramp_active_vec[pos].detach().cpu().item()
                    )
        # Per-step governor-action diagnostics (latched-step aggregation).
        raw_latched = (getattr(base_env, "_crit_latch_steps_remaining", None), getattr(base_env, "_crit_need_unlatch", None))
        raw_action_delta = getattr(base_env, "_crit_action_delta_norm_step", None)
        raw_action_transition_delta = getattr(base_env, "_crit_action_transition_delta_norm_step", None)
        raw_cmd_delta = getattr(base_env, "_crit_cmd_delta_norm_step", None)
        raw_post_unlatch_ramp_steps_remaining = getattr(
            base_env, "_crit_post_unlatch_action_ramp_steps_remaining", None
        )
        if (
            isinstance(raw_latched[0], torch.Tensor)
            and isinstance(raw_latched[1], torch.Tensor)
            and isinstance(raw_action_delta, torch.Tensor)
            and isinstance(raw_action_transition_delta, torch.Tensor)
            and isinstance(raw_cmd_delta, torch.Tensor)
            and raw_action_delta.ndim == 1
            and raw_action_transition_delta.ndim == 1
            and raw_cmd_delta.ndim == 1
            and raw_action_delta.shape[0] == num_envs
            and raw_action_transition_delta.shape[0] == num_envs
            and raw_cmd_delta.shape[0] == num_envs
        ):
            live_latched = ((raw_latched[0] > 0) | raw_latched[1]).detach().cpu().to(torch.float32)
            live_action_delta = raw_action_delta.detach().cpu().to(torch.float32)
            live_action_transition_delta = raw_action_transition_delta.detach().cpu().to(torch.float32)
            live_cmd_delta = raw_cmd_delta.detach().cpu().to(torch.float32)
            if (
                isinstance(raw_post_unlatch_ramp_steps_remaining, torch.Tensor)
                and raw_post_unlatch_ramp_steps_remaining.ndim == 1
                and raw_post_unlatch_ramp_steps_remaining.shape[0] == num_envs
            ):
                live_post_unlatch_ramp_active = (
                    raw_post_unlatch_ramp_steps_remaining > 0
                ).detach().cpu().to(torch.float32)
            else:
                live_post_unlatch_ramp_active = torch.zeros(num_envs, dtype=torch.float32)
            # Preserve terminal-step overrides for done envs (env.step reset contamination guard).
            live_latched[done_mask_cpu] = latched_step_mask[done_mask_cpu]
            live_action_delta[done_mask_cpu] = action_delta_step[done_mask_cpu]
            live_action_transition_delta[done_mask_cpu] = action_transition_delta_step[done_mask_cpu]
            live_cmd_delta[done_mask_cpu] = cmd_delta_step[done_mask_cpu]
            live_post_unlatch_ramp_active[done_mask_cpu] = post_unlatch_ramp_active_step[done_mask_cpu]
            latched_step_mask = live_latched
            action_delta_step = live_action_delta
            action_transition_delta_step = live_action_transition_delta
            cmd_delta_step = live_cmd_delta
            post_unlatch_ramp_active_step = live_post_unlatch_ramp_active
            if (
                isinstance(raw_governor_mode, torch.Tensor)
                and raw_governor_mode.ndim == 1
                and raw_governor_mode.shape[0] == num_envs
            ):
                live_mode = raw_governor_mode.detach().cpu().to(torch.long).clamp_(0, 4)
                live_mode[done_mask_cpu] = governor_mode_step[done_mask_cpu]
                governor_mode_step = live_mode

        # Defensive fallback: if ratio signal is unexpectedly zero while sat_any is active,
        # avoid reporting a misleading all-zero sat-window metric.
        ratio_fallback_mask = (crit_sat_ratio_step <= 1e-6) & (crit_sat_any_step > 0.5)
        if bool(torch.any(ratio_fallback_mask).item()):
            crit_sat_ratio_step = torch.where(ratio_fallback_mask, crit_sat_any_step, crit_sat_ratio_step)
            if not warned_sat_ratio_fallback:
                n_fallback = int(torch.sum(ratio_fallback_mask).item())
                print(
                    f"[WARN] crit_sat_ratio_step fallback engaged for {n_fallback}/{num_envs} env(s): "
                    "ratio signal was 0 while sat_any was 1.",
                    flush=True,
                )
                warned_sat_ratio_fallback = True

        ep_crit_sat_any_over_thr_steps += crit_sat_any_step
        ep_crit_sat_window_ratio_sum += crit_sat_ratio_step
        ep_crit_sat_ratio_fallback_steps += ratio_fallback_mask.to(torch.float32)
        for mode_i in range(5):
            ep_crit_governor_mode_steps[:, mode_i] += (governor_mode_step == mode_i).to(torch.float32)

        ep_crit_latched_steps += latched_step_mask
        ep_crit_action_delta_latched_sum += action_delta_step * latched_step_mask
        ep_crit_cmd_delta_latched_sum += cmd_delta_step * latched_step_mask
        ep_crit_action_transition_delta_sum += action_transition_delta_step
        ep_crit_post_unlatch_ramp_active_steps += post_unlatch_ramp_active_step
        ep_crit_action_delta_latched_max = torch.maximum(
            ep_crit_action_delta_latched_max, action_delta_step * latched_step_mask
        )
        ep_crit_action_transition_delta_max = torch.maximum(
            ep_crit_action_transition_delta_max, action_transition_delta_step
        )
        ep_crit_cmd_delta_latched_max = torch.maximum(
            ep_crit_cmd_delta_latched_max, cmd_delta_step * latched_step_mask
        )
        cmd_delta_active_step = (cmd_delta_step > float(crit_cmd_delta_active_eps)).to(torch.float32)
        ep_crit_cmd_delta_active_steps += cmd_delta_active_step
        ep_crit_cmd_delta_active_latched_steps += cmd_delta_active_step * latched_step_mask

        latched_bool = latched_step_mask > 0.5
        latch_started = (~ep_crit_prev_latched) & latched_bool
        latch_ended = ep_crit_prev_latched & (~latched_bool)
        step_index_now = ep_step_counter + 1
        first_latch_mask = (ep_crit_first_latch_step < 0) & latch_started
        ep_crit_first_latch_step[first_latch_mask] = step_index_now[first_latch_mask]
        first_unlatch_mask = (
            (ep_crit_first_unlatch_step < 0)
            & latch_ended
            & (ep_crit_first_latch_step >= 0)
        )
        ep_crit_first_unlatch_step[first_unlatch_mask] = step_index_now[first_unlatch_mask]
        if forced_walk_then_zero_enabled:
            step_time_now_s = step_index_now.to(torch.float32) * float(dt)
            unlatch_after_zero_mask = latch_ended & (step_time_now_s >= float(forced_walk_then_zero_walk_s))
            first_unlatch_after_zero_mask = (
                (ep_crit_first_unlatch_after_zero_step < 0)
                & unlatch_after_zero_mask
                & (ep_crit_first_latch_step >= 0)
            )
            ep_crit_first_unlatch_after_zero_step[first_unlatch_after_zero_mask] = step_index_now[
                first_unlatch_after_zero_mask
            ]
        ep_crit_latch_event_count += latch_started.to(torch.float32)
        ep_crit_latch_open_run_steps = torch.where(
            latched_bool,
            ep_crit_latch_open_run_steps + 1.0,
            ep_crit_latch_open_run_steps,
        )
        if torch.any(latch_ended):
            ended_float = latch_ended.to(torch.float32)
            ep_crit_latch_duration_steps_sum += ep_crit_latch_open_run_steps * ended_float
            ep_crit_latch_open_run_steps[latch_ended] = 0.0
        ep_crit_prev_latched = latched_bool

        crit_sat_ratio_np = crit_sat_ratio_step.detach().cpu().numpy()
        latched_np = latched_step_mask.detach().cpu().numpy() > 0.5
        for env_idx in range(num_envs):
            sat_val = float(crit_sat_ratio_np[env_idx])
            ep_crit_sat_ratio_step_history[env_idx].append(sat_val)
            if latched_np[env_idx]:
                ep_crit_sat_ratio_step_history_latched[env_idx].append(sat_val)

        total_steps += 1
        ep_step_counter += 1

        # Track per-env step data
        for i in range(num_envs):
            trk_xy = step_metrics["tracking_error_xy"][i].item()
            trk_ang = step_metrics["tracking_error_ang"][i].item()
            total_power = step_metrics["total_power"][i].item()
            max_temp = step_metrics["max_temp"][i].item()
            max_saturation = step_metrics["max_saturation"][i].item()
            # Use terminal cache when available (env.step resets done envs before returning).
            trk_xy_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "tracking_error_xy")
            trk_ang_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "tracking_error_ang")
            power_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "total_power")
            temp_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "max_temp")
            sat_terminal = _terminal_scalar(terminal_metrics, terminal_lookup, i, "max_saturation")
            if trk_xy_terminal is not None:
                trk_xy = trk_xy_terminal
            if trk_ang_terminal is not None:
                trk_ang = trk_ang_terminal
            if power_terminal is not None:
                total_power = power_terminal
            if temp_terminal is not None:
                max_temp = temp_terminal
            if sat_terminal is not None:
                max_saturation = sat_terminal
            ep_tracking_errors[i].append(trk_xy)
            ep_ang_errors[i].append(trk_ang)
            ep_power_history[i].append(total_power)
            ep_energy_cum_j[i] += float(total_power) * dt
            if float(max_temp) > float(ep_peak_temp_c[i].item()):
                ep_peak_temp_c[i] = float(max_temp)
            if float(max_saturation) > float(ep_peak_saturation[i].item()):
                ep_peak_saturation[i] = float(max_saturation)
            ep_max_saturation_step_history[i].append(float(max_saturation))
            if int(ep_crit_first_unlatch_step[i].item()) >= 0:
                ep_crit_post_unlatch_sat_history[i].append(float(max_saturation))

        # Process completed episodes
        done_envs = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_envs) > 0:
            for idx_t in done_envs:
                idx = idx_t.item()
                ep_len = ep_step_counter[idx].item()
                if ep_len < 2:
                    if ts_enabled and (not ts_dumped) and idx == ts_env:
                        _flush_timeseries(reason="episode_done_too_short")
                    ep_step_counter[idx] = 0
                    ep_total_reward[idx] = 0.0
                    ep_tracking_errors[idx] = []
                    ep_ang_errors[idx] = []
                    ep_power_history[idx] = []
                    ep_cmd_speed_sum[idx] = 0.0
                    ep_cmd_ang_sum[idx] = 0.0
                    ep_actual_speed_sum[idx] = 0.0
                    ep_path_length[idx] = 0.0
                    ep_progress_signed[idx] = 0.0
                    ep_progress_forward[idx] = 0.0
                    ep_nonzero_cmd[idx] = 0.0
                    ep_stand_cmd[idx] = 0.0
                    ep_energy_cum_j[idx] = 0.0
                    ep_peak_temp_c[idx] = -float("inf")
                    ep_peak_saturation[idx] = 0.0
                    ep_safe_stop_consec_steps[idx] = 0
                    ep_safe_stop_reached[idx] = False
                    ep_time_to_safe_stop_s[idx] = float("nan")
                    ep_energy_to_safe_stop_j[idx] = float("nan")
                    ep_sat_joint_over_0p95_steps[idx] = 0.0
                    ep_sat_joint_over_1p00_steps[idx] = 0.0
                    ep_sat_any_over_1p00_steps[idx] = 0.0
                    ep_crit_sat_any_over_thr_steps[idx] = 0.0
                    ep_crit_sat_window_ratio_sum[idx] = 0.0
                    ep_crit_latched_steps[idx] = 0.0
                    ep_crit_action_delta_latched_sum[idx] = 0.0
                    ep_crit_action_delta_latched_max[idx] = 0.0
                    ep_crit_action_transition_delta_sum[idx] = 0.0
                    ep_crit_action_transition_delta_max[idx] = 0.0
                    ep_crit_cmd_delta_latched_sum[idx] = 0.0
                    ep_crit_cmd_delta_latched_max[idx] = 0.0
                    ep_crit_post_unlatch_ramp_active_steps[idx] = 0.0
                    ep_crit_cmd_delta_active_steps[idx] = 0.0
                    ep_crit_cmd_delta_active_latched_steps[idx] = 0.0
                    ep_crit_sat_ratio_fallback_steps[idx] = 0.0
                    ep_crit_governor_mode_steps[idx] = 0.0
                    ep_crit_latch_event_count[idx] = 0.0
                    ep_crit_latch_duration_steps_sum[idx] = 0.0
                    ep_crit_latch_open_run_steps[idx] = 0.0
                    ep_crit_prev_latched[idx] = False
                    ep_crit_first_latch_step[idx] = -1
                    ep_crit_first_unlatch_step[idx] = -1
                    ep_crit_first_unlatch_after_zero_step[idx] = -1
                    ep_crit_sat_ratio_step_history[idx] = []
                    ep_crit_sat_ratio_step_history_latched[idx] = []
                    ep_crit_post_unlatch_sat_history[idx] = []
                    ep_max_saturation_step_history[idx] = []
                    ep_gait_valid_steps[idx] = 0.0
                    ep_gait_quad_support_steps[idx] = 0.0
                    if num_feet > 0 and ep_gait_contact_steps is not None and ep_gait_touchdown_count is not None and ep_gait_slip_distance is not None:
                        ep_gait_contact_steps[idx] = 0.0
                        ep_gait_touchdown_count[idx] = 0.0
                        ep_gait_slip_distance[idx] = 0.0
                        if prev_contact is not None:
                            prev_contact[idx] = False
                    for term_name in reward_breakdown.term_names:
                        ep_term_signed[term_name][idx] = 0.0
                        ep_term_abs[term_name][idx] = 0.0
                    continue  # Skip trivially short episodes

                # Aggregate episode stats
                final_soc = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "soc")
                final_avg_temp = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "avg_temp")
                final_max_temp = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "max_temp")
                final_max_fatigue = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "max_fatigue")
                final_min_soh = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "min_soh")
                final_max_saturation = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "max_saturation")

                episode_metrics["episode_length"].append(ep_len)
                episode_metrics["mean_total_reward"].append((ep_total_reward[idx].item() / max(ep_len, 1)))
                episode_metrics["mean_tracking_error_xy"].append(np.mean(ep_tracking_errors[idx]))
                episode_metrics["mean_tracking_error_ang"].append(np.mean(ep_ang_errors[idx]))
                episode_metrics["mean_power"].append(np.mean(ep_power_history[idx]))
                episode_metrics["total_energy"].append(np.sum(ep_power_history[idx]) * env.unwrapped.step_dt)
                mean_cmd_speed = float((ep_cmd_speed_sum[idx] / max(ep_len, 1)).item())
                mean_actual_speed = float((ep_actual_speed_sum[idx] / max(ep_len, 1)).item())
                mean_progress_speed_signed = float((ep_progress_signed[idx] / max(ep_len, 1) / dt).item())
                mean_progress_speed_forward = float((ep_progress_forward[idx] / max(ep_len, 1) / dt).item())
                episode_metrics["mean_cmd_speed_xy"].append(mean_cmd_speed)
                episode_metrics["mean_cmd_ang_abs"].append(float((ep_cmd_ang_sum[idx] / max(ep_len, 1)).item()))
                episode_metrics["mean_actual_speed_xy"].append(mean_actual_speed)
                episode_metrics["path_length_xy"].append(float(ep_path_length[idx].item()))
                episode_metrics["progress_distance_along_cmd_signed"].append(float(ep_progress_signed[idx].item()))
                episode_metrics["progress_distance_along_cmd_forward"].append(float(ep_progress_forward[idx].item()))
                episode_metrics["mean_progress_speed_along_cmd_signed"].append(mean_progress_speed_signed)
                episode_metrics["mean_progress_speed_along_cmd_forward"].append(mean_progress_speed_forward)
                episode_metrics["progress_distance_along_cmd"].append(float(ep_progress_forward[idx].item()))
                episode_metrics["nonzero_cmd_ratio"].append(float((ep_nonzero_cmd[idx] / max(ep_len, 1)).item()))
                episode_metrics["stand_cmd_ratio"].append(float((ep_stand_cmd[idx] / max(ep_len, 1)).item()))
                episode_metrics["speed_ratio_actual_over_cmd"].append(
                    mean_actual_speed / max(mean_cmd_speed, 1e-6)
                )
                episode_metrics["final_soc"].append(
                    final_soc if final_soc is not None else step_metrics["soc"][idx].item()
                )
                final_avg_temp_val = final_avg_temp if final_avg_temp is not None else step_metrics["avg_temp"][idx].item()
                final_max_temp_val = final_max_temp if final_max_temp is not None else step_metrics["max_temp"][idx].item()
                # Keep legacy keys for backward compatibility, and add explicit semantic keys.
                avg_temp_key = f"final_avg_temp_{temp_metric_semantics}"
                max_temp_key = f"final_max_temp_{temp_metric_semantics}"
                episode_metrics["final_avg_temp"].append(final_avg_temp_val)
                episode_metrics[avg_temp_key].append(final_avg_temp_val)
                episode_metrics["final_max_temp"].append(final_max_temp_val)
                episode_metrics[max_temp_key].append(final_max_temp_val)
                episode_metrics["final_max_fatigue"].append(
                    final_max_fatigue if final_max_fatigue is not None else step_metrics["max_fatigue"][idx].item()
                )
                episode_metrics["final_min_soh"].append(
                    final_min_soh if final_min_soh is not None else step_metrics["min_soh"][idx].item()
                )
                episode_metrics["max_saturation"].append(
                    final_max_saturation if final_max_saturation is not None else step_metrics["max_saturation"][idx].item()
                )

                # Survival: did it survive the full episode?
                max_ep_steps = int(env.unwrapped.cfg.episode_length_s / env.unwrapped.step_dt)
                survived_flag = 1.0 if ep_len >= max_ep_steps - 1 else 0.0
                episode_metrics["survived"].append(survived_flag)
                if ts_enabled and (not ts_dumped) and idx == ts_env:
                    ts_reason = "episode_done_timeout_survived" if survived_flag >= 0.5 else "episode_done_terminated"
                    _flush_timeseries(reason=ts_reason)

                # Critical safety-mode evidence:
                # safe-stop success requires sustained low linear speed and low yaw rate.
                safe_stop_reached = bool(ep_safe_stop_reached[idx].item())
                safe_stop_success = 1.0 if safe_stop_reached else 0.0
                time_to_safe_stop = float(ep_time_to_safe_stop_s[idx].item()) if safe_stop_reached else float("nan")
                energy_to_safe_stop = float(ep_energy_to_safe_stop_j[idx].item()) if safe_stop_reached else float("nan")
                time_to_end_s = float(ep_len * dt)
                time_to_fall_s = time_to_end_s if survived_flag < 0.5 else float("nan")
                peak_temp_ep = float(ep_peak_temp_c[idx].item())
                if not np.isfinite(peak_temp_ep):
                    peak_temp_ep = final_max_temp_val
                thermal_margin = (
                    float(thermal_threshold_c) - peak_temp_ep
                    if thermal_threshold_c is not None
                    else float("nan")
                )

                episode_metrics["safe_stop_success"].append(safe_stop_success)
                episode_metrics["safe_stop_success_rate"].append(safe_stop_success)
                episode_metrics["time_to_safe_stop_s"].append(time_to_safe_stop)
                episode_metrics["time_to_fall_s"].append(time_to_fall_s)
                episode_metrics["time_to_episode_end_s"].append(time_to_end_s)
                episode_metrics["energy_to_safe_stop_j"].append(energy_to_safe_stop)
                episode_metrics["energy_to_episode_end_j"].append(float(ep_energy_cum_j[idx].item()))
                episode_metrics["peak_temp_episode_c"].append(peak_temp_ep)
                episode_metrics["thermal_margin_to_threshold_c"].append(float(thermal_margin))
                episode_metrics["peak_saturation_episode"].append(float(ep_peak_saturation[idx].item()))
                ep_len_safe = max(float(ep_len), 1.0)
                sat_ratio_any = float(ep_sat_any_over_1p00_steps[idx].item() / ep_len_safe)
                episode_metrics["sat_any_over_1p00_ratio"].append(sat_ratio_any)
                crit_sat_any_ratio_ep = float(ep_crit_sat_any_over_thr_steps[idx].item() / ep_len_safe)
                crit_sat_window_ratio_ep = float(ep_crit_sat_window_ratio_sum[idx].item() / ep_len_safe)
                crit_latched_ratio_ep = float(ep_crit_latched_steps[idx].item() / ep_len_safe)
                latched_steps_raw = float(ep_crit_latched_steps[idx].item())
                latched_steps = max(latched_steps_raw, 1.0)
                latch_event_count_ep = float(ep_crit_latch_event_count[idx].item())
                latch_duration_steps_total_ep = float(ep_crit_latch_duration_steps_sum[idx].item())
                if bool(ep_crit_prev_latched[idx].item()):
                    latch_duration_steps_total_ep += float(ep_crit_latch_open_run_steps[idx].item())
                if latch_event_count_ep > 0.0:
                    latch_mean_dur_steps_ep = float(latch_duration_steps_total_ep / latch_event_count_ep)
                    latch_mean_dur_s_ep = float(latch_mean_dur_steps_ep * dt)
                else:
                    latch_mean_dur_steps_ep = float("nan")
                    latch_mean_dur_s_ep = float("nan")
                first_latch_step_val = int(ep_crit_first_latch_step[idx].item())
                first_unlatch_step_val = int(ep_crit_first_unlatch_step[idx].item())
                first_unlatch_after_zero_step_val = int(ep_crit_first_unlatch_after_zero_step[idx].item())
                time_to_first_latch_s = (
                    float(first_latch_step_val * dt)
                    if first_latch_step_val >= 0
                    else float("nan")
                )
                time_to_unlatch_s = (
                    float(first_unlatch_step_val * dt)
                    if first_unlatch_step_val >= 0
                    else float("nan")
                )
                if first_latch_step_val >= 0 and first_unlatch_step_val >= first_latch_step_val:
                    latch_recovery_delay_s = float((first_unlatch_step_val - first_latch_step_val) * dt)
                else:
                    latch_recovery_delay_s = float("nan")
                unlatch_success_ep = (
                    1.0 if (first_latch_step_val >= 0 and first_unlatch_step_val >= 0) else 0.0
                )
                if forced_walk_then_zero_enabled:
                    unlatch_after_zero_success_ep = (
                        1.0 if (first_latch_step_val >= 0 and first_unlatch_after_zero_step_val >= 0) else 0.0
                    )
                    if first_unlatch_after_zero_step_val >= 0:
                        time_to_unlatch_after_zero_s = float(
                            max(0.0, first_unlatch_after_zero_step_val * dt - forced_walk_then_zero_walk_s)
                        )
                    else:
                        time_to_unlatch_after_zero_s = float("nan")
                else:
                    unlatch_after_zero_success_ep = float("nan")
                    time_to_unlatch_after_zero_s = float("nan")
                crit_action_delta_latched_mean_ep = float(ep_crit_action_delta_latched_sum[idx].item() / latched_steps)
                crit_action_transition_delta_mean_ep = float(
                    ep_crit_action_transition_delta_sum[idx].item() / ep_len_safe
                )
                crit_cmd_delta_latched_mean_ep = float(ep_crit_cmd_delta_latched_sum[idx].item() / latched_steps)
                crit_cmd_delta_active_ratio_ep = float(ep_crit_cmd_delta_active_steps[idx].item() / ep_len_safe)
                crit_post_unlatch_ramp_active_ratio_ep = float(
                    ep_crit_post_unlatch_ramp_active_steps[idx].item() / ep_len_safe
                )
                if latched_steps_raw > 0.0:
                    crit_cmd_delta_active_latched_ratio_ep = float(
                        ep_crit_cmd_delta_active_latched_steps[idx].item() / latched_steps_raw
                    )
                    crit_latched_low_cmd_delta_ratio_ep = float(
                        1.0 - crit_cmd_delta_active_latched_ratio_ep
                    )
                else:
                    crit_cmd_delta_active_latched_ratio_ep = float("nan")
                    crit_latched_low_cmd_delta_ratio_ep = float("nan")
                crit_sat_ratio_fallback_ratio_ep = float(
                    ep_crit_sat_ratio_fallback_steps[idx].item() / ep_len_safe
                )

                sat_ratio_hist_all = ep_crit_sat_ratio_step_history[idx]
                sat_ratio_hist_latched = ep_crit_sat_ratio_step_history_latched[idx]
                if len(sat_ratio_hist_all) > 0:
                    sat_arr_all = np.array(sat_ratio_hist_all, dtype=np.float64)
                    sat_ratio_step_mean_ep = float(np.mean(sat_arr_all))
                    sat_ratio_step_p95_ep = float(np.percentile(sat_arr_all, 95))
                    sat_ratio_step_max_ep = float(np.max(sat_arr_all))
                else:
                    sat_ratio_step_mean_ep = float("nan")
                    sat_ratio_step_p95_ep = float("nan")
                    sat_ratio_step_max_ep = float("nan")
                if len(sat_ratio_hist_latched) > 0:
                    sat_arr_latched = np.array(sat_ratio_hist_latched, dtype=np.float64)
                    sat_ratio_latched_step_mean_ep = float(np.mean(sat_arr_latched))
                    sat_ratio_latched_step_p95_ep = float(np.percentile(sat_arr_latched, 95))
                    sat_ratio_latched_step_max_ep = float(np.max(sat_arr_latched))
                else:
                    sat_ratio_latched_step_mean_ep = float("nan")
                    sat_ratio_latched_step_p95_ep = float("nan")
                    sat_ratio_latched_step_max_ep = float("nan")
                post_unlatch_sat_hist = ep_crit_post_unlatch_sat_history[idx]
                if len(post_unlatch_sat_hist) > 0:
                    post_arr = np.array(post_unlatch_sat_hist, dtype=np.float64)
                    post_unlatch_peak_sat_ep = float(np.max(post_arr))
                    post_unlatch_peak_sat_p95_ep = float(np.percentile(post_arr, 95))
                else:
                    post_unlatch_peak_sat_ep = float("nan")
                    post_unlatch_peak_sat_p95_ep = float("nan")

                # Fixed-window post-unlatch saturation metric (fair comparison across policies).
                # Default: ignore 0.5s transient, then evaluate next 2.0s only.
                fixed_unlatch_ref_step = (
                    first_unlatch_after_zero_step_val
                    if (forced_walk_then_zero_enabled and first_unlatch_after_zero_step_val >= 0)
                    else first_unlatch_step_val
                )
                post_unlatch_fixedwin_used_steps_ep = 0.0
                post_unlatch_fixedwin_valid_ep = 0.0
                post_unlatch_fixedwin_peak_sat_ep = float("nan")
                post_unlatch_fixedwin_peak_sat_p95_ep = float("nan")
                sat_hist_all = ep_max_saturation_step_history[idx]
                if fixed_unlatch_ref_step >= 0 and len(sat_hist_all) > 0:
                    start_step_1b = fixed_unlatch_ref_step + int(post_unlatch_ignore_steps)
                    start_idx = max(start_step_1b - 1, 0)
                    end_idx = start_idx + int(post_unlatch_window_steps)
                    if start_idx < len(sat_hist_all):
                        window_vals = sat_hist_all[start_idx:min(end_idx, len(sat_hist_all))]
                        used_steps = int(len(window_vals))
                        post_unlatch_fixedwin_used_steps_ep = float(used_steps)
                        post_unlatch_fixedwin_valid_ep = 1.0 if used_steps >= int(post_unlatch_window_steps) else 0.0
                        if used_steps > 0:
                            window_arr = np.array(window_vals, dtype=np.float64)
                            post_unlatch_fixedwin_peak_sat_ep = float(np.max(window_arr))
                            post_unlatch_fixedwin_peak_sat_p95_ep = float(np.percentile(window_arr, 95))

                episode_metrics["crit_sat_any_over_thr_ratio_ep"].append(crit_sat_any_ratio_ep)
                episode_metrics["crit_sat_window_ratio_mean_ep"].append(crit_sat_window_ratio_ep)
                episode_metrics["crit_latched_step_ratio_ep"].append(crit_latched_ratio_ep)
                episode_metrics["crit_latch_count_ep"].append(latch_event_count_ep)
                episode_metrics["crit_latch_mean_dur_steps_ep"].append(latch_mean_dur_steps_ep)
                episode_metrics["crit_latch_mean_dur_s_ep"].append(latch_mean_dur_s_ep)
                episode_metrics["crit_time_to_first_latch_s"].append(time_to_first_latch_s)
                episode_metrics["crit_time_to_unlatch_s"].append(time_to_unlatch_s)
                episode_metrics["crit_latch_recovery_delay_s"].append(latch_recovery_delay_s)
                episode_metrics["crit_unlatch_success_ep"].append(float(unlatch_success_ep))
                episode_metrics["crit_unlatch_after_zero_success_ep"].append(float(unlatch_after_zero_success_ep))
                episode_metrics["crit_time_to_unlatch_after_zero_s"].append(float(time_to_unlatch_after_zero_s))
                episode_metrics["crit_cmd_delta_active_ratio_ep"].append(crit_cmd_delta_active_ratio_ep)
                episode_metrics["crit_cmd_delta_active_latched_ratio_ep"].append(crit_cmd_delta_active_latched_ratio_ep)
                episode_metrics["crit_latched_low_cmd_delta_ratio_ep"].append(crit_latched_low_cmd_delta_ratio_ep)
                episode_metrics["crit_sat_ratio_fallback_step_ratio_ep"].append(crit_sat_ratio_fallback_ratio_ep)
                for mode_i, mode_name in enumerate(governor_mode_names):
                    mode_ratio_ep = float(ep_crit_governor_mode_steps[idx, mode_i].item() / ep_len_safe)
                    episode_metrics[f"crit_governor_mode_{mode_name}_ratio_ep"].append(mode_ratio_ep)
                episode_metrics["crit_sat_ratio_step_mean_ep"].append(sat_ratio_step_mean_ep)
                episode_metrics["crit_sat_ratio_step_p95_ep"].append(sat_ratio_step_p95_ep)
                episode_metrics["crit_sat_ratio_step_max_ep"].append(sat_ratio_step_max_ep)
                episode_metrics["crit_sat_ratio_latched_step_mean_ep"].append(sat_ratio_latched_step_mean_ep)
                episode_metrics["crit_sat_ratio_latched_step_p95_ep"].append(sat_ratio_latched_step_p95_ep)
                episode_metrics["crit_sat_ratio_latched_step_max_ep"].append(sat_ratio_latched_step_max_ep)
                episode_metrics["crit_post_unlatch_peak_sat_ep"].append(post_unlatch_peak_sat_ep)
                episode_metrics["crit_post_unlatch_peak_sat_p95_ep"].append(post_unlatch_peak_sat_p95_ep)
                episode_metrics["crit_post_unlatch_fixedwin_used_steps_ep"].append(
                    float(post_unlatch_fixedwin_used_steps_ep)
                )
                episode_metrics["crit_post_unlatch_fixedwin_valid_ep"].append(
                    float(post_unlatch_fixedwin_valid_ep)
                )
                episode_metrics["crit_post_unlatch_fixedwin_peak_sat_ep"].append(
                    float(post_unlatch_fixedwin_peak_sat_ep)
                )
                episode_metrics["crit_post_unlatch_fixedwin_peak_sat_p95_ep"].append(
                    float(post_unlatch_fixedwin_peak_sat_p95_ep)
                )
                episode_metrics["crit_action_delta_latched_norm_mean_ep"].append(crit_action_delta_latched_mean_ep)
                episode_metrics["crit_action_delta_latched_norm_max_ep"].append(
                    float(ep_crit_action_delta_latched_max[idx].item())
                )
                episode_metrics["crit_action_transition_delta_norm_mean_ep"].append(
                    crit_action_transition_delta_mean_ep
                )
                episode_metrics["crit_action_transition_delta_norm_max_ep"].append(
                    float(ep_crit_action_transition_delta_max[idx].item())
                )
                episode_metrics["crit_post_unlatch_action_ramp_active_ratio_ep"].append(
                    crit_post_unlatch_ramp_active_ratio_ep
                )
                episode_metrics["crit_cmd_delta_latched_norm_mean_ep"].append(crit_cmd_delta_latched_mean_ep)
                episode_metrics["crit_cmd_delta_latched_norm_max_ep"].append(
                    float(ep_crit_cmd_delta_latched_max[idx].item())
                )
                for j in range(num_joints):
                    sat_ratio_095 = float(ep_sat_joint_over_0p95_steps[idx, j].item() / ep_len_safe)
                    sat_ratio_100 = float(ep_sat_joint_over_1p00_steps[idx, j].item() / ep_len_safe)
                    episode_metrics[f"sat_joint{j}_over_0p95_ratio"].append(sat_ratio_095)
                    episode_metrics[f"sat_joint{j}_over_1p00_ratio"].append(sat_ratio_100)

                # Governor diagnostics from per-env terminal snapshot (when available).
                term_crit_sat_ratio = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/sat_any_over_thr_ratio"
                )
                term_crit_is_latched = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/is_latched"
                )
                term_crit_sat_valid_steps = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/sat_ratio_valid_steps"
                )
                term_crit_governor_enabled = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/governor_enabled"
                )
                term_crit_action_delta_latched = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/action_delta_latched_norm_step"
                )
                term_crit_cmd_delta_latched = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/cmd_delta_latched_norm_step"
                )
                term_crit_governor_mode = _terminal_scalar(
                    terminal_metrics, terminal_lookup, idx, "crit/governor_mode_step"
                )
                term_crit_sat_ratio_val = float(term_crit_sat_ratio) if term_crit_sat_ratio is not None else float("nan")
                episode_metrics["crit_sat_any_over_thr_ratio"].append(
                    term_crit_sat_ratio_val
                )
                episode_metrics["crit_sat_any_over_thr_ratio_terminal"].append(
                    term_crit_sat_ratio_val
                )
                episode_metrics["crit_is_latched"].append(
                    float(term_crit_is_latched) if term_crit_is_latched is not None else float("nan")
                )
                episode_metrics["crit_sat_ratio_valid_steps"].append(
                    float(term_crit_sat_valid_steps) if term_crit_sat_valid_steps is not None else float("nan")
                )
                episode_metrics["crit_governor_enabled"].append(
                    float(term_crit_governor_enabled) if term_crit_governor_enabled is not None else float("nan")
                )
                episode_metrics["crit_action_delta_latched_norm_step"].append(
                    float(term_crit_action_delta_latched)
                    if term_crit_action_delta_latched is not None
                    else float("nan")
                )
                episode_metrics["crit_cmd_delta_latched_norm_step"].append(
                    float(term_crit_cmd_delta_latched)
                    if term_crit_cmd_delta_latched is not None
                    else float("nan")
                )
                episode_metrics["crit_governor_mode_step"].append(
                    float(term_crit_governor_mode)
                    if term_crit_governor_mode is not None
                    else float("nan")
                )

                # Gait/contact evidence from terminal snapshot when available.
                gait_step_count_total = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "gait/step_count_total")
                gait_quad_support_ratio = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "gait/quad_support_ratio")
                gait_slip_per_progress = _terminal_scalar(terminal_metrics, terminal_lookup, idx, "gait/slip_per_progress")

                gait_available = 0.0
                if (
                    gait_step_count_total is not None
                    and gait_quad_support_ratio is not None
                    and gait_slip_per_progress is not None
                ):
                    gait_available = 1.0
                elif num_feet > 0 and ep_gait_touchdown_count is not None and ep_gait_valid_steps[idx].item() > 0:
                    valid_steps = max(float(ep_gait_valid_steps[idx].item()), 1.0)
                    step_total_val = float(torch.sum(ep_gait_touchdown_count[idx]).item())
                    quad_ratio_val = float(ep_gait_quad_support_steps[idx].item() / valid_steps)
                    slip_total_val = float(torch.sum(ep_gait_slip_distance[idx]).item())
                    prog_val = float(ep_progress_forward[idx].item())
                    slip_per_prog_val = slip_total_val / max(prog_val, 0.05)
                    gait_step_count_total = step_total_val
                    gait_quad_support_ratio = quad_ratio_val
                    gait_slip_per_progress = slip_per_prog_val
                    gait_available = 1.0

                episode_metrics["gait_available"].append(float(gait_available))
                if gait_available > 0.5:
                    episode_metrics["gait_step_count_total"].append(float(gait_step_count_total))
                    episode_metrics["gait_quad_support_ratio"].append(float(gait_quad_support_ratio))
                    episode_metrics["gait_slip_per_progress"].append(float(gait_slip_per_progress))
                else:
                    episode_metrics["gait_step_count_total"].append(float("nan"))
                    episode_metrics["gait_quad_support_ratio"].append(float("nan"))
                    episode_metrics["gait_slip_per_progress"].append(float("nan"))

                # Walk / Stand / Shuffle episode classification.
                mc = float(mean_cmd_speed)
                ma = float(mean_actual_speed)
                pr = float(ep_progress_forward[idx].item())
                nz = float((ep_nonzero_cmd[idx] / max(ep_len, 1)).item())
                sr = float((ep_stand_cmd[idx] / max(ep_len, 1)).item())
                mean_cmd_ang_abs = float((ep_cmd_ang_sum[idx] / max(ep_len, 1)).item())
                mean_prog_speed_fwd = float(mean_progress_speed_forward)
                progress_ratio = mean_prog_speed_fwd / max(mc, 1e-6)

                is_stand = (
                    (mc <= stand_cmd_max)
                    and (ma <= stand_actual_max)
                    and (sr >= stand_ratio_min)
                    and (mean_cmd_ang_abs <= stand_cmd_ang_abs_max)
                )
                is_walk = (
                    (mc >= walk_cmd_min)
                    and (nz >= walk_nz_ratio_min)
                    and (pr >= walk_progress_min)
                    and (mean_prog_speed_fwd >= walk_progress_speed_min)
                    and (progress_ratio >= walk_progress_ratio_min)
                )

                # Shuffle covers commanded-walk but poor forward progress/slip-like behavior.
                if is_stand:
                    is_walk_f, is_stand_f, is_shuffle_f = 0.0, 1.0, 0.0
                elif is_walk:
                    is_walk_f, is_stand_f, is_shuffle_f = 1.0, 0.0, 0.0
                else:
                    is_walk_f, is_stand_f, is_shuffle_f = 0.0, 0.0, 1.0

                episode_metrics["is_walk"].append(is_walk_f)
                episode_metrics["is_stand"].append(is_stand_f)
                episode_metrics["is_shuffle"].append(is_shuffle_f)

                if len(reward_breakdown.term_names) > 0:
                    # Add exact terminal-step reward terms cached before reset.
                    for term_name in reward_breakdown.term_names:
                        terminal_term = _terminal_scalar(
                            terminal_metrics,
                            terminal_lookup,
                            idx,
                            f"reward_term/{term_name}",
                        )
                        if terminal_term is not None:
                            ep_term_signed[term_name][idx] += float(terminal_term)
                            ep_term_abs[term_name][idx] += abs(float(terminal_term))

                    abs_total = 0.0
                    for term_name in reward_breakdown.term_names:
                        abs_total += float(ep_term_abs[term_name][idx].item())
                    denom = max(abs_total, 1e-8)

                    for term_name in reward_breakdown.term_names:
                        signed_sum = float(ep_term_signed[term_name][idx].item())
                        abs_sum = float(ep_term_abs[term_name][idx].item())
                        episode_metrics[f"reward_{term_name}_signed_mean"].append(signed_sum / max(ep_len, 1))
                        episode_metrics[f"reward_{term_name}_abs_mean"].append(abs_sum / max(ep_len, 1))
                        episode_metrics[f"reward_share_{term_name}"].append(abs_sum / denom)

                completed_episodes += 1

                # Reset per-env buffers
                ep_step_counter[idx] = 0
                ep_total_reward[idx] = 0.0
                ep_tracking_errors[idx] = []
                ep_ang_errors[idx] = []
                ep_power_history[idx] = []
                ep_cmd_speed_sum[idx] = 0.0
                ep_cmd_ang_sum[idx] = 0.0
                ep_actual_speed_sum[idx] = 0.0
                ep_path_length[idx] = 0.0
                ep_progress_signed[idx] = 0.0
                ep_progress_forward[idx] = 0.0
                ep_nonzero_cmd[idx] = 0.0
                ep_stand_cmd[idx] = 0.0
                ep_energy_cum_j[idx] = 0.0
                ep_peak_temp_c[idx] = -float("inf")
                ep_peak_saturation[idx] = 0.0
                ep_safe_stop_consec_steps[idx] = 0
                ep_safe_stop_reached[idx] = False
                ep_time_to_safe_stop_s[idx] = float("nan")
                ep_energy_to_safe_stop_j[idx] = float("nan")
                ep_sat_joint_over_0p95_steps[idx] = 0.0
                ep_sat_joint_over_1p00_steps[idx] = 0.0
                ep_sat_any_over_1p00_steps[idx] = 0.0
                ep_crit_sat_any_over_thr_steps[idx] = 0.0
                ep_crit_sat_window_ratio_sum[idx] = 0.0
                ep_crit_latched_steps[idx] = 0.0
                ep_crit_action_delta_latched_sum[idx] = 0.0
                ep_crit_action_delta_latched_max[idx] = 0.0
                ep_crit_action_transition_delta_sum[idx] = 0.0
                ep_crit_action_transition_delta_max[idx] = 0.0
                ep_crit_cmd_delta_latched_sum[idx] = 0.0
                ep_crit_cmd_delta_latched_max[idx] = 0.0
                ep_crit_post_unlatch_ramp_active_steps[idx] = 0.0
                ep_crit_cmd_delta_active_steps[idx] = 0.0
                ep_crit_cmd_delta_active_latched_steps[idx] = 0.0
                ep_crit_sat_ratio_fallback_steps[idx] = 0.0
                ep_crit_governor_mode_steps[idx] = 0.0
                ep_crit_latch_event_count[idx] = 0.0
                ep_crit_latch_duration_steps_sum[idx] = 0.0
                ep_crit_latch_open_run_steps[idx] = 0.0
                ep_crit_prev_latched[idx] = False
                ep_crit_first_latch_step[idx] = -1
                ep_crit_first_unlatch_step[idx] = -1
                ep_crit_first_unlatch_after_zero_step[idx] = -1
                ep_crit_sat_ratio_step_history[idx] = []
                ep_crit_sat_ratio_step_history_latched[idx] = []
                ep_crit_post_unlatch_sat_history[idx] = []
                ep_max_saturation_step_history[idx] = []
                ep_gait_valid_steps[idx] = 0.0
                ep_gait_quad_support_steps[idx] = 0.0
                if num_feet > 0 and ep_gait_contact_steps is not None and ep_gait_touchdown_count is not None and ep_gait_slip_distance is not None:
                    ep_gait_contact_steps[idx] = 0.0
                    ep_gait_touchdown_count[idx] = 0.0
                    ep_gait_slip_distance[idx] = 0.0
                    if prev_contact is not None:
                        prev_contact[idx] = False
                for term_name in reward_breakdown.term_names:
                    ep_term_signed[term_name][idx] = 0.0
                    ep_term_abs[term_name][idx] = 0.0

                if completed_episodes >= num_target_episodes:
                    break

            # Re-inject scenario for reset envs
            if completed_episodes < num_target_episodes:
                done_ids_device = done_envs.to(env.unwrapped.device)
                apply_scenario(env, scenario_name, done_ids_device)
                # Avoid one-step stale observation for environments just reinjected.
                obs = env.get_observations()
                _capture_cmd_targets(done_ids_device)
                # Re-initialize prev_contact on reset envs to avoid artificial touchdown count
                # on the first step after reset.
                if num_feet > 0 and prev_contact is not None:
                    gait_after_reset = collect_gait_contact_metrics(
                        env,
                        foot_body_ids=foot_body_ids,
                        contact_force_threshold=float(contact_force_threshold),
                    )
                    done_ids_cpu = done_envs.detach().cpu().to(torch.long)
                    if bool(gait_after_reset.get("available", False)):
                        prev_contact[done_ids_cpu] = gait_after_reset["contact"][done_ids_cpu]
                    else:
                        prev_contact[done_ids_cpu] = False

        if completed_episodes > 0 and (completed_episodes // 20) > (last_reported_episodes // 20):
            surv = np.mean(episode_metrics["survived"][-20:]) if len(episode_metrics["survived"]) >= 20 else np.mean(episode_metrics["survived"])
            trk = np.mean(episode_metrics["mean_tracking_error_xy"][-20:]) if len(episode_metrics["mean_tracking_error_xy"]) >= 20 else np.mean(episode_metrics["mean_tracking_error_xy"])
            print(f"  [{completed_episodes}/{num_target_episodes}] "
                  f"Survival={surv:.2%} | TrackErr={trk:.4f}")
            last_reported_episodes = completed_episodes

    if ts_enabled and (not ts_dumped) and (len(ts_rows) > 0):
        _flush_timeseries(reason="loop_end")

    # Compute summary statistics
    summary = {}
    nan_tolerant_keys = {
        "gait_step_count_total",
        "gait_quad_support_ratio",
        "gait_slip_per_progress",
        "time_to_safe_stop_s",
        "time_to_fall_s",
        "energy_to_safe_stop_j",
        "thermal_margin_to_threshold_c",
        "crit_cmd_delta_active_latched_ratio_ep",
        "crit_latched_low_cmd_delta_ratio_ep",
        "crit_sat_ratio_fallback_step_ratio_ep",
        "crit_latch_mean_dur_steps_ep",
        "crit_latch_mean_dur_s_ep",
        "crit_time_to_first_latch_s",
        "crit_time_to_unlatch_s",
        "crit_time_to_unlatch_after_zero_s",
        "crit_latch_recovery_delay_s",
        "crit_unlatch_after_zero_success_ep",
        "crit_sat_ratio_latched_step_mean_ep",
        "crit_sat_ratio_latched_step_p95_ep",
        "crit_sat_ratio_latched_step_max_ep",
        "crit_post_unlatch_peak_sat_ep",
        "crit_post_unlatch_peak_sat_p95_ep",
        "crit_post_unlatch_fixedwin_peak_sat_ep",
        "crit_post_unlatch_fixedwin_peak_sat_p95_ep",
        "crit_governor_mode_step",
    }
    for key, values in episode_metrics.items():
        if len(values) == 0:
            summary[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "count": 0,
            }
            continue
        arr = np.array(values, dtype=np.float64)
        if key in nan_tolerant_keys:
            valid_mask = ~np.isnan(arr)
            valid_count = int(np.sum(valid_mask))
            if valid_count > 0:
                summary[key] = {
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr)),
                    "min": float(np.nanmin(arr)),
                    "max": float(np.nanmax(arr)),
                    "median": float(np.nanmedian(arr)),
                    "p95": float(np.nanpercentile(arr, 95)),
                    "count": len(arr),
                    "valid_count": valid_count,
                }
            else:
                summary[key] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                    "median": float("nan"),
                    "p95": float("nan"),
                    "count": len(arr),
                    "valid_count": 0,
                }
        else:
            summary[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "count": len(arr),
            }

    if observed_cmd_speed_count > 0:
        summary["_debug_observed_cmd_speed_xy_mean"] = float(observed_cmd_speed_sum / observed_cmd_speed_count)
        summary["_debug_observed_cmd_speed_xy_max"] = float(observed_cmd_speed_max)
    summary["_debug_cmd_nonzero_threshold"] = float(nonzero_cmd_threshold)
    summary["_debug_cmd_stand_threshold"] = float(stand_cmd_threshold)
    summary["_debug_eval_contact_force_threshold"] = float(contact_force_threshold)
    summary["_debug_eval_foot_body_resolve_status"] = str(foot_resolve_status)
    summary["_debug_eval_foot_body_names"] = list(resolved_foot_names)
    summary["_debug_safe_stop_thresholds"] = {
        "safe_stop_lin_vel_max": float(safe_stop_lin_vel_max),
        "safe_stop_ang_vel_max": float(safe_stop_ang_vel_max),
        "safe_stop_hold_s": float(safe_stop_hold_s),
        "safe_stop_hold_steps": int(safe_stop_hold_steps),
        "safe_stop_require_pose": bool(safe_stop_require_pose),
        "safe_stop_max_tilt_rad": float(safe_stop_max_tilt_rad),
        "safe_stop_min_height_m": float(safe_stop_min_height_m),
        "oracle_safe_stop_steps": int(oracle_safe_stop_steps),
        "crit_cmd_delta_active_eps": float(crit_cmd_delta_active_eps),
    }
    summary["_debug_crit_sat_ratio_semantics"] = {
        "crit_sat_any_over_thr_ratio": "terminal_snapshot_window_ratio (legacy)",
        "crit_sat_any_over_thr_ratio_terminal": "terminal_snapshot_window_ratio",
        "crit_sat_any_over_thr_ratio_ep": "episode_mean_of_stepwise_any_over_thr",
        "crit_sat_window_ratio_mean_ep": "episode_mean_of_stepwise_window_ratio",
        "crit_sat_window_ratio_mean_ep_fallback": "if ratio signal is zero while sat_any=1, fallback uses sat_any for that step",
        "crit_sat_ratio_fallback_step_ratio_ep": "episode_step_ratio[fallback applied]",
        "crit_latched_step_ratio_ep": "episode_mean_of_stepwise_latched_mask",
        "crit_cmd_delta_active_ratio_ep": "episode_step_ratio[||cmd_eff-cmd_raw|| > crit_cmd_delta_active_eps]",
        "crit_latched_low_cmd_delta_ratio_ep": "episode_latched_step_ratio[||cmd_eff-cmd_raw|| <= crit_cmd_delta_active_eps]",
        "crit_latch_count_ep": "episode_count[latched transitions false->true]",
        "crit_latch_mean_dur_steps_ep": "episode_mean_duration_steps_per_latch_event",
        "crit_latch_mean_dur_s_ep": "episode_mean_duration_seconds_per_latch_event",
        "crit_time_to_first_latch_s": "episode_time_seconds_to_first_false->true_latch (nan if no latch)",
        "crit_time_to_unlatch_s": "episode_time_seconds_to_first_true->false_unlatch_after_latch (nan if no unlatch)",
        "crit_time_to_unlatch_after_zero_s": (
            "forced_walk_then_zero only: max(0, first_unlatch_after_zero_time - walk_s)"
        ),
        "crit_latch_recovery_delay_s": "crit_time_to_unlatch_s - crit_time_to_first_latch_s (nan if unavailable)",
        "crit_unlatch_success_ep": "episode_indicator[first latch occurred and later unlatch occurred]",
        "crit_unlatch_after_zero_success_ep": (
            "forced_walk_then_zero only: episode_indicator[first latch occurred and later unlatch occurred after zero-phase]"
        ),
        "crit_post_unlatch_peak_sat_ep": "episode_max(max_saturation) over steps after first unlatch",
        "crit_post_unlatch_peak_sat_p95_ep": "episode_p95(max_saturation) over steps after first unlatch",
        "crit_post_unlatch_fixedwin_used_steps_ep": (
            "post-unlatch fixed-window sampled steps count (ignore 0.5s, then window 2.0s)"
        ),
        "crit_post_unlatch_fixedwin_valid_ep": (
            "post-unlatch fixed-window validity indicator (1 if full window available else 0)"
        ),
        "crit_post_unlatch_fixedwin_peak_sat_ep": (
            "episode_max(max_saturation) over fixed window after unlatch (0.5s ignore + 2.0s window)"
        ),
        "crit_post_unlatch_fixedwin_peak_sat_p95_ep": (
            "episode_p95(max_saturation) over fixed window after unlatch (0.5s ignore + 2.0s window)"
        ),
        "crit_post_unlatch_action_ramp_active_ratio_ep": (
            "episode_step_ratio[post-unlatch action-ramp transition active]"
        ),
        "crit_action_transition_delta_norm_mean_ep": (
            "episode_mean(||action_after_transition - action_before_transition||)"
        ),
        "crit_action_transition_delta_norm_max_ep": (
            "episode_max(||action_after_transition - action_before_transition||)"
        ),
        "crit_governor_mode_step": "terminal enum {0:none,1:v_cap,2:stand,3:stop_latch,4:reserved_other}",
    }
    summary["_debug_forced_walk_ramp"] = {
        "enabled": bool(forced_walk_ramp_enabled),
        "ramp_s": float(forced_walk_ramp_s),
        "cmd_profile_effective": str(cmd_profile_effective),
    }
    summary["_debug_forced_walk_then_zero"] = {
        "enabled": bool(forced_walk_then_zero_enabled),
        "walk_s": float(forced_walk_then_zero_walk_s),
        "post_unlatch_ignore_s": float(post_unlatch_ignore_s),
        "post_unlatch_window_s": float(post_unlatch_window_s),
        "post_unlatch_ignore_steps": int(post_unlatch_ignore_steps),
        "post_unlatch_window_steps": int(post_unlatch_window_steps),
        "cmd_profile_effective": str(cmd_profile_effective),
    }
    summary["_debug_episode_classification_thresholds"] = {
        "stand_cmd_max": float(stand_cmd_max),
        "stand_actual_max": float(stand_actual_max),
        "stand_ratio_min": float(stand_ratio_min),
        "stand_cmd_ang_abs_max": float(stand_cmd_ang_abs_max),
        "walk_cmd_min": float(walk_cmd_min),
        "walk_nz_ratio_min": float(walk_nz_ratio_min),
        "walk_progress_min": float(walk_progress_min),
        "walk_progress_speed_min": float(walk_progress_speed_min),
        "walk_progress_ratio_min": float(walk_progress_ratio_min),
    }
    summary["_debug_timeseries_dump"] = {
        "enabled": bool(ts_enabled),
        "scenario": str(dump_timeseries_scenario),
        "env_id": int(ts_env),
        "max_steps": int(ts_max_steps),
        "rows_collected": int(len(ts_rows)),
        "csv_path": str(ts_path) if ts_path is not None else None,
        "meta_path": str(ts_meta_path) if ts_meta_path is not None else None,
    }

    # Derived episode-category summaries.
    survived = np.array(episode_metrics.get("survived", []), dtype=np.float32)
    is_walk = np.array(episode_metrics.get("is_walk", []), dtype=np.float32)
    is_stand = np.array(episode_metrics.get("is_stand", []), dtype=np.float32)
    is_shuffle = np.array(episode_metrics.get("is_shuffle", []), dtype=np.float32)
    total_eps = float(len(survived))
    if total_eps > 0:
        walk_count = float(np.sum(is_walk))
        stand_count = float(np.sum(is_stand))
        shuffle_count = float(np.sum(is_shuffle))
        walk_ratio = walk_count / total_eps
        stand_ratio = stand_count / total_eps
        shuffle_ratio = shuffle_count / total_eps
        summary["walk_ratio"] = {
            "mean": float(walk_ratio),
            "std": 0.0,
            "min": float(walk_ratio),
            "max": float(walk_ratio),
            "median": float(walk_ratio),
            "p95": float(walk_ratio),
            "count": int(total_eps),
        }
        summary["stand_ratio"] = {
            "mean": float(stand_ratio),
            "std": 0.0,
            "min": float(stand_ratio),
            "max": float(stand_ratio),
            "median": float(stand_ratio),
            "p95": float(stand_ratio),
            "count": int(total_eps),
        }
        summary["shuffle_ratio"] = {
            "mean": float(shuffle_ratio),
            "std": 0.0,
            "min": float(shuffle_ratio),
            "max": float(shuffle_ratio),
            "median": float(shuffle_ratio),
            "p95": float(shuffle_ratio),
            "count": int(total_eps),
        }
        if walk_count > 0.0:
            survival_walk_only = float(np.sum(survived * is_walk) / walk_count)
        else:
            survival_walk_only = float("nan")
        summary["survival_walk_only"] = {
            "mean": float(survival_walk_only),
            "std": 0.0,
            "min": float(survival_walk_only),
            "max": float(survival_walk_only),
            "median": float(survival_walk_only),
            "p95": float(survival_walk_only),
            "count": int(walk_count),
        }
    else:
        for k in ("walk_ratio", "stand_ratio", "shuffle_ratio", "survival_walk_only"):
            summary[k] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "count": 0,
            }

    # Ensure requested paper-check keys are always present.
    required_keys = (
        "mean_cmd_speed_xy",
        "progress_distance_along_cmd",
        "path_length_xy",
        "gait_available",
        "gait_step_count_total",
        "gait_quad_support_ratio",
        "gait_slip_per_progress",
        "safe_stop_success",
        "safe_stop_success_rate",
        "time_to_safe_stop_s",
        "time_to_fall_s",
        "time_to_episode_end_s",
        "energy_to_safe_stop_j",
        "energy_to_episode_end_j",
        "peak_temp_episode_c",
        "thermal_margin_to_threshold_c",
        "peak_saturation_episode",
        "crit_sat_any_over_thr_ratio_ep",
        "crit_sat_window_ratio_mean_ep",
        "crit_latched_step_ratio_ep",
        "crit_latch_count_ep",
        "crit_latch_mean_dur_steps_ep",
        "crit_latch_mean_dur_s_ep",
        "crit_time_to_first_latch_s",
        "crit_time_to_unlatch_s",
        "crit_time_to_unlatch_after_zero_s",
        "crit_latch_recovery_delay_s",
        "crit_unlatch_success_ep",
        "crit_unlatch_after_zero_success_ep",
        "crit_cmd_delta_active_ratio_ep",
        "crit_cmd_delta_active_latched_ratio_ep",
        "crit_latched_low_cmd_delta_ratio_ep",
        "crit_sat_ratio_fallback_step_ratio_ep",
        "crit_governor_mode_none_ratio_ep",
        "crit_governor_mode_v_cap_ratio_ep",
        "crit_governor_mode_stand_ratio_ep",
        "crit_governor_mode_stop_latch_ratio_ep",
        "crit_governor_mode_other_ratio_ep",
        "crit_sat_ratio_step_mean_ep",
        "crit_sat_ratio_step_p95_ep",
        "crit_sat_ratio_step_max_ep",
        "crit_sat_ratio_latched_step_mean_ep",
        "crit_sat_ratio_latched_step_p95_ep",
        "crit_sat_ratio_latched_step_max_ep",
        "crit_post_unlatch_peak_sat_ep",
        "crit_post_unlatch_peak_sat_p95_ep",
        "crit_post_unlatch_fixedwin_used_steps_ep",
        "crit_post_unlatch_fixedwin_valid_ep",
        "crit_post_unlatch_fixedwin_peak_sat_ep",
        "crit_post_unlatch_fixedwin_peak_sat_p95_ep",
        "crit_post_unlatch_action_ramp_active_ratio_ep",
        "crit_action_transition_delta_norm_mean_ep",
        "crit_action_transition_delta_norm_max_ep",
        "crit_sat_any_over_thr_ratio_terminal",
    )
    for key in required_keys:
        if key not in summary:
            summary[key] = {
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
                "p95": float("nan"),
                "count": 0,
            }

    summary["reward_breakdown"] = reward_breakdown.meta()
    return summary


def _write_reward_breakdown_csv(all_results: dict, output_dir: str, timestamp: str) -> str | None:
    """Write scenario-wise reward contribution/share summary for quick tuning."""
    rows: list[dict[str, Any]] = []
    for scenario_name, summary in all_results.items():
        rb_meta = summary.get("reward_breakdown", {})
        dt_scaled = rb_meta.get("dt_scaled", None)
        recon_mae = rb_meta.get("last_reconstruction_mae", None)

        share_keys = [k for k in summary.keys() if k.startswith("reward_share_")]
        for share_key in sorted(share_keys):
            term_name = share_key[len("reward_share_") :]
            abs_key = f"reward_{term_name}_abs_mean"
            signed_key = f"reward_{term_name}_signed_mean"

            share_stats = summary.get(share_key, {})
            abs_stats = summary.get(abs_key, {})
            signed_stats = summary.get(signed_key, {})

            rows.append(
                {
                    "scenario": scenario_name,
                    "term": term_name,
                    "share_abs_mean": share_stats.get("mean", float("nan")),
                    "share_abs_std": share_stats.get("std", float("nan")),
                    "abs_contrib_mean": abs_stats.get("mean", float("nan")),
                    "abs_contrib_std": abs_stats.get("std", float("nan")),
                    "signed_contrib_mean": signed_stats.get("mean", float("nan")),
                    "signed_contrib_std": signed_stats.get("std", float("nan")),
                    "dt_scaled": dt_scaled,
                    "reconstruction_mae": recon_mae,
                }
            )

    if len(rows) == 0:
        return None

    csv_path = os.path.join(output_dir, f"reward_breakdown_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Main evaluation entry point."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    eval_protocol_mode = str(args_cli.eval_protocol_mode).strip().lower()
    if eval_protocol_mode == "combined":
        scenario_names_for_run = list(SCENARIOS.keys())
    elif eval_protocol_mode == "locomotion_only":
        scenario_names_for_run = ["fresh", "used", "aged"]
    elif eval_protocol_mode == "safety_only":
        scenario_names_for_run = ["critical"]
    else:
        raise ValueError(
            f"Invalid --eval_protocol_mode={eval_protocol_mode} "
            "(expected combined|locomotion_only|safety_only)."
        )

    requested_eval_cmd_profile = str(args_cli.eval_cmd_profile).strip().lower()
    effective_eval_cmd_profile = requested_eval_cmd_profile
    if eval_protocol_mode == "safety_only" and requested_eval_cmd_profile == "from_env":
        effective_eval_cmd_profile = str(args_cli.eval_safety_cmd_profile).strip().lower()
        print(
            "[INFO] safety_only protocol: overriding eval_cmd_profile "
            f"from '{requested_eval_cmd_profile}' to '{effective_eval_cmd_profile}'."
        )

    forced_walk_ang_z_min = float(args_cli.eval_forced_walk_ang_z_min)
    forced_walk_ang_z_max = float(args_cli.eval_forced_walk_ang_z_max)
    forced_walk_lin_x_min = float(args_cli.eval_forced_walk_lin_x_min)
    forced_walk_lin_x_max = float(args_cli.eval_forced_walk_lin_x_max)
    if forced_walk_lin_x_min > forced_walk_lin_x_max:
        raise ValueError(
            f"Invalid forced_walk lin_x range: min={forced_walk_lin_x_min} > max={forced_walk_lin_x_max}. "
            "Use --eval_forced_walk_lin_x_min <= --eval_forced_walk_lin_x_max."
        )
    if forced_walk_ang_z_min > forced_walk_ang_z_max:
        raise ValueError(
            f"Invalid forced_walk yaw range: min={forced_walk_ang_z_min} > max={forced_walk_ang_z_max}. "
            "Use --eval_forced_walk_ang_z_min <= --eval_forced_walk_ang_z_max."
        )
    forced_walk_then_zero_walk_s = float(args_cli.eval_forced_walk_then_zero_walk_s)
    if effective_eval_cmd_profile == "forced_walk_then_zero" and forced_walk_then_zero_walk_s <= 0.0:
        raise ValueError(
            "Invalid --eval_forced_walk_then_zero_walk_s (must be >0 when "
            "--eval_cmd_profile=forced_walk_then_zero)."
        )
    eval_foot_body_names = _parse_foot_body_names(args_cli.eval_foot_body_names)
    if eval_foot_body_names is not None and len(eval_foot_body_names) != 4:
        raise ValueError(
            f"--eval_foot_body_names expects exactly 4 names, got {len(eval_foot_body_names)}: "
            f"{eval_foot_body_names}"
        )
    contact_sensor_enable_path = _try_enable_contact_sensors(env_cfg)
    if contact_sensor_enable_path is not None:
        print(f"[INFO] Enabled contact sensors via cfg path: {contact_sensor_enable_path}")
    else:
        print("[WARN] Contact sensor flag path not found in cfg; gait metrics may fall back or be unavailable.")
    eval_cmd_profile_info = _apply_eval_command_profile(
        env_cfg,
        effective_eval_cmd_profile,
        forced_walk_lin_x=(forced_walk_lin_x_min, forced_walk_lin_x_max),
        forced_walk_ang_z=(forced_walk_ang_z_min, forced_walk_ang_z_max),
    )
    if effective_eval_cmd_profile == "forced_walk_then_zero":
        eval_cmd_profile_info["forced_walk_then_zero_walk_s"] = float(forced_walk_then_zero_walk_s)
    eval_velocity_curriculum_disabled = False
    if effective_eval_cmd_profile != "from_env":
        if hasattr(env_cfg, "velocity_cmd_curriculum_enable"):
            env_cfg.velocity_cmd_curriculum_enable = False
            eval_velocity_curriculum_disabled = True
    if bool(args_cli.eval_disable_velocity_curriculum) and hasattr(env_cfg, "velocity_cmd_curriculum_enable"):
        env_cfg.velocity_cmd_curriculum_enable = False
        eval_velocity_curriculum_disabled = True

    eval_fault_mode = str(args_cli.eval_fault_mode).strip().lower()
    eval_fault_motor_id = int(args_cli.eval_fault_motor_id)
    if eval_fault_mode == "single_motor_fixed" and eval_fault_motor_id < 0:
        raise ValueError(
            "--eval_fault_mode=single_motor_fixed requires --eval_fault_motor_id in [0..11]. "
            "Refusing silent fallback to random."
        )
    if eval_fault_motor_id >= 0 and not (0 <= eval_fault_motor_id < 12):
        raise ValueError(f"Invalid --eval_fault_motor_id={eval_fault_motor_id} (expected 0..11).")
    if hasattr(env_cfg, "motor_deg_fault_injection_mode") and str(args_cli.eval_fault_mode) != "from_env":
        env_cfg.motor_deg_fault_injection_mode = str(args_cli.eval_fault_mode)
    if hasattr(env_cfg, "motor_deg_fault_fixed_motor_id") and int(args_cli.eval_fault_motor_id) >= 0:
        env_cfg.motor_deg_fault_fixed_motor_id = int(args_cli.eval_fault_motor_id)
        if hasattr(env_cfg, "motor_deg_fault_injection_mode"):
            env_cfg.motor_deg_fault_injection_mode = "single_motor_fixed"

    paper_b_runtime_cfg = apply_paper_b_runtime_overrides(
        env_cfg,
        critical_governor_enable=args_cli.critical_governor_enable,
        paper_b_obs_ablation=args_cli.paper_b_obs_ablation,
        paper_b_sensor_preset=args_cli.paper_b_sensor_preset,
    )

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if hasattr(env, "unwrapped"):
        try:
            if hasattr(env.unwrapped, "configure_eval_gait_metrics"):
                env.unwrapped.configure_eval_gait_metrics(
                    enabled=True,
                    foot_body_names=eval_foot_body_names,
                    contact_force_threshold=float(args_cli.eval_contact_force_threshold),
                    nonzero_cmd_threshold=float(args_cli.eval_nonzero_cmd_threshold),
                    stand_cmd_threshold=float(args_cli.eval_stand_cmd_threshold),
                )
            else:
                # Backward-compat fallback for older envs without explicit config hook.
                env.unwrapped.enable_eval_gait_metrics = True
        except Exception as e:
            print(f"[WARN] Failed to configure env gait metrics: {e}")
    if bool(eval_cmd_profile_info.get("applied", False)):
        print(
            "[INFO] Eval command profile applied:",
            json.dumps(eval_cmd_profile_info, ensure_ascii=False),
        )
    elif "warning" in eval_cmd_profile_info:
        print(f"[WARN] {eval_cmd_profile_info['warning']}")
    if eval_velocity_curriculum_disabled:
        print("[INFO] Velocity command curriculum disabled for this evaluation run.")
    if hasattr(env.unwrapped, "_enable_terminal_snapshot"):
        env.unwrapped._enable_terminal_snapshot = True
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    threshold_temp, use_case_proxy, coil_to_case_delta_c = _thermal_failure_params(env.unwrapped)
    temp_metric_semantics, temp_metric_semantics_source = _temperature_metric_semantics_with_source(env.unwrapped)
    num_motors = int(env.unwrapped.motor_deg_state.fatigue_index.shape[1])
    fault_mode_eval, fault_fixed_eval = _fault_injection_params(env.unwrapped, num_motors=num_motors, strict=True)
    is_fixed_fault_eval = bool(fault_mode_eval == "single_motor_fixed" and fault_fixed_eval >= 0)
    if bool(args_cli.paper_protocol_strict) and not is_fixed_fault_eval:
        raise ValueError(
            "Paper protocol strict mode requires fixed-fault evaluation. "
            "Run with --eval_fault_mode single_motor_fixed --eval_fault_motor_id <0..11> "
            "or pass --no-paper-protocol-strict for exploratory runs."
        )
    if not is_fixed_fault_eval:
        print(
            "[WARN] Evaluation fault mode is not single_motor_fixed. "
            "For paper protocol, use --eval_fault_mode single_motor_fixed --eval_fault_motor_id <id>."
        )

    # Load trained policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Create output directory
    output_dir = args_cli.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Run selected scenarios by protocol mode.
    all_results = {}
    print(f"[INFO] Eval protocol mode: {eval_protocol_mode} | scenarios={scenario_names_for_run}")
    for scenario_name in scenario_names_for_run:
        summary = run_evaluation(
            env,
            policy,
            policy_nn,
            scenario_name,
            args_cli.num_episodes,
            temp_metric_semantics=temp_metric_semantics,
            nonzero_cmd_threshold=float(args_cli.eval_nonzero_cmd_threshold),
            stand_cmd_threshold=float(args_cli.eval_stand_cmd_threshold),
            contact_force_threshold=float(args_cli.eval_contact_force_threshold),
            eval_foot_body_names=eval_foot_body_names,
            eval_debug_print_body_names=bool(args_cli.eval_debug_print_body_names),
            cls_stand_actual_max=float(args_cli.eval_cls_stand_actual_max),
            cls_stand_ratio_min=float(args_cli.eval_cls_stand_ratio_min),
            cls_stand_cmd_ang_abs_max=float(args_cli.eval_cls_stand_cmd_ang_abs_max),
            cls_walk_cmd_min=float(args_cli.eval_cls_walk_cmd_min),
            cls_walk_nz_ratio_min=float(args_cli.eval_cls_walk_nz_ratio_min),
            cls_walk_progress_min=float(args_cli.eval_cls_walk_progress_min),
            cls_walk_progress_speed_min=float(args_cli.eval_cls_walk_progress_speed_min),
            cls_walk_progress_ratio_min=float(args_cli.eval_cls_walk_progress_ratio_min),
            cmd_profile_effective=str(effective_eval_cmd_profile),
            forced_walk_ramp_s=float(args_cli.eval_forced_walk_ramp_s),
            forced_walk_then_zero_walk_s=float(args_cli.eval_forced_walk_then_zero_walk_s),
            safe_stop_lin_vel_max=float(args_cli.eval_safe_stop_lin_vel_max),
            safe_stop_ang_vel_max=float(args_cli.eval_safe_stop_ang_vel_max),
            safe_stop_hold_s=float(args_cli.eval_safe_stop_hold_s),
            safe_stop_require_pose=bool(args_cli.eval_safe_stop_require_pose),
            safe_stop_max_tilt_rad=float(args_cli.eval_safe_stop_max_tilt_rad),
            safe_stop_min_height_m=float(args_cli.eval_safe_stop_min_height_m),
            oracle_safe_stop_steps=int(args_cli.eval_oracle_safe_stop_steps),
            crit_cmd_delta_active_eps=float(args_cli.eval_crit_cmd_delta_active_eps),
            thermal_threshold_c=float(threshold_temp) if threshold_temp is not None else None,
            dump_timeseries=bool(args_cli.eval_dump_timeseries),
            dump_timeseries_scenario=str(args_cli.eval_dump_timeseries_scenario),
            dump_timeseries_env_id=int(args_cli.eval_dump_timeseries_env_id),
            dump_timeseries_max_steps=int(args_cli.eval_dump_timeseries_max_steps),
            dump_timeseries_output_dir=str(output_dir),
        )
        all_results[scenario_name] = summary

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(output_dir, f"eval_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Results saved to: {result_path}")

    meta_path = os.path.join(output_dir, f"eval_{timestamp}_meta.json")
    critical_governor_enable_effective = (
        bool(getattr(env.unwrapped, "_crit_governor_enable"))
        if hasattr(env.unwrapped, "_crit_governor_enable")
        else None
    )
    critical_governor_v_cap_norm_effective = (
        float(getattr(env.unwrapped, "_crit_v_cap_norm"))
        if hasattr(env.unwrapped, "_crit_v_cap_norm")
        else None
    )
    critical_governor_wz_cap_effective = (
        float(getattr(env.unwrapped, "_crit_wz_cap"))
        if hasattr(env.unwrapped, "_crit_wz_cap")
        else None
    )
    critical_governor_sat_thr_effective = (
        float(getattr(env.unwrapped, "_crit_sat_thr"))
        if hasattr(env.unwrapped, "_crit_sat_thr")
        else None
    )
    critical_governor_sat_window_steps_effective = (
        int(getattr(env.unwrapped, "_crit_sat_window_steps"))
        if hasattr(env.unwrapped, "_crit_sat_window_steps")
        else None
    )
    critical_governor_sat_trigger_effective = (
        float(getattr(env.unwrapped, "_crit_sat_trigger"))
        if hasattr(env.unwrapped, "_crit_sat_trigger")
        else None
    )
    critical_governor_sat_trigger_hi_effective = (
        float(getattr(env.unwrapped, "_crit_sat_trigger_hi"))
        if hasattr(env.unwrapped, "_crit_sat_trigger_hi")
        else critical_governor_sat_trigger_effective
    )
    critical_governor_sat_trigger_lo_effective = (
        float(getattr(env.unwrapped, "_crit_sat_trigger_lo"))
        if hasattr(env.unwrapped, "_crit_sat_trigger_lo")
        else critical_governor_sat_trigger_effective
    )
    critical_governor_unlatch_require_low_cmd_effective = (
        bool(getattr(env.unwrapped, "_crit_unlatch_require_low_cmd"))
        if hasattr(env.unwrapped, "_crit_unlatch_require_low_cmd")
        else None
    )
    critical_governor_unlatch_require_sat_recovery_effective = (
        bool(getattr(env.unwrapped, "_crit_unlatch_require_sat_recovery"))
        if hasattr(env.unwrapped, "_crit_unlatch_require_sat_recovery")
        else None
    )
    critical_governor_latch_hold_steps_effective = (
        int(getattr(env.unwrapped, "_crit_latch_hold_steps"))
        if hasattr(env.unwrapped, "_crit_latch_hold_steps")
        else None
    )
    critical_governor_unlatch_stable_steps_effective = (
        int(getattr(env.unwrapped, "_crit_unlatch_stable_steps_req"))
        if hasattr(env.unwrapped, "_crit_unlatch_stable_steps_req")
        else None
    )
    critical_governor_unlatch_cmd_norm_effective = (
        float(getattr(env.unwrapped, "_crit_unlatch_cmd_norm"))
        if hasattr(env.unwrapped, "_crit_unlatch_cmd_norm")
        else None
    )
    critical_governor_post_unlatch_action_ramp_s_effective = (
        float(getattr(env.unwrapped, "_crit_post_unlatch_action_ramp_s"))
        if hasattr(env.unwrapped, "_crit_post_unlatch_action_ramp_s")
        else None
    )
    critical_governor_post_unlatch_action_delta_max_effective = (
        float(getattr(env.unwrapped, "_crit_post_unlatch_action_delta_max"))
        if hasattr(env.unwrapped, "_crit_post_unlatch_action_delta_max")
        else None
    )

    paper_b_contract_summary = summarize_paper_b_task_cfg(env.unwrapped.cfg)
    eval_meta = {
        "task": args_cli.task,
        "paper_b_obs_ablation": str(getattr(env.unwrapped.cfg, "paper_b_obs_ablation", "none")),
        "paper_b_sensor_preset": str(getattr(env.unwrapped.cfg, "paper_b_sensor_preset", "full")),
        "paper_b_runtime_overrides": paper_b_runtime_cfg,
        "paper_b_family": str(paper_b_contract_summary["paper_b_family"]),
        "paper_b_variant": str(paper_b_contract_summary["paper_b_variant"]),
        "paper_b_observation_scope": str(paper_b_contract_summary["paper_b_observation_scope"]),
        "paper_b_reward_scope": str(paper_b_contract_summary["paper_b_reward_scope"]),
        "paper_b_deployable": bool(paper_b_contract_summary["paper_b_deployable"]),
        "realobs_require_voltage_sensor": bool(getattr(env.unwrapped.cfg, "realobs_require_voltage_sensor", False)),
        "realobs_allow_true_voltage_fallback": bool(
            getattr(env.unwrapped.cfg, "realobs_allow_true_voltage_fallback", False)
        ),
        "realobs_require_case_temperature_proxy": bool(
            getattr(env.unwrapped.cfg, "realobs_require_case_temperature_proxy", False)
        ),
        "realobs_allow_case_temperature_from_coil_fallback": bool(
            getattr(env.unwrapped.cfg, "realobs_allow_case_temperature_from_coil_fallback", False)
        ),
        "realobs_voltage_source_effective": str(getattr(env.unwrapped, "_realobs_voltage_source", "n/a")),
        "realobs_case_temperature_source_effective": str(
            getattr(env.unwrapped, "_realobs_case_temperature_source", "n/a")
        ),
        "temperature_metric_semantics": temp_metric_semantics,
        "temperature_metric_field": f"final_max_temp_{temp_metric_semantics}",
        "thermal_termination_threshold_c": float(threshold_temp) if threshold_temp is not None else None,
        "thermal_termination_enabled": bool(threshold_temp is not None),
        "thermal_use_case_proxy": bool(use_case_proxy),
        "coil_to_case_delta_c": float(coil_to_case_delta_c),
        "temperature_metric_semantics_source": str(temp_metric_semantics_source),
        "scenario_injection_note": (
            "critical/aged injection ranges are adapted and capped with the shared case-proxy safe reset bound "
            "when case-proxy thermal termination is enabled."
        ),
        "fault_injection_mode_eval": str(fault_mode_eval),
        "fault_fixed_motor_id_eval": int(fault_fixed_eval),
        "paper_protocol_strict": bool(args_cli.paper_protocol_strict),
        "paper_protocol_fixed_fault_eval": is_fixed_fault_eval,
        "eval_protocol_mode": str(eval_protocol_mode),
        "eval_scenarios": list(scenario_names_for_run),
        "eval_cmd_profile": str(effective_eval_cmd_profile),
        "eval_cmd_profile_requested": str(requested_eval_cmd_profile),
        "eval_cmd_profile_effective": str(effective_eval_cmd_profile),
        "eval_safety_cmd_profile": str(args_cli.eval_safety_cmd_profile),
        "eval_forced_walk_lin_x_min": float(args_cli.eval_forced_walk_lin_x_min),
        "eval_forced_walk_lin_x_max": float(args_cli.eval_forced_walk_lin_x_max),
        "eval_forced_walk_ang_z_min": float(args_cli.eval_forced_walk_ang_z_min),
        "eval_forced_walk_ang_z_max": float(args_cli.eval_forced_walk_ang_z_max),
        "eval_forced_walk_ramp_s": float(args_cli.eval_forced_walk_ramp_s),
        "eval_forced_walk_then_zero_walk_s": float(args_cli.eval_forced_walk_then_zero_walk_s),
        "eval_cmd_profile_applied": bool(eval_cmd_profile_info.get("applied", False)),
        "eval_cmd_profile_info": eval_cmd_profile_info,
        "eval_velocity_curriculum_disabled": bool(eval_velocity_curriculum_disabled),
        "eval_nonzero_cmd_threshold": float(args_cli.eval_nonzero_cmd_threshold),
        "eval_stand_cmd_threshold": float(args_cli.eval_stand_cmd_threshold),
        "eval_safe_stop_lin_vel_max": float(args_cli.eval_safe_stop_lin_vel_max),
        "eval_safe_stop_ang_vel_max": float(args_cli.eval_safe_stop_ang_vel_max),
        "eval_safe_stop_hold_s": float(args_cli.eval_safe_stop_hold_s),
        "eval_safe_stop_require_pose": bool(args_cli.eval_safe_stop_require_pose),
        "eval_safe_stop_max_tilt_rad": float(args_cli.eval_safe_stop_max_tilt_rad),
        "eval_safe_stop_min_height_m": float(args_cli.eval_safe_stop_min_height_m),
        "eval_oracle_safe_stop_steps": int(args_cli.eval_oracle_safe_stop_steps),
        "eval_crit_cmd_delta_active_eps": float(args_cli.eval_crit_cmd_delta_active_eps),
        "eval_dump_timeseries": bool(args_cli.eval_dump_timeseries),
        "eval_dump_timeseries_scenario": str(args_cli.eval_dump_timeseries_scenario),
        "eval_dump_timeseries_env_id": int(args_cli.eval_dump_timeseries_env_id),
        "eval_dump_timeseries_max_steps": int(args_cli.eval_dump_timeseries_max_steps),
        "eval_contact_force_threshold": float(args_cli.eval_contact_force_threshold),
        "eval_debug_print_body_names": bool(args_cli.eval_debug_print_body_names),
        "eval_foot_body_names": list(eval_foot_body_names or []),
        "eval_contact_sensor_enable_path": contact_sensor_enable_path,
        "eval_contact_sensor_enable_applied": bool(contact_sensor_enable_path is not None),
        "eval_cls_stand_actual_max": float(args_cli.eval_cls_stand_actual_max),
        "eval_cls_stand_ratio_min": float(args_cli.eval_cls_stand_ratio_min),
        "eval_cls_stand_cmd_ang_abs_max": float(args_cli.eval_cls_stand_cmd_ang_abs_max),
        "eval_cls_walk_cmd_min": float(args_cli.eval_cls_walk_cmd_min),
        "eval_cls_walk_nz_ratio_min": float(args_cli.eval_cls_walk_nz_ratio_min),
        "eval_cls_walk_progress_min": float(args_cli.eval_cls_walk_progress_min),
        "eval_cls_walk_progress_speed_min": float(args_cli.eval_cls_walk_progress_speed_min),
        "eval_cls_walk_progress_ratio_min": float(args_cli.eval_cls_walk_progress_ratio_min),
        "critical_governor_enable_effective": critical_governor_enable_effective,
        "critical_governor_v_cap_norm_effective": critical_governor_v_cap_norm_effective,
        "critical_governor_wz_cap_effective": critical_governor_wz_cap_effective,
        "critical_governor_sat_thr_effective": critical_governor_sat_thr_effective,
        "critical_governor_sat_window_steps_effective": critical_governor_sat_window_steps_effective,
        "critical_governor_sat_trigger_effective": critical_governor_sat_trigger_effective,
        "critical_governor_sat_trigger_hi_effective": critical_governor_sat_trigger_hi_effective,
        "critical_governor_sat_trigger_lo_effective": critical_governor_sat_trigger_lo_effective,
        "critical_governor_unlatch_require_low_cmd_effective": critical_governor_unlatch_require_low_cmd_effective,
        "critical_governor_unlatch_require_sat_recovery_effective": critical_governor_unlatch_require_sat_recovery_effective,
        "critical_governor_latch_hold_steps_effective": critical_governor_latch_hold_steps_effective,
        "critical_governor_unlatch_stable_steps_effective": critical_governor_unlatch_stable_steps_effective,
        "critical_governor_unlatch_cmd_norm_effective": critical_governor_unlatch_cmd_norm_effective,
        "critical_governor_post_unlatch_action_ramp_s_effective": (
            critical_governor_post_unlatch_action_ramp_s_effective
        ),
        "critical_governor_post_unlatch_action_delta_max_effective": (
            critical_governor_post_unlatch_action_delta_max_effective
        ),
        "critical_governor_sat_metric_default": "crit_sat_any_over_thr_ratio_ep",
        "critical_governor_sat_metric_legacy_terminal": "crit_sat_any_over_thr_ratio",
    }
    with open(meta_path, "w") as f:
        json.dump(eval_meta, f, indent=2)
    print(f"[DONE] Evaluation metadata saved to: {meta_path}")

    reward_csv_path = _write_reward_breakdown_csv(all_results=all_results, output_dir=output_dir, timestamp=timestamp)
    if reward_csv_path is not None:
        print(f"[DONE] Reward breakdown CSV saved to: {reward_csv_path}")

    # Print summary table
    if eval_protocol_mode == "safety_only":
        print_safety_table(all_results, temp_metric_semantics=temp_metric_semantics)
    else:
        print_paper_table(all_results, temp_metric_semantics=temp_metric_semantics)

    env.close()


def print_safety_table(results: dict, temp_metric_semantics: str = "coil_hotspot"):
    """Print safety-only summary table for critical scenario analysis."""
    print("\n" + "=" * 296)
    print("  SAFETY TABLE: Critical Safe-Stop Metrics")
    print("=" * 296)
    print(f"Note: Max Temp semantics = {temp_metric_semantics}")
    header = (
        f"{'Scenario':<12} | {'SafeStop%':>10} | {'T_safe(s)':>10} | {'Nsafe':>6} | "
        f"{'T_fall(s)':>10} | {'Nfall':>6} | {'E_safe(J)':>10} | {'Nener':>6} | "
        f"{'PeakTemp':>10} | {'ThermMargin':>11} | {'PeakSat':>8} | {'GovSatEp%':>9} | {'Latch%':>7} | "
        f"{'LatStep%':>8} | {'SatWin%':>8} | {'CmdΔAct%':>9} | {'LowΔLat%':>9} | {'FbSat%':>7} | "
        f"{'ModeSL%':>8} | {'LatchN':>7} | {'DurLat(s)':>9} | {'T1Latch':>8} | {'TUnlat':>8} | "
        f"{'PostUSat95':>10} | {'ActΔLat':>8} | {'CmdΔLat':>8}"
    )
    print(header)
    print("-" * 296)

    for scenario_name, summary in results.items():
        safe_rate = summary.get("safe_stop_success_rate", {}).get("mean", float("nan")) * 100.0
        t_safe_stats = summary.get("time_to_safe_stop_s", {})
        t_fall_stats = summary.get("time_to_fall_s", {})
        e_safe_stats = summary.get("energy_to_safe_stop_j", {})
        peak_temp = summary.get("peak_temp_episode_c", {}).get("mean", float("nan"))
        therm_margin = summary.get("thermal_margin_to_threshold_c", {}).get("mean", float("nan"))
        peak_sat = summary.get("peak_saturation_episode", {}).get("mean", float("nan"))
        gov_sat_ep = summary.get("crit_sat_any_over_thr_ratio_ep", {}).get("mean", float("nan")) * 100.0
        latch_ratio = summary.get("crit_is_latched", {}).get("mean", float("nan")) * 100.0
        latched_step_ratio = summary.get("crit_latched_step_ratio_ep", {}).get("mean", float("nan")) * 100.0
        sat_window_ratio = summary.get("crit_sat_window_ratio_mean_ep", {}).get("mean", float("nan")) * 100.0
        cmd_delta_active_ratio = summary.get("crit_cmd_delta_active_ratio_ep", {}).get("mean", float("nan")) * 100.0
        low_delta_latched_ratio = summary.get("crit_latched_low_cmd_delta_ratio_ep", {}).get("mean", float("nan")) * 100.0
        fallback_ratio = summary.get("crit_sat_ratio_fallback_step_ratio_ep", {}).get("mean", float("nan")) * 100.0
        mode_stop_latch_ratio = summary.get("crit_governor_mode_stop_latch_ratio_ep", {}).get("mean", float("nan")) * 100.0
        latch_count_ep = summary.get("crit_latch_count_ep", {}).get("mean", float("nan"))
        latch_dur_s_ep = summary.get("crit_latch_mean_dur_s_ep", {}).get("mean", float("nan"))
        t_first_latch_s = summary.get("crit_time_to_first_latch_s", {}).get("mean", float("nan"))
        t_unlatch_s = summary.get("crit_time_to_unlatch_s", {}).get("mean", float("nan"))
        post_unlatch_sat_p95 = summary.get(
            "crit_post_unlatch_fixedwin_peak_sat_p95_ep",
            summary.get("crit_post_unlatch_peak_sat_p95_ep", {}),
        ).get("mean", float("nan"))
        action_delta_latched = summary.get("crit_action_delta_latched_norm_mean_ep", {}).get(
            "mean",
            summary.get("crit_action_delta_latched_norm_step", {}).get("mean", float("nan")),
        )
        cmd_delta_latched = summary.get("crit_cmd_delta_latched_norm_mean_ep", {}).get(
            "mean",
            summary.get("crit_cmd_delta_latched_norm_step", {}).get("mean", float("nan")),
        )

        t_safe = t_safe_stats.get("mean", float("nan"))
        t_safe_n = int(t_safe_stats.get("valid_count", t_safe_stats.get("count", 0)))
        t_fall = t_fall_stats.get("mean", float("nan"))
        t_fall_n = int(t_fall_stats.get("valid_count", t_fall_stats.get("count", 0)))
        e_safe = e_safe_stats.get("mean", float("nan"))
        e_safe_n = int(e_safe_stats.get("valid_count", e_safe_stats.get("count", 0)))

        row = (
            f"{scenario_name:<12} | {safe_rate:>9.1f}% | {t_safe:>10.2f} | {t_safe_n:>6d} | "
            f"{t_fall:>10.2f} | {t_fall_n:>6d} | {e_safe:>10.2f} | {e_safe_n:>6d} | "
            f"{peak_temp:>9.2f}C | {therm_margin:>10.2f}C | {peak_sat:>8.3f} | {gov_sat_ep:>8.1f}% | {latch_ratio:>6.1f}% | "
            f"{latched_step_ratio:>7.1f}% | {sat_window_ratio:>7.1f}% | {cmd_delta_active_ratio:>8.1f}% | "
            f"{low_delta_latched_ratio:>8.1f}% | {fallback_ratio:>6.1f}% | {mode_stop_latch_ratio:>7.1f}% | "
            f"{latch_count_ep:>7.2f} | {latch_dur_s_ep:>9.2f} | {t_first_latch_s:>8.2f} | {t_unlatch_s:>8.2f} | "
            f"{post_unlatch_sat_p95:>10.3f} | {action_delta_latched:>8.3f} | {cmd_delta_latched:>8.3f}"
        )
        print(row)

    print("=" * 296)
    print(
        "[INFO] safety_only interpretation: prioritize safe-stop metrics. "
        "Governor saturation default = crit_sat_any_over_thr_ratio_ep."
    )


def print_paper_table(results: dict, temp_metric_semantics: str = "coil_hotspot"):
    """Print a LaTeX-friendly summary table."""
    print("\n" + "=" * 90)
    print("  PAPER TABLE: Performance Under Degradation Scenarios")
    print("=" * 90)
    print(f"Note: Max Temp semantics = {temp_metric_semantics}")
    temp_col = "MaxTemp(case)" if temp_metric_semantics == "case_proxy" else "MaxTemp(coil)"
    header = f"{'Scenario':<12} | {'Survival%':>10} | {'Track Err':>10} | {'Power(W)':>10} | {'Energy(J)':>10} | {temp_col:>12} | {'Final SOC':>10}"
    print(header)
    print("-" * 90)

    for scenario_name, summary in results.items():
        surv = summary.get("survived", {}).get("mean", 0) * 100
        trk = summary.get("mean_tracking_error_xy", {}).get("mean", 0)
        pwr = summary.get("mean_power", {}).get("mean", 0)
        eng = summary.get("total_energy", {}).get("mean", 0)
        temp_key = f"final_max_temp_{temp_metric_semantics}"
        tmp = summary.get(temp_key, summary.get("final_max_temp", {})).get("mean", 0)
        soc = summary.get("final_soc", {}).get("mean", 0)

        row = f"{scenario_name:<12} | {surv:>9.1f}% | {trk:>10.4f} | {pwr:>10.1f} | {eng:>10.1f} | {tmp:>11.1f}°C | {soc:>10.3f}"
        print(row)

    print("=" * 90)

    if "critical" in results:
        crit = results["critical"]
        safe_rate = crit.get("safe_stop_success_rate", {}).get("mean", float("nan")) * 100.0
        t_safe = crit.get("time_to_safe_stop_s", {}).get("mean", float("nan"))
        t_fall = crit.get("time_to_fall_s", {}).get("mean", float("nan"))
        e_safe = crit.get("energy_to_safe_stop_j", {}).get("mean", float("nan"))
        margin = crit.get("thermal_margin_to_threshold_c", {}).get("mean", float("nan"))
        sat_peak = crit.get("peak_saturation_episode", {}).get("mean", float("nan"))
        gov_sat_ep = crit.get("crit_sat_any_over_thr_ratio_ep", {}).get("mean", float("nan")) * 100.0
        latch_ratio = crit.get("crit_is_latched", {}).get("mean", float("nan")) * 100.0
        latched_step_ratio = crit.get("crit_latched_step_ratio_ep", {}).get("mean", float("nan")) * 100.0
        sat_window_ratio = crit.get("crit_sat_window_ratio_mean_ep", {}).get("mean", float("nan")) * 100.0
        cmd_delta_active_ratio = crit.get("crit_cmd_delta_active_ratio_ep", {}).get("mean", float("nan")) * 100.0
        low_delta_latched_ratio = crit.get("crit_latched_low_cmd_delta_ratio_ep", {}).get("mean", float("nan")) * 100.0
        sat_fallback_ratio = crit.get("crit_sat_ratio_fallback_step_ratio_ep", {}).get("mean", float("nan")) * 100.0
        mode_stop_latch_ratio = crit.get("crit_governor_mode_stop_latch_ratio_ep", {}).get("mean", float("nan")) * 100.0
        latch_count_ep = crit.get("crit_latch_count_ep", {}).get("mean", float("nan"))
        latch_dur_s_ep = crit.get("crit_latch_mean_dur_s_ep", {}).get("mean", float("nan"))
        t_first_latch = crit.get("crit_time_to_first_latch_s", {}).get("mean", float("nan"))
        t_unlatch = crit.get("crit_time_to_unlatch_s", {}).get("mean", float("nan"))
        post_unlatch_sat_p95 = crit.get(
            "crit_post_unlatch_fixedwin_peak_sat_p95_ep",
            crit.get("crit_post_unlatch_peak_sat_p95_ep", {}),
        ).get("mean", float("nan"))
        print("  CRITICAL SAFETY METRICS")
        print(
            f"  safe_stop_success={safe_rate:.1f}% | time_to_safe_stop={t_safe:.2f}s | "
            f"time_to_fall={t_fall:.2f}s | energy_to_safe_stop={e_safe:.2f}J | "
            f"thermal_margin={margin:.2f}C | peak_saturation={sat_peak:.3f} | "
            f"gov_sat_ep={gov_sat_ep:.1f}% | latch={latch_ratio:.1f}% | "
            f"latched_step_ep={latched_step_ratio:.1f}% | sat_window_ep={sat_window_ratio:.1f}% | "
            f"cmd_delta_active_ep={cmd_delta_active_ratio:.1f}% | "
            f"low_delta_latched_ep={low_delta_latched_ratio:.1f}% | "
            f"sat_fallback_ep={sat_fallback_ratio:.1f}% | mode_stop_latch_ep={mode_stop_latch_ratio:.1f}% | "
            f"latch_count_ep={latch_count_ep:.2f} | latch_mean_dur_s_ep={latch_dur_s_ep:.2f} | "
            f"time_to_first_latch={t_first_latch:.2f}s | time_to_unlatch={t_unlatch:.2f}s | "
            f"post_unlatch_peak_sat_p95={post_unlatch_sat_p95:.3f}"
        )
        print("=" * 90)

    # LaTeX output
    print("\n% --- LaTeX Table ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Performance comparison under degradation scenarios (temperature semantics: {temp_metric_semantics})}}")
    print("\\label{tab:degradation_results}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print(f"Scenario & Survival (\\%) & Track. Err & Power (W) & Energy (J) & Max Temp ({temp_metric_semantics}, °C) & Final SOC \\\\")
    print("\\midrule")
    for scenario_name, summary in results.items():
        surv = summary.get("survived", {}).get("mean", 0) * 100
        trk = summary.get("mean_tracking_error_xy", {}).get("mean", 0)
        trk_std = summary.get("mean_tracking_error_xy", {}).get("std", 0)
        pwr = summary.get("mean_power", {}).get("mean", 0)
        eng = summary.get("total_energy", {}).get("mean", 0)
        temp_key = f"final_max_temp_{temp_metric_semantics}"
        tmp = summary.get(temp_key, summary.get("final_max_temp", {})).get("mean", 0)
        soc = summary.get("final_soc", {}).get("mean", 0)
        print(f"{scenario_name.capitalize()} & {surv:.1f} & {trk:.4f}$\\pm${trk_std:.4f} & {pwr:.1f} & {eng:.1f} & {tmp:.1f} & {soc:.3f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    main()
    simulation_app.close()
