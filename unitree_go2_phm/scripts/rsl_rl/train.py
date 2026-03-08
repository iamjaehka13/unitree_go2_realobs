# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train an RSL-RL policy with optional PHM fault-sampling overrides/schedules."""

"""Training entrypoint.

AppLauncher must be initialized before importing Isaac task modules.
"""

import argparse
import math
import os
import sys

# Deterministic CuBLAS config must be set before torch/CUDA context is initialized.
# Keep user's explicit setting if already provided.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--train_fault_mode",
    type=str,
    default="default",
    choices=["default", "single_motor_random", "single_motor_fixed", "all_motors_random"],
    help="Override PHM fault injection mode for training. 'default' keeps task config as-is.",
)
parser.add_argument(
    "--train_fault_motor_id",
    type=int,
    default=-1,
    help=(
        "If set to [0..11], force single_motor_fixed training at this motor index "
        "(overrides --train_fault_mode)."
    ),
)
parser.add_argument(
    "--train_fault_pair_uniform",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Override mirror-pair-uniform sampling for single_motor_random mode. "
        "Use --train-fault-pair-uniform/--no-train-fault-pair-uniform."
    ),
)
parser.add_argument(
    "--train_fault_hold_steps",
    type=int,
    default=None,
    help="Override fault motor hold window (env steps) for single_motor_random mode.",
)
parser.add_argument(
    "--train_fault_focus_prob",
    type=float,
    default=None,
    help=(
        "Override focus control for single_motor_random mode. "
        "In plain focus sampling this is the replacement probability; "
        "in weighted pair sampling it becomes the uniform-vs-target mixing alpha "
        "(0.0~1.0, 0 disables focus bias)."
    ),
)
parser.add_argument(
    "--train_fault_focus_motor_ids",
    type=str,
    default=None,
    help=(
        "Comma-separated motor IDs for hard-case focus sampling in single_motor_random mode "
        "(e.g., '0,3,7,10')."
    ),
)
parser.add_argument(
    "--train_fault_focus_pairs",
    type=str,
    default=None,
    help=(
        "Semicolon-separated focus motor pairs for hard-case sampling "
        "(e.g., '7-10;2-8' or '7,10;2,8'). "
        "If omitted, all mirror pairs are auto-selected when no explicit focus selector "
        "and no motor-adaptive focus are requested."
    ),
)
parser.add_argument(
    "--train_fault_pair_weighted_enable",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Enable weighted mirror-pair sampling for single_motor_random mode. "
        "When enabled, pair probability uses a floor/cap-clamped mixture and "
        "--train_fault_focus_prob becomes the uniform-vs-target mixing alpha."
    ),
)
parser.add_argument(
    "--train_fault_pair_prob_floor",
    type=float,
    default=None,
    help="Lower clamp bound for per-pair sampling probability (0~1).",
)
parser.add_argument(
    "--train_fault_pair_prob_cap",
    type=float,
    default=None,
    help="Upper clamp bound for per-pair sampling probability (0~1).",
)
parser.add_argument(
    "--train_fault_pair_target_weights",
    type=str,
    default=None,
    help=(
        "Comma-separated target pair weights in mirror-pair order "
        "[(0,3),(1,4),(2,5),(6,9),(7,10),(8,11)] "
        "(e.g., '0.10,0.10,0.30,0.10,0.25,0.15')."
    ),
)
parser.add_argument(
    "--train_fault_pair_adaptive_enable",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Enable adaptive difficulty-driven pair target distribution. "
        "Requires weighted pair sampling."
    ),
)
parser.add_argument(
    "--train_fault_pair_adaptive_mix",
    type=float,
    default=None,
    help="Blend ratio between manual target and adaptive target (0~1).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_beta",
    type=float,
    default=None,
    help="Softmax sharpness for adaptive pair difficulty -> probability mapping (>=0).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_ema",
    type=float,
    default=None,
    help="EMA decay for adaptive target probabilities (0~1, higher=slower update).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_min_episode_per_pair",
    type=float,
    default=None,
    help="Episode-count confidence scale for adaptive pair difficulty (>=1).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_w_fail",
    type=float,
    default=None,
    help="Adaptive difficulty weight for non-timeout termination ratio (>=0).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_w_sat",
    type=float,
    default=None,
    help="Adaptive difficulty weight for saturation ratio proxy (>=0).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_w_latch",
    type=float,
    default=None,
    help="Adaptive difficulty weight for latch ratio proxy (>=0).",
)
parser.add_argument(
    "--train_fault_pair_adaptive_sat_scale",
    type=float,
    default=None,
    help="Normalization scale for adaptive saturation mean term (>0).",
)
parser.add_argument(
    "--train_fault_motor_adaptive_enable",
    action=argparse.BooleanOptionalAction,
    default=None,
    help=(
        "Enable recent worst-motor focus sampling for single_motor_random mode. "
        "This keeps the base sampler and replaces a focus-prob fraction with recent worst top-k motors."
    ),
)
parser.add_argument(
    "--train_fault_motor_adaptive_topk",
    type=int,
    default=None,
    help="Number of recent worst motors to include in adaptive focus set (>=1).",
)
parser.add_argument(
    "--train_fault_motor_adaptive_min_episode_per_motor",
    type=float,
    default=None,
    help="Episode-count confidence scale for adaptive worst-motor ranking (>=1).",
)
parser.add_argument(
    "--train_fault_focus_ramp_start_iter",
    type=int,
    default=None,
    help="Absolute learning iteration where focus-prob ramp starts (inclusive).",
)
parser.add_argument(
    "--train_fault_focus_ramp_end_iter",
    type=int,
    default=None,
    help="Absolute learning iteration where focus-prob ramp ends (inclusive).",
)
parser.add_argument(
    "--train_fault_focus_ramp_start_prob",
    type=float,
    default=None,
    help="Focus probability at ramp start iteration.",
)
parser.add_argument(
    "--train_fault_focus_ramp_end_prob",
    type=float,
    default=None,
    help="Focus probability at ramp end iteration.",
)
parser.add_argument(
    "--train_fault_focus_ramp_segment_iters",
    type=int,
    default=50,
    help="Iteration granularity for focus-prob updates while ramp is active.",
)
parser.add_argument(
    "--num_steps_per_env",
    type=int,
    default=None,
    help="Override PPO rollout horizon (steps per environment, per iteration).",
)
parser.add_argument(
    "--num_mini_batches",
    type=int,
    default=None,
    help="Override PPO mini-batch count per update (higher increases update compute).",
)
parser.add_argument(
    "--num_learning_epochs",
    type=int,
    default=None,
    help="Override PPO learning epochs per update (higher increases update compute).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--debug_startup",
    action="store_true",
    default=False,
    help="Enable verbose startup tracing and a one-step warmup before runner construction.",
)
parser.add_argument(
    "--perf_mode",
    action="store_true",
    default=False,
    help="Use throughput-oriented backend settings (non-deterministic, TF32 enabled).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
carb.settings.get_settings().set_int("/log/channels/omni.physx.tensors.plugin/level", 1)

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform
from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Isaac task imports follow AppLauncher initialization."""

import gymnasium as gym
import logging
import random
import time
import torch
import numpy as np
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import unitree_go2_phm.tasks  # noqa: F401

_FOCUS_ALL_MIRROR_PAIRS_12: tuple[tuple[int, int], ...] = (
    (0, 3),
    (1, 4),
    (2, 5),
    (6, 9),
    (7, 10),
    (8, 11),
)


def _parse_focus_motor_ids(raw_focus: str) -> tuple[int, ...]:
    """Parse CSV-style motor id list into a unique ordered tuple."""
    raw_focus = str(raw_focus).strip()
    if raw_focus == "":
        return ()

    parts = [p.strip() for p in raw_focus.replace(";", ",").split(",") if p.strip() != ""]
    if len(parts) == 0:
        return ()

    parsed_ids: list[int] = []
    for part in parts:
        try:
            motor_id = int(part)
        except Exception as e:
            raise ValueError(
                f"Invalid --train_fault_focus_motor_ids entry '{part}' (expected integer)."
            ) from e
        if not (0 <= motor_id < 12):
            raise ValueError(
                f"Invalid --train_fault_focus_motor_ids entry '{motor_id}' (expected 0..11)."
            )
        parsed_ids.append(motor_id)

    # Keep order but remove duplicates.
    return tuple(dict.fromkeys(parsed_ids))


def _parse_focus_pairs(raw_pairs: str) -> tuple[tuple[int, int], ...]:
    """Parse focus pair list into unique ordered tuple of int pairs."""
    raw_pairs = str(raw_pairs).strip()
    if raw_pairs == "":
        return ()

    # Pair separator: ';' (also accept '|').
    chunks = [c.strip() for c in raw_pairs.replace("|", ";").split(";") if c.strip() != ""]
    if len(chunks) == 0:
        return ()

    parsed_pairs: list[tuple[int, int]] = []
    for chunk in chunks:
        if "-" in chunk:
            parts = [p.strip() for p in chunk.split("-")]
        elif ":" in chunk:
            parts = [p.strip() for p in chunk.split(":")]
        elif "/" in chunk:
            parts = [p.strip() for p in chunk.split("/")]
        elif "," in chunk:
            parts = [p.strip() for p in chunk.split(",")]
        else:
            raise ValueError(
                f"Invalid --train_fault_focus_pairs entry '{chunk}'. "
                "Expected pair format like '7-10' or '7,10'."
            )

        if len(parts) != 2:
            raise ValueError(
                f"Invalid --train_fault_focus_pairs entry '{chunk}'. "
                "Expected exactly two motor ids per pair."
            )

        try:
            a = int(parts[0])
            b = int(parts[1])
        except Exception as e:
            raise ValueError(
                f"Invalid --train_fault_focus_pairs entry '{chunk}' (expected integers)."
            ) from e

        if not (0 <= a < 12) or not (0 <= b < 12):
            raise ValueError(
                f"Invalid --train_fault_focus_pairs entry '{chunk}' (expected motor ids in 0..11)."
            )
        if a == b:
            raise ValueError(
                f"Invalid --train_fault_focus_pairs entry '{chunk}' (pair members must differ)."
            )
        parsed_pairs.append((a, b))

    # Keep order but remove duplicates.
    return tuple(dict.fromkeys(parsed_pairs))


def _parse_pair_target_weights(raw_weights: str, num_pairs: int = 6) -> tuple[float, ...]:
    """Parse comma-separated non-negative weights for mirror-pair target distribution."""
    raw_weights = str(raw_weights).strip()
    if raw_weights == "":
        return ()

    parts = [p.strip() for p in raw_weights.replace(";", ",").split(",") if p.strip() != ""]
    if len(parts) != int(num_pairs):
        raise ValueError(
            f"Invalid --train_fault_pair_target_weights: expected {num_pairs} values, got {len(parts)}."
        )

    values: list[float] = []
    for part in parts:
        try:
            val = float(part)
        except Exception as e:
            raise ValueError(
                f"Invalid --train_fault_pair_target_weights entry '{part}' (expected float)."
            ) from e
        if val < 0.0:
            raise ValueError(
                f"Invalid --train_fault_pair_target_weights entry '{part}' (expected >= 0)."
            )
        values.append(val)

    if sum(values) <= 0.0:
        raise ValueError("--train_fault_pair_target_weights sum must be > 0.")
    return tuple(values)


def _resolve_focus_ramp_cfg(args: argparse.Namespace) -> dict | None:
    """Build validated focus-ramp config from CLI arguments."""
    fields = (
        args.train_fault_focus_ramp_start_iter,
        args.train_fault_focus_ramp_end_iter,
        args.train_fault_focus_ramp_start_prob,
        args.train_fault_focus_ramp_end_prob,
    )
    if all(v is None for v in fields):
        return None
    if any(v is None for v in fields):
        raise ValueError(
            "Focus ramp requires all arguments: "
            "--train_fault_focus_ramp_start_iter, --train_fault_focus_ramp_end_iter, "
            "--train_fault_focus_ramp_start_prob, --train_fault_focus_ramp_end_prob."
        )

    start_iter = int(args.train_fault_focus_ramp_start_iter)
    end_iter = int(args.train_fault_focus_ramp_end_iter)
    start_prob = float(args.train_fault_focus_ramp_start_prob)
    end_prob = float(args.train_fault_focus_ramp_end_prob)
    segment_iters = int(args.train_fault_focus_ramp_segment_iters)

    if start_iter < 0 or end_iter < 0:
        raise ValueError("Focus ramp iterations must be >= 0.")
    if end_iter <= start_iter:
        raise ValueError(
            f"Invalid focus ramp range: start_iter={start_iter}, end_iter={end_iter}. "
            "Require end_iter > start_iter."
        )
    if not (0.0 <= start_prob <= 1.0):
        raise ValueError(
            f"Invalid --train_fault_focus_ramp_start_prob={start_prob} (expected in [0, 1])."
        )
    if not (0.0 <= end_prob <= 1.0):
        raise ValueError(
            f"Invalid --train_fault_focus_ramp_end_prob={end_prob} (expected in [0, 1])."
        )
    if segment_iters <= 0:
        raise ValueError(
            f"Invalid --train_fault_focus_ramp_segment_iters={segment_iters} (expected > 0)."
        )

    return {
        "start_iter": start_iter,
        "end_iter": end_iter,
        "start_prob": start_prob,
        "end_prob": end_prob,
        "segment_iters": segment_iters,
    }


def _piecewise_linear_value(
    iter_idx: int,
    start_iter: int,
    end_iter: int,
    start_value: float,
    end_value: float,
) -> float:
    """Linear interpolation clamped to [start_iter, end_iter]."""
    if iter_idx <= start_iter:
        return float(start_value)
    if iter_idx >= end_iter:
        return float(end_value)
    t = float(iter_idx - start_iter) / float(max(end_iter - start_iter, 1))
    return float(start_value + (end_value - start_value) * t)


def _set_env_fault_focus_prob(runner: OnPolicyRunner | DistillationRunner, focus_prob: float) -> bool:
    """Update runtime env focus probability so reset-time PHM sampling uses new value."""
    try:
        env = getattr(runner, "env", None)
        if env is None:
            return False
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        cfg_obj = getattr(base_env, "cfg", None)
        if cfg_obj is None:
            return False
        setattr(cfg_obj, "phm_fault_focus_prob", float(focus_prob))
        return True
    except Exception:
        return False

def _configure_torch_runtime(perf_mode: bool) -> None:
    """Configure backend flags for either reproducibility or throughput."""
    if perf_mode:
        # Throughput mode for faster training on Ampere+ GPUs (e.g., RTX 3090).
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        logger.warning(
            "[Runtime] PERF mode enabled: TF32=ON, deterministic=OFF, cudnn.benchmark=ON (results may vary run-to-run)."
        )
        return

    # Reproducibility-first defaults for paper experiments.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("[Runtime] Deterministic mode enabled: TF32=OFF, cudnn.benchmark=OFF.")


def _piecewise_stage_value(
    iter_idx: int,
    stage1_end_iter: int,
    stage2_end_iter: int,
    stage0_value: float,
    stage1_value: float,
    stage2_value: float,
) -> float:
    """3-stage piecewise constant schedule value by absolute learning iteration."""
    if iter_idx < int(stage1_end_iter):
        return float(stage0_value)
    if iter_idx < int(stage2_end_iter):
        return float(stage1_value)
    return float(stage2_value)


def _set_runner_entropy_coef(runner: OnPolicyRunner | DistillationRunner, entropy_coef: float) -> bool:
    """Set entropy coefficient on runner algorithm if supported."""
    alg = getattr(runner, "alg", None)
    if alg is None or not hasattr(alg, "entropy_coef"):
        return False
    setattr(alg, "entropy_coef", float(entropy_coef))
    return True


def _set_runner_action_std_cap_only(
    runner: OnPolicyRunner | DistillationRunner,
    target_std: float,
    prev_applied_log: torch.Tensor | None = None,
    enable_rate_limit: bool = False,
    max_up_log: float = 0.0,
    max_down_log: float = 0.0,
) -> tuple[bool, float, float | None, float | None, float | None, torch.Tensor | None]:
    """Apply action-std with cap-only semantics and optional target-tracking log-space slew limiter."""
    alg = getattr(runner, "alg", None)
    policy = getattr(alg, "policy", None) if alg is not None else None
    if policy is None:
        return False, max(float(target_std), 1e-6), None, None, None, None
    std_value = max(float(target_std), 1e-6)
    with torch.no_grad():
        cap_value_log = math.log(std_value)
        if hasattr(policy, "std"):
            # Some policy variants expose std directly. We still rate-limit in log-space.
            current_std = torch.clamp(policy.std.data, min=1e-6)
            current_mean = float(current_std.mean().item())
            cap_std_tensor = torch.full_like(current_std, std_value)
            cap_log_tensor = torch.full_like(current_std, cap_value_log)

            # Cap-only semantics: never increase exploration std at stage boundaries.
            capped_std = torch.minimum(current_std, cap_std_tensor)
            capped_log = torch.log(torch.clamp(capped_std, min=1e-6))

            applied_log = capped_log
            prev_mean = None
            if enable_rate_limit:
                prev_log = prev_applied_log
                if prev_log is None or prev_log.shape != capped_log.shape:
                    prev_log = capped_log.clone()
                else:
                    prev_log = prev_log.to(device=capped_log.device, dtype=capped_log.dtype)
                prev_mean = float(torch.exp(prev_log).mean().item())
                # Recovery mode: move from previous applied std toward target cap (not toward current std).
                desired_log = cap_log_tensor
                diff = torch.clamp(desired_log - prev_log, min=-float(max_down_log), max=float(max_up_log))
                applied_log = prev_log + diff
                # Preserve cap-only invariant even under limiter dynamics.
                applied_log = torch.minimum(applied_log, cap_log_tensor)

            applied_std = torch.exp(applied_log)
            policy.std.data.copy_(applied_std)
            applied_mean = float(applied_std.mean().item())
            return True, std_value, current_mean, prev_mean, applied_mean, applied_log.detach().clone()
        if hasattr(policy, "log_std"):
            current_log_std = policy.log_std.data
            current_mean = float(torch.exp(current_log_std).mean().item())
            cap_log_tensor = torch.full_like(current_log_std, cap_value_log)
            # Cap-only semantics in log-space for log_std policies.
            capped_log = torch.minimum(current_log_std, cap_log_tensor)

            applied_log_std = capped_log
            prev_mean = None
            if enable_rate_limit:
                prev_log = prev_applied_log
                if prev_log is None or prev_log.shape != capped_log.shape:
                    prev_log = capped_log.clone()
                else:
                    prev_log = prev_log.to(device=capped_log.device, dtype=capped_log.dtype)
                prev_mean = float(torch.exp(prev_log).mean().item())
                # Recovery mode: move from previous applied std toward target cap (not toward current std).
                desired_log = cap_log_tensor
                diff = torch.clamp(desired_log - prev_log, min=-float(max_down_log), max=float(max_up_log))
                applied_log_std = prev_log + diff
                # Preserve cap-only invariant even under limiter dynamics.
                applied_log_std = torch.minimum(applied_log_std, cap_log_tensor)

            policy.log_std.data.copy_(applied_log_std)
            applied_mean = float(torch.exp(applied_log_std).mean().item())
            return True, std_value, current_mean, prev_mean, applied_mean, applied_log_std.detach().clone()
    return False, std_value, None, None, None, None


def _learn_with_exploration_schedule(
    runner: OnPolicyRunner | DistillationRunner,
    agent_cfg: RslRlBaseRunnerCfg,
    focus_ramp_cfg: dict | None = None,
    debug_startup: bool = False,
) -> None:
    """
    Train with staged entropy/action-std schedule and optional fault-focus ramp.

    Default target profile:
      - entropy: 0.01 (iter < 1000), 0.005 (1000~1999), 0.003 (>=2000)
      - action std target-cap: 1.0 (iter < 1000), 0.7 (1000~1999), 0.5 (>=2000)
      - optional late limiter (default on): iter>=2200, segment=20, up=1.02x, down=1.00x(disabled)
      - optional focus ramp: update reset-time `phm_fault_focus_prob` on configured iteration boundaries
    """
    total_iters = int(agent_cfg.max_iterations)
    if total_iters <= 0:
        return

    entropy_enable = bool(getattr(agent_cfg, "entropy_schedule_enable", True))
    entropy_stage1_end = int(getattr(agent_cfg, "entropy_stage1_end_iter", 1000))
    entropy_stage2_end = int(getattr(agent_cfg, "entropy_stage2_end_iter", 2000))
    entropy_stage0 = float(getattr(agent_cfg, "entropy_stage0_value", 0.01))
    entropy_stage1 = float(getattr(agent_cfg, "entropy_stage1_value", 0.005))
    entropy_stage2 = float(getattr(agent_cfg, "entropy_stage2_value", 0.003))

    std_enable = bool(getattr(agent_cfg, "action_std_schedule_enable", True))
    std_stage1_end = int(getattr(agent_cfg, "action_std_stage1_end_iter", 1000))
    std_stage2_end = int(getattr(agent_cfg, "action_std_stage2_end_iter", 2000))
    std_stage0 = float(getattr(agent_cfg, "action_std_stage0_value", 1.0))
    std_stage1 = float(getattr(agent_cfg, "action_std_stage1_value", 0.7))
    std_stage2 = float(getattr(agent_cfg, "action_std_stage2_value", 0.5))
    std_late_rate_limit_enable = bool(getattr(agent_cfg, "action_std_late_rate_limit_enable", True))
    std_late_rate_limit_start_iter = int(getattr(agent_cfg, "action_std_late_rate_limit_start_iter", 2200))
    std_late_rate_limit_segment_iters = max(int(getattr(agent_cfg, "action_std_late_rate_limit_segment_iters", 20)), 1)
    std_late_max_up_factor = max(float(getattr(agent_cfg, "action_std_late_max_up_factor", 1.02)), 1.0)
    std_late_max_down_factor = max(float(getattr(agent_cfg, "action_std_late_max_down_factor", 1.00)), 1.0)
    std_late_max_up_log = math.log(std_late_max_up_factor) if std_late_max_up_factor > 0.0 else 0.0
    std_late_max_down_log = math.log(std_late_max_down_factor) if std_late_max_down_factor > 0.0 else 0.0

    start_iter = int(getattr(runner, "current_learning_iteration", 0))
    final_iter = start_iter + total_iters

    boundaries = {start_iter, final_iter}
    if entropy_enable:
        if start_iter < entropy_stage1_end < final_iter:
            boundaries.add(entropy_stage1_end)
        if start_iter < entropy_stage2_end < final_iter:
            boundaries.add(entropy_stage2_end)
    if std_enable:
        if start_iter < std_stage1_end < final_iter:
            boundaries.add(std_stage1_end)
        if start_iter < std_stage2_end < final_iter:
            boundaries.add(std_stage2_end)
        # Late phase segmentation to enable per-segment std limiter updates.
        if std_late_rate_limit_enable:
            seg_anchor = max(start_iter, std_late_rate_limit_start_iter)
            if start_iter < std_late_rate_limit_start_iter < final_iter:
                boundaries.add(std_late_rate_limit_start_iter)
                seg_anchor = std_late_rate_limit_start_iter
            if seg_anchor < final_iter:
                b = seg_anchor + std_late_rate_limit_segment_iters
                while b < final_iter:
                    boundaries.add(b)
                    b += std_late_rate_limit_segment_iters
    focus_enable = focus_ramp_cfg is not None
    if focus_enable:
        fr_start = int(focus_ramp_cfg["start_iter"])
        fr_end = int(focus_ramp_cfg["end_iter"])
        fr_seg = max(int(focus_ramp_cfg["segment_iters"]), 1)
        if start_iter < fr_start < final_iter:
            boundaries.add(fr_start)
        if start_iter < fr_end < final_iter:
            boundaries.add(fr_end)
        seg_anchor = max(start_iter, fr_start)
        if seg_anchor < min(fr_end, final_iter):
            b = seg_anchor + fr_seg
            while b < min(fr_end, final_iter):
                boundaries.add(b)
                b += fr_seg
    phase_bounds = sorted(boundaries)

    first_phase = True
    std_prev_applied_log: torch.Tensor | None = None
    std_late_rate_initialized = False
    for phase_idx in range(len(phase_bounds) - 1):
        seg_start = int(phase_bounds[phase_idx])
        seg_end = int(phase_bounds[phase_idx + 1])
        seg_iters = seg_end - seg_start
        if seg_iters <= 0:
            continue

        # Avoid duplicate "boundary iteration" when calling runner.learn() multiple times.
        runner.current_learning_iteration = seg_start

        entropy_value = None
        entropy_set = False
        if entropy_enable:
            entropy_value = _piecewise_stage_value(
                iter_idx=seg_start,
                stage1_end_iter=entropy_stage1_end,
                stage2_end_iter=entropy_stage2_end,
                stage0_value=entropy_stage0,
                stage1_value=entropy_stage1,
                stage2_value=entropy_stage2,
            )
            entropy_set = _set_runner_entropy_coef(runner, entropy_value)

        std_target_value = None
        std_set = False
        std_current_mean = None
        std_prev_mean = None
        std_applied_mean = None
        std_rate_limit_on = False
        if std_enable:
            std_target_value = _piecewise_stage_value(
                iter_idx=seg_start,
                stage1_end_iter=std_stage1_end,
                stage2_end_iter=std_stage2_end,
                stage0_value=std_stage0,
                stage1_value=std_stage1,
                stage2_value=std_stage2,
            )
            std_rate_limit_on = bool(
                std_late_rate_limit_enable
                and seg_start >= std_late_rate_limit_start_iter
                and (std_late_max_up_log > 0.0 or std_late_max_down_log > 0.0)
            )
            prev_for_call = std_prev_applied_log
            # Initialize limiter without shock at first activation.
            if std_rate_limit_on and not std_late_rate_initialized:
                prev_for_call = None
            std_set, std_target_value, std_current_mean, std_prev_mean, std_applied_mean, std_prev_applied_log = _set_runner_action_std_cap_only(
                runner,
                std_target_value,
                prev_applied_log=prev_for_call,
                enable_rate_limit=std_rate_limit_on,
                max_up_log=std_late_max_up_log,
                max_down_log=std_late_max_down_log,
            )
            if std_set and std_rate_limit_on:
                std_late_rate_initialized = True

        focus_value = None
        focus_set = False
        if focus_enable:
            focus_value = _piecewise_linear_value(
                iter_idx=seg_start,
                start_iter=int(focus_ramp_cfg["start_iter"]),
                end_iter=int(focus_ramp_cfg["end_iter"]),
                start_value=float(focus_ramp_cfg["start_prob"]),
                end_value=float(focus_ramp_cfg["end_prob"]),
            )
            focus_set = _set_env_fault_focus_prob(runner, focus_value)

        if debug_startup:
            print(
                "[DBG] Exploration schedule phase "
                f"{phase_idx + 1}/{len(phase_bounds) - 1}: iter[{seg_start},{seg_end}) "
                f"entropy={entropy_value if entropy_set else 'N/A'} "
                f"action_std_target={std_target_value if std_set else 'N/A'} "
                f"action_std_current={std_current_mean if std_set else 'N/A'} "
                f"action_std_prev={std_prev_mean if std_set and std_prev_mean is not None else 'N/A'} "
                f"action_std_applied={std_applied_mean if std_set else 'N/A'} "
                f"action_std_rate_limit={'ON' if std_rate_limit_on else 'OFF'} "
                f"fault_focus_prob={focus_value if focus_set else 'N/A'}",
                flush=True,
            )
        else:
            logging.info(
                "[TrainSchedule] phase=%d/%d iter=[%d,%d) entropy=%s action_std_target=%s action_std_current=%s action_std_prev=%s action_std_applied=%s action_std_rate_limit=%s fault_focus_prob=%s",
                phase_idx + 1,
                len(phase_bounds) - 1,
                seg_start,
                seg_end,
                f"{entropy_value:.4f}" if entropy_set and entropy_value is not None else "N/A",
                f"{std_target_value:.4f}" if std_set and std_target_value is not None else "N/A",
                f"{std_current_mean:.4f}" if std_set and std_current_mean is not None else "N/A",
                f"{std_prev_mean:.4f}" if std_set and std_prev_mean is not None else "N/A",
                f"{std_applied_mean:.4f}" if std_set and std_applied_mean is not None else "N/A",
                "ON" if std_rate_limit_on else "OFF",
                f"{focus_value:.4f}" if focus_set and focus_value is not None else "N/A",
            )

        runner.learn(num_learning_iterations=seg_iters, init_at_random_ep_len=first_phase)
        first_phase = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    debug_startup = bool(args_cli.debug_startup)
    _configure_torch_runtime(bool(args_cli.perf_mode))
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.num_steps_per_env is not None:
        if int(args_cli.num_steps_per_env) <= 0:
            raise ValueError(f"--num_steps_per_env must be > 0, got {args_cli.num_steps_per_env}")
        agent_cfg.num_steps_per_env = int(args_cli.num_steps_per_env)
    if args_cli.num_mini_batches is not None:
        if int(args_cli.num_mini_batches) <= 0:
            raise ValueError(f"--num_mini_batches must be > 0, got {args_cli.num_mini_batches}")
        agent_cfg.algorithm.num_mini_batches = int(args_cli.num_mini_batches)
    if args_cli.num_learning_epochs is not None:
        if int(args_cli.num_learning_epochs) <= 0:
            raise ValueError(f"--num_learning_epochs must be > 0, got {args_cli.num_learning_epochs}")
        agent_cfg.algorithm.num_learning_epochs = int(args_cli.num_learning_epochs)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if env_cfg.seed is not None:
        random.seed(int(env_cfg.seed))
        np.random.seed(int(env_cfg.seed))
        torch.manual_seed(int(env_cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(env_cfg.seed))
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    train_fault_mode = str(args_cli.train_fault_mode).strip().lower()
    train_fault_motor_id = int(args_cli.train_fault_motor_id)
    if train_fault_mode == "single_motor_fixed" and train_fault_motor_id < 0:
        raise ValueError(
            "--train_fault_mode=single_motor_fixed requires --train_fault_motor_id in [0..11]. "
            "Refusing silent fallback to random fault mode."
        )
    if train_fault_motor_id >= 0 and not (0 <= train_fault_motor_id < 12):
        raise ValueError(
            f"Invalid --train_fault_motor_id={train_fault_motor_id} (expected 0..11)."
        )

    if train_fault_motor_id >= 0:
        setattr(env_cfg, "phm_fault_injection_mode", "single_motor_fixed")
        setattr(env_cfg, "phm_fault_fixed_motor_id", train_fault_motor_id)
        logger.info("[Train] Overriding fixed fault motor: id=%d (single_motor_fixed)", train_fault_motor_id)
    elif train_fault_mode != "default":
        setattr(env_cfg, "phm_fault_injection_mode", train_fault_mode)
        logger.info("[Train] Overriding phm_fault_injection_mode=%s", train_fault_mode)
    if args_cli.train_fault_pair_uniform is not None:
        setattr(env_cfg, "phm_fault_pair_uniform_enable", bool(args_cli.train_fault_pair_uniform))
        logger.info(
            "[Train] Overriding phm_fault_pair_uniform_enable=%s",
            bool(args_cli.train_fault_pair_uniform),
        )
    if args_cli.train_fault_hold_steps is not None:
        if int(args_cli.train_fault_hold_steps) < 0:
            raise ValueError(
                f"Invalid --train_fault_hold_steps={args_cli.train_fault_hold_steps} (expected >= 0)."
            )
        setattr(env_cfg, "phm_fault_hold_steps", int(args_cli.train_fault_hold_steps))
        logger.info("[Train] Overriding phm_fault_hold_steps=%d", int(args_cli.train_fault_hold_steps))
    if args_cli.train_fault_focus_prob is not None:
        focus_prob = float(args_cli.train_fault_focus_prob)
        if not (0.0 <= focus_prob <= 1.0):
            raise ValueError(
                f"Invalid --train_fault_focus_prob={focus_prob} (expected in [0.0, 1.0])."
            )
        setattr(env_cfg, "phm_fault_focus_prob", focus_prob)
        logger.info("[Train] Overriding phm_fault_focus_prob=%.4f", focus_prob)
    if args_cli.train_fault_pair_weighted_enable is not None:
        setattr(env_cfg, "phm_fault_pair_weighted_enable", bool(args_cli.train_fault_pair_weighted_enable))
        logger.info(
            "[Train] Overriding phm_fault_pair_weighted_enable=%s",
            bool(args_cli.train_fault_pair_weighted_enable),
        )
    if args_cli.train_fault_pair_prob_floor is not None:
        pair_floor = float(args_cli.train_fault_pair_prob_floor)
        if not (0.0 <= pair_floor <= 1.0):
            raise ValueError(
                f"Invalid --train_fault_pair_prob_floor={pair_floor} (expected in [0.0, 1.0])."
            )
        setattr(env_cfg, "phm_fault_pair_prob_floor", pair_floor)
        logger.info("[Train] Overriding phm_fault_pair_prob_floor=%.4f", pair_floor)
    if args_cli.train_fault_pair_prob_cap is not None:
        pair_cap = float(args_cli.train_fault_pair_prob_cap)
        if not (0.0 <= pair_cap <= 1.0):
            raise ValueError(
                f"Invalid --train_fault_pair_prob_cap={pair_cap} (expected in [0.0, 1.0])."
            )
        setattr(env_cfg, "phm_fault_pair_prob_cap", pair_cap)
        logger.info("[Train] Overriding phm_fault_pair_prob_cap=%.4f", pair_cap)
    if args_cli.train_fault_pair_target_weights is not None:
        pair_target_weights = _parse_pair_target_weights(args_cli.train_fault_pair_target_weights, num_pairs=6)
        setattr(env_cfg, "phm_fault_pair_target_weights", pair_target_weights)
        logger.info("[Train] Overriding phm_fault_pair_target_weights=%s", list(pair_target_weights))
    if args_cli.train_fault_pair_adaptive_enable is not None:
        setattr(env_cfg, "phm_fault_pair_adaptive_enable", bool(args_cli.train_fault_pair_adaptive_enable))
        logger.info(
            "[Train] Overriding phm_fault_pair_adaptive_enable=%s",
            bool(args_cli.train_fault_pair_adaptive_enable),
        )
    if args_cli.train_fault_pair_adaptive_mix is not None:
        adaptive_mix = float(args_cli.train_fault_pair_adaptive_mix)
        if not (0.0 <= adaptive_mix <= 1.0):
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_mix={adaptive_mix} (expected in [0.0, 1.0])."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_mix", adaptive_mix)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_mix=%.4f", adaptive_mix)
    if args_cli.train_fault_pair_adaptive_beta is not None:
        adaptive_beta = float(args_cli.train_fault_pair_adaptive_beta)
        if adaptive_beta < 0.0:
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_beta={adaptive_beta} (expected >= 0.0)."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_beta", adaptive_beta)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_beta=%.4f", adaptive_beta)
    if args_cli.train_fault_pair_adaptive_ema is not None:
        adaptive_ema = float(args_cli.train_fault_pair_adaptive_ema)
        if not (0.0 <= adaptive_ema <= 1.0):
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_ema={adaptive_ema} (expected in [0.0, 1.0])."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_ema", adaptive_ema)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_ema=%.4f", adaptive_ema)
    if args_cli.train_fault_pair_adaptive_min_episode_per_pair is not None:
        adaptive_min_ep = float(args_cli.train_fault_pair_adaptive_min_episode_per_pair)
        if adaptive_min_ep < 1.0:
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_min_episode_per_pair={adaptive_min_ep} (expected >= 1.0)."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_min_episode_per_pair", adaptive_min_ep)
        logger.info(
            "[Train] Overriding phm_fault_pair_adaptive_min_episode_per_pair=%.4f",
            adaptive_min_ep,
        )
    if args_cli.train_fault_pair_adaptive_w_fail is not None:
        w_fail = float(args_cli.train_fault_pair_adaptive_w_fail)
        if w_fail < 0.0:
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_w_fail={w_fail} (expected >= 0.0)."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_w_fail", w_fail)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_w_fail=%.4f", w_fail)
    if args_cli.train_fault_pair_adaptive_w_sat is not None:
        w_sat = float(args_cli.train_fault_pair_adaptive_w_sat)
        if w_sat < 0.0:
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_w_sat={w_sat} (expected >= 0.0)."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_w_sat", w_sat)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_w_sat=%.4f", w_sat)
    if args_cli.train_fault_pair_adaptive_w_latch is not None:
        w_latch = float(args_cli.train_fault_pair_adaptive_w_latch)
        if w_latch < 0.0:
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_w_latch={w_latch} (expected >= 0.0)."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_w_latch", w_latch)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_w_latch=%.4f", w_latch)
    if args_cli.train_fault_pair_adaptive_sat_scale is not None:
        sat_scale = float(args_cli.train_fault_pair_adaptive_sat_scale)
        if sat_scale <= 0.0:
            raise ValueError(
                f"Invalid --train_fault_pair_adaptive_sat_scale={sat_scale} (expected > 0.0)."
            )
        setattr(env_cfg, "phm_fault_pair_adaptive_sat_scale", sat_scale)
        logger.info("[Train] Overriding phm_fault_pair_adaptive_sat_scale=%.4f", sat_scale)
    adaptive_hint_used = any(
        x is not None
        for x in (
            args_cli.train_fault_pair_adaptive_mix,
            args_cli.train_fault_pair_adaptive_beta,
            args_cli.train_fault_pair_adaptive_ema,
            args_cli.train_fault_pair_adaptive_min_episode_per_pair,
            args_cli.train_fault_pair_adaptive_w_fail,
            args_cli.train_fault_pair_adaptive_w_sat,
            args_cli.train_fault_pair_adaptive_w_latch,
            args_cli.train_fault_pair_adaptive_sat_scale,
        )
    ) or bool(args_cli.train_fault_pair_adaptive_enable is True)
    if adaptive_hint_used and args_cli.train_fault_pair_adaptive_enable is None:
        setattr(env_cfg, "phm_fault_pair_adaptive_enable", True)
        logger.info(
            "[Train] Auto-enabling phm_fault_pair_adaptive_enable=True "
            "(adaptive overrides were provided)."
        )
    if args_cli.train_fault_motor_adaptive_enable is not None:
        setattr(env_cfg, "phm_fault_motor_adaptive_enable", bool(args_cli.train_fault_motor_adaptive_enable))
        logger.info(
            "[Train] Overriding phm_fault_motor_adaptive_enable=%s",
            bool(args_cli.train_fault_motor_adaptive_enable),
        )
    if args_cli.train_fault_motor_adaptive_topk is not None:
        motor_topk = int(args_cli.train_fault_motor_adaptive_topk)
        if motor_topk < 1:
            raise ValueError(
                f"Invalid --train_fault_motor_adaptive_topk={motor_topk} (expected >= 1)."
            )
        setattr(env_cfg, "phm_fault_motor_adaptive_topk", motor_topk)
        logger.info("[Train] Overriding phm_fault_motor_adaptive_topk=%d", motor_topk)
    if args_cli.train_fault_motor_adaptive_min_episode_per_motor is not None:
        motor_min_ep = float(args_cli.train_fault_motor_adaptive_min_episode_per_motor)
        if motor_min_ep < 1.0:
            raise ValueError(
                f"Invalid --train_fault_motor_adaptive_min_episode_per_motor={motor_min_ep} "
                "(expected >= 1.0)."
            )
        setattr(env_cfg, "phm_fault_motor_adaptive_min_episode_per_motor", motor_min_ep)
        logger.info(
            "[Train] Overriding phm_fault_motor_adaptive_min_episode_per_motor=%.4f",
            motor_min_ep,
        )
    motor_adaptive_hint_used = any(
        x is not None
        for x in (
            args_cli.train_fault_motor_adaptive_topk,
            args_cli.train_fault_motor_adaptive_min_episode_per_motor,
        )
    ) or bool(args_cli.train_fault_motor_adaptive_enable is True)
    if motor_adaptive_hint_used and args_cli.train_fault_motor_adaptive_enable is None:
        setattr(env_cfg, "phm_fault_motor_adaptive_enable", True)
        logger.info(
            "[Train] Auto-enabling phm_fault_motor_adaptive_enable=True "
            "(motor-adaptive overrides were provided)."
        )

    pair_weighted_hint_used = any(
        x is not None
        for x in (
            args_cli.train_fault_pair_prob_floor,
            args_cli.train_fault_pair_prob_cap,
            args_cli.train_fault_pair_target_weights,
        )
    ) or adaptive_hint_used
    if pair_weighted_hint_used and args_cli.train_fault_pair_weighted_enable is None:
        setattr(env_cfg, "phm_fault_pair_weighted_enable", True)
        logger.info(
            "[Train] Auto-enabling phm_fault_pair_weighted_enable=True "
            "(pair/adaptive overrides were provided)."
        )
    focus_ramp_cfg = _resolve_focus_ramp_cfg(args_cli)
    focus_prob_effective = float(getattr(env_cfg, "phm_fault_focus_prob", 0.0))
    focus_enabled = bool(focus_prob_effective > 0.0)
    if focus_ramp_cfg is not None:
        focus_enabled = focus_enabled or bool(
            max(float(focus_ramp_cfg["start_prob"]), float(focus_ramp_cfg["end_prob"])) > 0.0
        )

    focus_pairs_override: tuple[tuple[int, int], ...] | None = None
    if args_cli.train_fault_focus_pairs is not None:
        focus_pairs_override = _parse_focus_pairs(args_cli.train_fault_focus_pairs)
    elif (
        args_cli.train_fault_focus_motor_ids is None
        and args_cli.train_fault_focus_pairs is None
        and not bool(getattr(env_cfg, "phm_fault_motor_adaptive_enable", False))
    ):
        # Safe default: use all mirror pairs unless the user supplied selectors
        # or enabled motor-adaptive focus instead.
        focus_pairs_override = _FOCUS_ALL_MIRROR_PAIRS_12
        logger.info(
            "[Train] Auto default focus pairs enabled: %s",
            list(focus_pairs_override),
        )

    if focus_pairs_override is not None:
        setattr(env_cfg, "phm_fault_focus_pairs", focus_pairs_override)
        logger.info("[Train] Overriding phm_fault_focus_pairs=%s", list(focus_pairs_override))
    if args_cli.train_fault_focus_motor_ids is not None:
        focus_motor_ids = _parse_focus_motor_ids(args_cli.train_fault_focus_motor_ids)
        setattr(env_cfg, "phm_fault_focus_motor_ids", focus_motor_ids)
        logger.info("[Train] Overriding phm_fault_focus_motor_ids=%s", list(focus_motor_ids))
    elif focus_pairs_override is not None:
        # For visibility in logs/metadata, mirror pair selection into flattened motor-id list.
        flattened: list[int] = []
        for a, b in focus_pairs_override:
            flattened.append(int(a))
            flattened.append(int(b))
        auto_ids = tuple(dict.fromkeys(flattened))
        setattr(env_cfg, "phm_fault_focus_motor_ids", auto_ids)
        logger.info(
            "[Train] Auto-derived phm_fault_focus_motor_ids from pairs=%s",
            list(auto_ids),
        )
    if focus_ramp_cfg is not None:
        if args_cli.train_fault_mode == "single_motor_fixed" or train_fault_motor_id >= 0:
            logger.warning(
                "[Train] Focus ramp is configured but fault mode is fixed-motor; "
                "ramp effect will be minimal/non-applicable."
            )
        if args_cli.train_fault_focus_prob is not None:
            logger.info(
                "[Train] Focus ramp is enabled and will override constant phm_fault_focus_prob during training."
            )
        # Set initial cfg value for transparency in dumped env.yaml and first reset.
        setattr(env_cfg, "phm_fault_focus_prob", float(focus_ramp_cfg["start_prob"]))
        logger.info(
            "[Train] Focus ramp configured: iter %d -> %d, prob %.4f -> %.4f, segment=%d",
            int(focus_ramp_cfg["start_iter"]),
            int(focus_ramp_cfg["end_iter"]),
            float(focus_ramp_cfg["start_prob"]),
            float(focus_ramp_cfg["end_prob"]),
            int(focus_ramp_cfg["segment_iters"]),
        )
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    if debug_startup:
        print(
            f"[DBG] Creating gym env at {datetime.now().isoformat()} with num_envs={env_cfg.scene.num_envs} on device={env_cfg.sim.device}",
            flush=True,
        )
    _t_env = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if debug_startup:
        print(f"[DBG] gym env created in {time.time() - _t_env:.2f}s", flush=True)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    if debug_startup:
        print(f"[DBG] Wrapping env for rsl-rl at {datetime.now().isoformat()}", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if debug_startup:
        print(f"[DBG] Env wrapped at {datetime.now().isoformat()}", flush=True)

    # warmup: helps pinpoint hangs (reset vs first step) and may trigger initial JIT/graph compilation
    if debug_startup:
        print(f"[DBG] Warmup: calling env.reset() at {datetime.now().isoformat()}", flush=True)
        _t_reset = time.time()
        _ = env.reset()
        print(f"[DBG] Warmup: env.reset() done in {time.time() - _t_reset:.2f}s", flush=True)

        try:
            action_dim = int(getattr(env, "num_actions"))
        except Exception:
            action_dim = int(env.action_space.shape[0])
        warmup_actions = torch.zeros((env.num_envs, action_dim), device=getattr(env, "device", agent_cfg.device))
        print(
            f"[DBG] Warmup: calling env.step() at {datetime.now().isoformat()} action_shape={tuple(warmup_actions.shape)}",
            flush=True,
        )
        _t_step = time.time()
        _ = env.step(warmup_actions)
        print(f"[DBG] Warmup: env.step() done in {time.time() - _t_step:.2f}s", flush=True)

    # create runner from rsl-rl
    if debug_startup:
        print(f"[DBG] Creating runner at {datetime.now().isoformat()} (runner={agent_cfg.class_name})", flush=True)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if debug_startup:
        print(f"[DBG] Runner created at {datetime.now().isoformat()}", flush=True)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    if debug_startup:
        print(
            f"[DBG] Starting runner.learn at {datetime.now().isoformat()} max_iterations={agent_cfg.max_iterations}",
            flush=True,
        )
    if agent_cfg.class_name == "OnPolicyRunner":
        _learn_with_exploration_schedule(
            runner,
            agent_cfg,
            focus_ramp_cfg=focus_ramp_cfg,
            debug_startup=debug_startup,
        )
    else:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
