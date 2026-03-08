# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL checkpoint for visualization, video capture, or debug."""

"""Playback entrypoint.

AppLauncher must be initialized before importing Isaac task modules.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a trained RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_flat_folder",
    type=str,
    default="",
    help="If set, save all video files into this single folder instead of per-tag subfolders.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--motor_overload_vis",
    action="store_true",
    default=False,
    help="Visualize per-leg motor overload (green/yellow/orange/red markers).",
)
parser.add_argument(
    "--motor_overload_print_hz",
    type=float,
    default=0.0,
    help="If > 0, print per-leg overload score at this frequency (Hz).",
)
parser.add_argument(
    "--fault_leg_color_vis",
    action="store_true",
    default=False,
    help="Color currently faulted leg links in red (env_0 only, based on PHM fault state).",
)
parser.add_argument(
    "--fault_motor_color_vis",
    action="store_true",
    default=False,
    help="Color currently faulted motor link in red (env_0 only, based on PHM fault state).",
)
parser.add_argument(
    "--force_fault_scenario",
    type=str,
    default="none",
    choices=["none", "fresh", "used", "aged", "critical"],
    help="Force PHM reset scenario during play for visualization/debug (default: none).",
)
parser.add_argument(
    "--force_fault_motor_id",
    type=int,
    default=-1,
    help="If set to [0..11], play uses single_motor_fixed at this motor index.",
)
parser.add_argument(
    "--force_walk_command",
    action="store_true",
    default=False,
    help="Force base velocity command to a fixed walking command during play (disables standing command sampling).",
)
parser.add_argument(
    "--play_cmd_lin_x",
    type=float,
    default=0.6,
    help="Fixed linear x velocity used when --force_walk_command is enabled.",
)
parser.add_argument(
    "--play_cmd_lin_y",
    type=float,
    default=0.0,
    help="Fixed linear y velocity used when --force_walk_command is enabled.",
)
parser.add_argument(
    "--play_cmd_ang_z",
    type=float,
    default=0.0,
    help="Fixed yaw velocity used when --force_walk_command is enabled.",
)
parser.add_argument(
    "--play_cmd_resample_s",
    type=float,
    default=10.0,
    help="Command resampling period used when --force_walk_command is enabled.",
)
parser.add_argument(
    "--follow_camera",
    action="store_true",
    default=False,
    help="Continuously track env_0 robot root with a follower camera during playback/video.",
)
parser.add_argument(
    "--follow_cam_offset_x",
    type=float,
    default=-2.2,
    help="Follower camera local offset X (m) in robot base frame.",
)
parser.add_argument(
    "--follow_cam_offset_y",
    type=float,
    default=0.0,
    help="Follower camera local offset Y (m) in robot base frame.",
)
parser.add_argument(
    "--follow_cam_offset_z",
    type=float,
    default=1.1,
    help="Follower camera local offset Z (m) in robot base frame.",
)
parser.add_argument(
    "--follow_cam_lookat_z",
    type=float,
    default=0.35,
    help="Follower camera look-at height offset above robot root (m).",
)
parser.add_argument(
    "--follow_cam_use_yaw_only",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Use only robot yaw for follower offset rotation (reduces pitch/roll camera jitter).",
)
parser.add_argument(
    "--follow_cam_smooth_alpha",
    type=float,
    default=1.0,
    help="EMA smoothing alpha for follower camera in (0,1]. Lower is smoother.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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

"""Isaac task imports follow AppLauncher initialization."""

import gymnasium as gym
import os
import time
import torch

import isaaclab.sim as sim_utils
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
try:
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ImportError:
    get_published_pretrained_checkpoint = None

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_go2_phm.tasks  # noqa: F401


def _obs_from_reset_output(reset_output):
    """Normalize Gymnasium reset output across API versions."""
    if isinstance(reset_output, tuple):
        return reset_output[0]
    return reset_output


def _safe_reset_recurrent(policy_nn, dones: torch.Tensor):
    """Reset recurrent state only when the policy exposes a reset method."""
    if hasattr(policy_nn, "reset"):
        policy_nn.reset(dones)


def _make_motor_overload_markers() -> VisualizationMarkers:
    """Create color-coded marker prototypes for leg overload visualization."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PHM/motor_overload",
        markers={
            "ok": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.85, 0.15),
                    emissive_color=(0.0, 0.2, 0.0),
                ),
            ),
            "warn": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.85, 0.1),
                    emissive_color=(0.2, 0.2, 0.0),
                ),
            ),
            "hot": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.45, 0.05),
                    emissive_color=(0.25, 0.08, 0.0),
                ),
            ),
            "critical": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.05, 0.05),
                    emissive_color=(0.25, 0.0, 0.0),
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _resolve_leg_body_indices(robot) -> dict[int, list[int]]:
    """Resolve env_0 body indices grouped by leg index (FR=0, FL=1, RR=2, RL=3)."""
    leg_prefixes = ("FR_", "FL_", "RR_", "RL_")
    leg_indices = {0: [], 1: [], 2: [], 3: []}
    body_names = list(robot.body_names)
    for body_idx, body_name in enumerate(body_names):
        for leg_idx, prefix in enumerate(leg_prefixes):
            if body_name.startswith(prefix):
                leg_indices[leg_idx].append(body_idx)
                break
    return leg_indices


def _resolve_motor_body_indices(robot) -> dict[int, int]:
    """Resolve env_0 body index per motor index [0..11]. Missing motors map to -1."""
    motor_names = (
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    )
    motor_indices: dict[int, int] = {i: -1 for i in range(12)}
    body_names = list(robot.body_names)
    for body_idx, body_name in enumerate(body_names):
        for motor_idx, prefix in enumerate(motor_names):
            if motor_indices[motor_idx] >= 0:
                continue
            if body_name.startswith(prefix):
                motor_indices[motor_idx] = body_idx
                break
    return motor_indices


def _make_fault_overlay_marker(prim_path: str, radius: float) -> VisualizationMarkers:
    """Create red overlay marker used for fault highlighting without touching robot materials."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "fault": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.95, 0.08, 0.08),
                    emissive_color=(0.28, 0.0, 0.0),
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


def _fault_legs_from_phm(base_env) -> torch.Tensor:
    """Read PHM fault state from env_0 and return FR/FL/RR/RL active flags (bool, CPU)."""
    out = torch.zeros((4,), dtype=torch.bool)
    phm = base_env.phm_state

    if hasattr(phm, "fault_mask"):
        mask = phm.fault_mask[0, :12].detach().cpu()
        out[0] = bool(torch.any(mask[0:3] > 0.5).item())
        out[1] = bool(torch.any(mask[3:6] > 0.5).item())
        out[2] = bool(torch.any(mask[6:9] > 0.5).item())
        out[3] = bool(torch.any(mask[9:12] > 0.5).item())
        return out

    if hasattr(phm, "fault_motor_id"):
        motor_id = int(phm.fault_motor_id[0].item())
        if 0 <= motor_id < 12:
            out[motor_id // 3] = True
    return out


def _fault_motor_from_phm(base_env) -> int:
    """Read PHM fault motor index from env_0. Returns -1 if unavailable."""
    phm = base_env.phm_state
    if hasattr(phm, "fault_motor_id"):
        motor_id = int(phm.fault_motor_id[0].item())
        if 0 <= motor_id < 12:
            return motor_id
    if hasattr(phm, "fault_mask"):
        mask = phm.fault_mask[0, :12].detach().cpu()
        active = torch.nonzero(mask > 0.5, as_tuple=False)
        if active.numel() > 0:
            return int(active[0].item())
    return -1


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Playback readability:
    # - Disable command debug arrows to avoid noisy point-instancer warnings in captured videos.
    # - Keep behavior deterministic/cleaner for human inspection.
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        cmd_cfg = env_cfg.commands.base_velocity
        if hasattr(cmd_cfg, "debug_vis"):
            cmd_cfg.debug_vis = False

    # Known issue in some Isaac/driver combinations:
    # articulation visual updates can look desynchronized with Fabric enabled.
    # Recommend disabling Fabric for trustworthy playback videos.
    if not bool(args_cli.disable_fabric):
        print(
            "[WARNING] Fabric is enabled. If you observe base/leg visual desync in play videos, "
            "re-run with --disable_fabric."
        )

    # Optional PHM forcing for playback/debug:
    # default play starts near step 0, where curriculum is often fresh-dominant.
    force_scenario = str(args_cli.force_fault_scenario).strip().lower()
    if force_scenario != "none":
        setattr(env_cfg, "phm_force_scenario_label", force_scenario)
        print(f"[Play] force_fault_scenario enabled: {force_scenario}")
    if int(args_cli.force_fault_motor_id) >= 0:
        motor_id = int(args_cli.force_fault_motor_id)
        if 0 <= motor_id < 12:
            setattr(env_cfg, "phm_fault_injection_mode", "single_motor_fixed")
            setattr(env_cfg, "phm_fault_fixed_motor_id", motor_id)
            print(f"[Play] force_fault_motor_id enabled: {motor_id}")
        else:
            print(f"[WARNING] --force_fault_motor_id must be in [0, 11], got {motor_id}. Ignoring.")
    if bool(args_cli.force_walk_command):
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
            cmd_cfg = env_cfg.commands.base_velocity
            cmd_cfg.rel_standing_envs = 0.0
            cmd_cfg.resampling_time_range = (
                float(args_cli.play_cmd_resample_s),
                float(args_cli.play_cmd_resample_s),
            )
            if hasattr(cmd_cfg, "ranges"):
                cmd_cfg.ranges.lin_vel_x = (
                    float(args_cli.play_cmd_lin_x),
                    float(args_cli.play_cmd_lin_x),
                )
                cmd_cfg.ranges.lin_vel_y = (
                    float(args_cli.play_cmd_lin_y),
                    float(args_cli.play_cmd_lin_y),
                )
                cmd_cfg.ranges.ang_vel_z = (
                    float(args_cli.play_cmd_ang_z),
                    float(args_cli.play_cmd_ang_z),
                )
            print(
                "[Play] force_walk_command enabled: "
                f"vx={float(args_cli.play_cmd_lin_x):.3f}, "
                f"vy={float(args_cli.play_cmd_lin_y):.3f}, "
                f"wz={float(args_cli.play_cmd_ang_z):.3f}, "
                f"resample={float(args_cli.play_cmd_resample_s):.2f}s, "
                "rel_standing_envs=0.0"
            )
        else:
            print("[WARNING] --force_walk_command requested but base_velocity command cfg not found.")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        if get_published_pretrained_checkpoint is None:
            print("[WARNING] Pretrained checkpoint loading is unavailable in this isaaclab_rl installation.")
            return
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        try:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        except FileNotFoundError:
            try:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, args_cli.checkpoint)
            except FileNotFoundError:
                logs_root = os.path.abspath(os.path.join("logs", "rsl_rl"))
                found = None
                if os.path.isdir(logs_root) and agent_cfg.load_run is not None:
                    for exp_name in os.listdir(logs_root):
                        exp_dir = os.path.join(logs_root, exp_name)
                        if not os.path.isdir(exp_dir):
                            continue
                        candidate = os.path.join(exp_dir, agent_cfg.load_run, args_cli.checkpoint)
                        if os.path.isfile(candidate):
                            found = candidate
                            break
                if found is None:
                    raise
                resume_path = found
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        # Tag playback videos with forced PHM settings for easy later comparison.
        scenario_tag = force_scenario if force_scenario != "none" else "from_env"
        motor_tag = (
            f"m{int(args_cli.force_fault_motor_id)}"
            if 0 <= int(args_cli.force_fault_motor_id) < 12
            else "mrand"
        )
        ckpt_name = os.path.splitext(os.path.basename(resume_path))[0]
        ckpt_tag = ckpt_name if ckpt_name else "ckpt"

        def _num_tag(x: float) -> str:
            sign = "p" if float(x) >= 0.0 else "n"
            mag = abs(float(x))
            return f"{sign}{mag:.2f}".replace(".", "p")

        if bool(args_cli.force_walk_command):
            cmd_tag = (
                f"vx{_num_tag(args_cli.play_cmd_lin_x)}_"
                f"vy{_num_tag(args_cli.play_cmd_lin_y)}_"
                f"wz{_num_tag(args_cli.play_cmd_ang_z)}"
            )
        else:
            cmd_tag = "cmd_auto"

        video_tag = f"{scenario_tag}_{motor_tag}_{cmd_tag}_{ckpt_tag}"
        flat_folder = str(args_cli.video_flat_folder).strip()
        if flat_folder != "":
            video_folder = os.path.abspath(flat_folder)
        else:
            video_folder = os.path.join(log_dir, "videos", "play", video_tag)
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": f"play_{video_tag}",
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # keep direct handle to base manager env for optional PHM visualization
    base_env = env.unwrapped

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = _obs_from_reset_output(env.reset())
    timestep = 0

    follow_camera_enabled = bool(args_cli.follow_camera) and hasattr(base_env, "robot")
    follow_cam_offset = torch.tensor(
        [[float(args_cli.follow_cam_offset_x), float(args_cli.follow_cam_offset_y), float(args_cli.follow_cam_offset_z)]],
        device=base_env.device,
        dtype=torch.float32,
    )
    follow_cam_lookat_z = float(args_cli.follow_cam_lookat_z)
    follow_cam_use_yaw_only = bool(args_cli.follow_cam_use_yaw_only)
    follow_cam_smooth_alpha = float(max(1e-3, min(1.0, args_cli.follow_cam_smooth_alpha)))
    follow_cam_eye_prev: torch.Tensor | None = None
    follow_cam_target_prev: torch.Tensor | None = None
    if bool(args_cli.follow_camera) and not hasattr(base_env.sim, "set_camera_view"):
        print("[WARNING] --follow_camera requested but sim.set_camera_view is unavailable; fallback to static camera.")
        follow_camera_enabled = False

    overload_markers: VisualizationMarkers | None = None
    overload_offsets: torch.Tensor | None = None
    overload_print_every = 0
    fault_leg_vis_enabled = False
    fault_leg_marker: VisualizationMarkers | None = None
    fault_leg_body_indices: dict[int, list[int]] | None = None
    last_fault_legs = torch.zeros((4,), dtype=torch.bool)
    fault_motor_vis_enabled = False
    fault_motor_marker: VisualizationMarkers | None = None
    fault_motor_body_indices: dict[int, int] | None = None
    last_fault_motor = -2
    if bool(args_cli.motor_overload_vis):
        if not hasattr(base_env, "robot") or not hasattr(base_env, "phm_state"):
            print("[WARNING] --motor_overload_vis requires PHM env state; skipping marker visualization.")
        else:
            overload_markers = _make_motor_overload_markers()
            overload_markers.set_visibility(True)
            # FR, FL, RR, RL marker offsets in base frame.
            overload_offsets = torch.tensor(
                [
                    [0.22, -0.14, 0.18],
                    [0.22, 0.14, 0.18],
                    [-0.22, -0.14, 0.18],
                    [-0.22, 0.14, 0.18],
                ],
                device=base_env.device,
                dtype=torch.float32,
            )
            if args_cli.motor_overload_print_hz > 0.0:
                overload_print_every = max(int(round(1.0 / (args_cli.motor_overload_print_hz * dt))), 1)

    if bool(args_cli.fault_leg_color_vis):
        if not hasattr(base_env, "robot") or not hasattr(base_env, "phm_state"):
            print("[WARNING] --fault_leg_color_vis requires PHM env state; skipping fault leg coloring.")
        else:
            # Keep flag name for backward-compat; implementation is overlay marker (no material binding).
            fault_leg_body_indices = _resolve_leg_body_indices(base_env.robot)
            if sum(len(v) for v in fault_leg_body_indices.values()) == 0:
                print("[WARNING] --fault_leg_color_vis: no leg body indices resolved; skipping fault leg overlay.")
                fault_leg_body_indices = None
                fault_leg_vis_enabled = False
            else:
                fault_leg_marker = _make_fault_overlay_marker("/Visuals/PHM/fault_leg_overlay", radius=0.055)
                fault_leg_marker.set_visibility(False)
                fault_leg_vis_enabled = True
                current_fault_legs = _fault_legs_from_phm(base_env)
                last_fault_legs = current_fault_legs.clone()
                leg_names = ("FR", "FL", "RR", "RL")
                active = [leg_names[i] for i in range(4) if bool(current_fault_legs[i].item())]
                if active:
                    print(f"[PHM Vis] Fault leg overlay ON (env_0): {','.join(active)}")
                else:
                    print("[PHM Vis] Fault leg overlay ON (env_0): none")

    if bool(args_cli.fault_motor_color_vis):
        if not hasattr(base_env, "robot") or not hasattr(base_env, "phm_state"):
            print("[WARNING] --fault_motor_color_vis requires PHM env state; skipping fault motor coloring.")
        else:
            # Keep flag name for backward-compat; implementation is overlay marker (no material binding).
            fault_motor_body_indices = _resolve_motor_body_indices(base_env.robot)
            if sum(1 for _, idx in fault_motor_body_indices.items() if int(idx) >= 0) == 0:
                print("[WARNING] --fault_motor_color_vis: no motor body indices resolved; skipping fault motor overlay.")
                fault_motor_body_indices = None
                fault_motor_vis_enabled = False
            else:
                fault_motor_marker = _make_fault_overlay_marker("/Visuals/PHM/fault_motor_overlay", radius=0.040)
                fault_motor_marker.set_visibility(False)
                fault_motor_vis_enabled = True
                current_fault_motor = _fault_motor_from_phm(base_env)
                last_fault_motor = int(current_fault_motor)
                motor_names = (
                    "FR_hip", "FR_thigh", "FR_calf",
                    "FL_hip", "FL_thigh", "FL_calf",
                    "RR_hip", "RR_thigh", "RR_calf",
                    "RL_hip", "RL_thigh", "RL_calf",
                )
                if 0 <= current_fault_motor < 12:
                    print(f"[PHM Vis] Fault motor overlay ON (env_0): {motor_names[current_fault_motor]} (id={current_fault_motor})")
                else:
                    print("[PHM Vis] Fault motor overlay ON (env_0): none")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Keep env.step out of inference_mode to avoid inference-tensor reset issues.
        with torch.no_grad():
            actions = policy(obs)
        actions = actions.clone()
        obs, _, dones, _ = env.step(actions)
        # reset recurrent states for episodes that have terminated
        _safe_reset_recurrent(policy_nn, dones)
        timestep += 1

        if follow_camera_enabled:
            root_pos = base_env.robot.data.root_pos_w[0:1]
            root_quat = base_env.robot.data.root_quat_w[0:1]
            if follow_cam_use_yaw_only:
                # Rotate offset by base yaw only to avoid roll/pitch-induced camera shake.
                w = root_quat[:, 0]
                x = root_quat[:, 1]
                y = root_quat[:, 2]
                z = root_quat[:, 3]
                yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                cy = torch.cos(yaw)
                sy = torch.sin(yaw)
                ox = follow_cam_offset[:, 0]
                oy = follow_cam_offset[:, 1]
                oz = follow_cam_offset[:, 2]
                wx = cy * ox - sy * oy
                wy = sy * ox + cy * oy
                cam_eye_des = root_pos + torch.stack((wx, wy, oz), dim=-1)
            else:
                cam_eye_des = root_pos + quat_apply(root_quat, follow_cam_offset)
            cam_target_des = root_pos.clone()
            cam_target_des[:, 2] += follow_cam_lookat_z

            if follow_cam_smooth_alpha < 0.999:
                if follow_cam_eye_prev is None:
                    follow_cam_eye_prev = cam_eye_des.clone()
                    follow_cam_target_prev = cam_target_des.clone()
                cam_eye = follow_cam_smooth_alpha * cam_eye_des + (1.0 - follow_cam_smooth_alpha) * follow_cam_eye_prev
                cam_target = (
                    follow_cam_smooth_alpha * cam_target_des + (1.0 - follow_cam_smooth_alpha) * follow_cam_target_prev
                )
                follow_cam_eye_prev = cam_eye
                follow_cam_target_prev = cam_target
            else:
                cam_eye = cam_eye_des
                cam_target = cam_target_des
            base_env.sim.set_camera_view(
                eye=cam_eye[0].detach().cpu().tolist(),
                target=cam_target[0].detach().cpu().tolist(),
            )

        if overload_markers is not None and overload_offsets is not None:
            phm = base_env.phm_state
            # Use joint-level stress proxy: max(temp margin use, torque saturation).
            case_temp = phm.motor_case_temp[0, :12]
            temp_score = torch.clamp((case_temp - 60.0) / 10.0, min=0.0, max=1.0)
            torque_sat = torch.clamp(phm.torque_saturation[0, :12], min=0.0, max=1.0)
            joint_score = torch.maximum(temp_score, torque_sat)
            leg_score = torch.stack(
                (
                    torch.max(joint_score[0:3]),   # FR
                    torch.max(joint_score[3:6]),   # FL
                    torch.max(joint_score[6:9]),   # RR
                    torch.max(joint_score[9:12]),  # RL
                ),
                dim=0,
            )

            marker_indices = torch.zeros((4,), dtype=torch.int32, device=base_env.device)
            marker_indices[leg_score >= 0.25] = 1
            marker_indices[leg_score >= 0.50] = 2
            marker_indices[leg_score >= 0.75] = 3

            marker_scale = (0.9 + 0.6 * leg_score).unsqueeze(-1).repeat(1, 3)
            root_pos = base_env.robot.data.root_pos_w[0:1]
            root_quat = base_env.robot.data.root_quat_w[0:1]
            world_offsets = quat_apply(root_quat.repeat(4, 1), overload_offsets)
            marker_pos = root_pos.repeat(4, 1) + world_offsets
            overload_markers.visualize(
                translations=marker_pos,
                scales=marker_scale,
                marker_indices=marker_indices,
            )

            if overload_print_every > 0 and (timestep % overload_print_every == 0):
                print(
                    "[PHM Vis] FR/FL/RR/RL overload = "
                    f"{leg_score[0].item():.2f}, {leg_score[1].item():.2f}, "
                    f"{leg_score[2].item():.2f}, {leg_score[3].item():.2f}"
                )
        if fault_leg_vis_enabled and fault_leg_marker is not None and fault_leg_body_indices is not None:
            current_fault_legs = _fault_legs_from_phm(base_env)
            active_leg_positions: list[torch.Tensor] = []
            for leg_idx in range(4):
                if not bool(current_fault_legs[leg_idx].item()):
                    continue
                body_ids = [i for i in fault_leg_body_indices.get(leg_idx, []) if int(i) >= 0]
                if len(body_ids) == 0:
                    continue
                pos = torch.mean(base_env.robot.data.body_pos_w[0, body_ids, :3], dim=0)
                pos = pos.clone()
                pos[2] += 0.06
                active_leg_positions.append(pos)

            if len(active_leg_positions) > 0:
                marker_pos = torch.stack(active_leg_positions, dim=0)
                marker_scale = torch.ones((marker_pos.shape[0], 3), device=base_env.device, dtype=torch.float32)
                fault_leg_marker.set_visibility(True)
                fault_leg_marker.visualize(translations=marker_pos, scales=marker_scale)
            else:
                fault_leg_marker.set_visibility(False)

            if not torch.equal(current_fault_legs, last_fault_legs):
                last_fault_legs = current_fault_legs.clone()
                leg_names = ("FR", "FL", "RR", "RL")
                active = [leg_names[i] for i in range(4) if bool(current_fault_legs[i].item())]
                if active:
                    print(f"[PHM Vis] Fault leg changed (env_0, overlay): {','.join(active)}")
                else:
                    print("[PHM Vis] Fault leg changed (env_0, overlay): none")
        if (
            fault_motor_vis_enabled
            and fault_motor_body_indices is not None
            and fault_motor_marker is not None
        ):
            current_fault_motor = _fault_motor_from_phm(base_env)
            motor_body_id = int(fault_motor_body_indices.get(int(current_fault_motor), -1))
            if 0 <= int(current_fault_motor) < 12 and motor_body_id >= 0:
                marker_pos = base_env.robot.data.body_pos_w[0, motor_body_id, :3].clone().unsqueeze(0)
                marker_pos[:, 2] += 0.05
                marker_scale = torch.ones((1, 3), device=base_env.device, dtype=torch.float32)
                fault_motor_marker.set_visibility(True)
                fault_motor_marker.visualize(translations=marker_pos, scales=marker_scale)
            else:
                fault_motor_marker.set_visibility(False)
            if int(current_fault_motor) != int(last_fault_motor):
                last_fault_motor = int(current_fault_motor)
                motor_names = (
                    "FR_hip", "FR_thigh", "FR_calf",
                    "FL_hip", "FL_thigh", "FL_calf",
                    "RR_hip", "RR_thigh", "RR_calf",
                    "RL_hip", "RL_thigh", "RL_calf",
                )
                if 0 <= current_fault_motor < 12:
                    print(f"[PHM Vis] Fault motor changed (env_0, overlay): {motor_names[current_fault_motor]} (id={current_fault_motor})")
                else:
                    print("[PHM Vis] Fault motor changed (env_0, overlay): none")
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
