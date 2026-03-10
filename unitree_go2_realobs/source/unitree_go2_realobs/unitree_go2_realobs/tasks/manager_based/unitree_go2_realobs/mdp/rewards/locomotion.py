# =============================================================================
# unitree_go2_realobs/mdp/rewards/locomotion.py
# Locomotion reward helpers for velocity tracking, gait quality, and terrain fit.
# =============================================================================

from __future__ import annotations
import torch
from typing import TYPE_CHECKING, List

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# [Fix #6] Isaac Lab 표준 함수(flat_orientation_l2, action_rate_l2 등)와의
# 이름 충돌 방지. MotorDeg 전용 보상 함수만 wildcard export.
__all__ = [
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_air_time",
    "foot_clearance",
    "feet_slide",
    "feet_air_time_aggregated",
    "feet_slide_aggregated",
    "base_height_l2_terrain_aware",
]

# =============================================================================
# 1. Task Performance
# =============================================================================

def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_xy = asset.data.root_lin_vel_b[:, :2]
    command = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum(torch.square(command - vel_xy), dim=1)
    return torch.exp(-error / (std**2))

def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_z = asset.data.root_ang_vel_b[:, 2]
    command_z = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(command_z - ang_vel_z)
    return torch.exp(-error / (std**2))

# =============================================================================
# 2. Gait & Locomotion Quality
# =============================================================================

def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
) -> torch.Tensor:
    """[Gait] Touchdown moment에만 지급되는 체공 시간 보상."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    # Match Isaac Lab semantics:
    # - sensor.force_threshold decides contact state / last_air_time updates
    # - reward threshold is the minimum target air-time, not a force threshold
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    total_reward = torch.sum((air_time - threshold) * first_contact, dim=1)

    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving = (cmd > 0.1).float()
    return total_reward * is_moving


def _nearest_height_scanner_hits(
    sensor: RayCaster,
    foot_pos_w: torch.Tensor,
) -> torch.Tensor:
    """Map each foot to the nearest height-scanner ray and return its terrain hit z."""
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1)
    sensor_quat_yaw = math_utils.yaw_quat(sensor.data.quat_w).unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
    foot_pos_yaw = math_utils.quat_apply_inverse(sensor_quat_yaw, foot_pos_w - sensor_pos_w)

    ray_xy = sensor.ray_starts[:, :, :2]
    foot_xy = foot_pos_yaw[:, :, :2]
    dist_sq = torch.sum(torch.square(foot_xy.unsqueeze(2) - ray_xy.unsqueeze(1)), dim=-1)
    nearest_ray_ids = torch.argmin(dist_sq, dim=-1)

    ray_hits_z = sensor.data.ray_hits_w[:, :, 2]
    terrain_z = torch.gather(ray_hits_z, 1, nearest_ray_ids)

    valid_hits = torch.isfinite(ray_hits_z) & (ray_hits_z > -100.0)
    fallback_terrain_z = torch.sum(ray_hits_z * valid_hits.float(), dim=1) / valid_hits.float().sum(dim=1).clamp_min(1.0)
    terrain_z = torch.where(torch.isfinite(terrain_z), terrain_z, fallback_terrain_z.unsqueeze(1))
    return terrain_z


def foot_clearance(
    env: ManagerBasedRLEnv,
    command_name: str,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Reward swing feet for maintaining a terrain-relative clearance target."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    height_sensor: RayCaster = env.scene[height_sensor_cfg.name]

    if asset_cfg.body_ids is None:
        asset_cfg.resolve(env.scene)
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    terrain_z = _nearest_height_scanner_hits(height_sensor, foot_pos_w)
    foot_clearance_m = foot_pos_w[:, :, 2] - terrain_z

    contact_force = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    swing_mask = (contact_force < float(contact_sensor.cfg.force_threshold)).float()
    foot_speed_xy = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    speed_gate = torch.tanh(tanh_mult * foot_speed_xy)
    active_gate = swing_mask * speed_gate

    clearance_error = torch.square(foot_clearance_m - target_height)
    mean_error = torch.sum(clearance_error * active_gate, dim=1) / active_gate.sum(dim=1).clamp_min(1.0)
    reward = torch.exp(-mean_error / (std**2))
    reward = torch.where(active_gate.sum(dim=1) > 0.0, reward, torch.zeros_like(reward))

    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving = (cmd > 0.1).float()
    return reward * is_moving


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """[Gait/MotorDeg] 미끄러짐 방지 (단일 ContactSensor, body_names 필터링)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids is None:
        asset_cfg.resolve(env.scene)

    contact_force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    in_contact = (contact_force > 1.0).float()  # (num_envs, num_feet)

    foot_vel = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    slide_penalty = torch.sum(foot_vel * in_contact, dim=1)
    return slide_penalty


def feet_air_time_aggregated(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    sensor_names: List[str]
) -> torch.Tensor:
    """
    [Gait] Aggregated touchdown-only air-time reward.

    `threshold` is the minimum target air-time, following Isaac Lab semantics.
    """
    total_reward = torch.zeros(env.num_envs, device=env.device)
    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving = (cmd > 0.1).float()

    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]

        first_contact = sensor.compute_first_contact(env.step_dt)[:, 0]
        air_time = sensor.data.last_air_time[:, 0]
        total_reward += (air_time - threshold) * first_contact
            
    return total_reward * is_moving


def feet_slide_aggregated(
    env: ManagerBasedRLEnv,
    sensor_names: List[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [Gait/MotorDeg] 미끄러짐 방지.
    net_forces_w 사용 (일관성 유지)
    """
    # body_ids 해결
    if asset_cfg.body_ids is None:
        asset_cfg.resolve(env.scene)
    
    asset: Articulation = env.scene[asset_cfg.name]
    total_slide_penalty = torch.zeros(env.num_envs, device=env.device)
    
    for i, name in enumerate(sensor_names):
        sensor: ContactSensor = env.scene[name]
        
        # net_forces_w 사용
        contact_force = torch.norm(sensor.data.net_forces_w[:, 0, :], dim=-1)
        in_contact = contact_force > 1.0
        
        # 발 속도
        if isinstance(asset_cfg.body_ids, (list, tuple)):
            body_idx = asset_cfg.body_ids[i]
        else:
            body_idx = i
        
        foot_vel = torch.norm(asset.data.body_lin_vel_w[:, body_idx, :2], dim=-1)
        total_slide_penalty += (foot_vel * in_contact.float())
        
    return total_slide_penalty

# =============================================================================
# 3. Stability & Terrain Awareness
# =============================================================================

def base_height_l2_terrain_aware(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    
    root_z = asset.data.root_pos_w[:, 2]
    ray_hits_z = sensor.data.ray_hits_w[..., 2]
    valid_mask = ray_hits_z > -100.0 
    terrain_z = torch.sum(ray_hits_z * valid_mask.float(), dim=1) / (torch.sum(valid_mask.float(), dim=1) + 1e-6)
    no_hit = torch.sum(valid_mask.float(), dim=1) < 1.0
    terrain_z = torch.where(no_hit, torch.zeros_like(root_z), terrain_z)

    current_height_rel = root_z - terrain_z
    return torch.square(target_height - current_height_rel)
