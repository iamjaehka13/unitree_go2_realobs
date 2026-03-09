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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# [Fix #6] Isaac Lab 표준 함수(flat_orientation_l2, action_rate_l2 등)와의
# 이름 충돌 방지. MotorDeg 전용 보상 함수만 wildcard export.
__all__ = [
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_air_time",
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
    """[Gait] 체공 시간 보상 (단일 ContactSensor, body_names 필터링)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    if sensor_cfg.body_ids is None:
        sensor_cfg.resolve(env.scene)

    contact_force = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    in_contact = contact_force > threshold  # (num_envs, num_feet)

    air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward_per_foot = air_time * in_contact.float()
    total_reward = torch.sum(reward_per_foot, dim=1)

    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving = (cmd > 0.1).float()
    return total_reward * is_moving


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
    [Gait] 체공 시간 보상.
    last_air_time이 net_forces_w 기반이므로, 접촉 판단도 net_forces_w를 사용합니다.
    """
    total_reward = torch.zeros(env.num_envs, device=env.device)
    cmd = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    is_moving = (cmd > 0.1).float()

    for name in sensor_names:
        sensor: ContactSensor = env.scene[name]
        
        # net_forces_w 사용 (last_air_time과 동일한 소스)
        contact_force = torch.norm(sensor.data.net_forces_w[:, 0, :], dim=-1)
        in_contact = contact_force > threshold
        
        # 발이 착지한 순간, 지금까지의 체공시간을 보상으로 지급
        air_time = sensor.data.last_air_time[:, 0]
        total_reward += (air_time * in_contact.float())
            
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
