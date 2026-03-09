# ---------------------------------------------------------------------
# unitree_go2_realobs/mdp/terminations/fall.py
# ---------------------------------------------------------------------
from __future__ import annotations

import torch
import math
import logging
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import Articulation

# =============================================================================
# [MotorDeg SSOT] Constants
# =============================================================================
try:
    from ...motor_deg.constants import (
        CONTACT_THRESHOLD_N,
        CHASSIS_SCAN_OFFSETS,
        INVALID_TERRAIN_HEIGHT,
        BASE_HEIGHT_MIN
    )
except ImportError:
    CONTACT_THRESHOLD_N = 5.0
    CHASSIS_SCAN_OFFSETS = [
        [ 0.2,  0.0,  0.05],
        [-0.2,  0.0,  0.05],
        [ 0.0,  0.1,  0.05],
        [ 0.0, -0.1,  0.05],
        [ 0.0,  0.0,  0.05],
    ]
    INVALID_TERRAIN_HEIGHT = -100.0
    BASE_HEIGHT_MIN = 0.21

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# [Internal Constants]
# =============================================================================
DEG_TO_RAD = math.pi / 180.0
RAD_TO_DEG = 180.0 / math.pi
MIN_HISTORY_FOR_WMA = 3
ACOS_EPS = 1e-7
FLYING_BLIND_HEIGHT_THRESHOLD = -500.0
FLYING_SAFE_MARGIN_FACTOR = 2.0

_FALL_LOG_FLAGS = {
    "terrain_sample_failed": False,
    "height_sensor_sample_failed": False,
}


def _body_ids_in_bounds(
    body_ids: list[int] | slice | torch.Tensor | None,
    num_bodies: int,
) -> bool:
    """Return True when body index spec is valid for the given body count."""
    if body_ids is None:
        return True
    if isinstance(body_ids, slice):
        # Slice indexing is inherently bounded by tensor shape.
        return True

    if isinstance(body_ids, torch.Tensor):
        ids = body_ids.detach().to(device="cpu", dtype=torch.long).view(-1)
    else:
        try:
            ids = torch.as_tensor(body_ids, dtype=torch.long).view(-1)
        except Exception:
            return False

    if ids.numel() == 0:
        return True

    min_idx = int(ids.min().item())
    max_idx = int(ids.max().item())
    return min_idx >= -num_bodies and max_idx < num_bodies


def _sample_ground_from_height_sensor(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    height_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor | None:
    """Sample terrain reference height from height scanner ray hits."""
    if height_sensor_cfg is None or not hasattr(height_sensor_cfg, "name"):
        return None

    try:
        sensor: RayCaster = env.scene[height_sensor_cfg.name]
        ray_hits_w = sensor.data.ray_hits_w
        if ray_hits_w is None or ray_hits_w.numel() == 0:
            return None

        if env_ids is not None:
            ray_hits_w = ray_hits_w[env_ids]

        hit_z = ray_hits_w[..., 2]
        valid_mask = torch.isfinite(hit_z) & (hit_z > INVALID_TERRAIN_HEIGHT)
        safe_hits = torch.where(valid_mask, hit_z, torch.full_like(hit_z, INVALID_TERRAIN_HEIGHT))
        max_hit_z = torch.max(safe_hits, dim=1).values
        has_valid_hit = torch.any(valid_mask, dim=1)
        invalid_ref = torch.full_like(max_hit_z, INVALID_TERRAIN_HEIGHT)
        return torch.where(has_valid_hit, max_hit_z, invalid_ref)
    except Exception as err:
        if not _FALL_LOG_FLAGS["height_sensor_sample_failed"]:
            logging.warning(
                "[Fall] Height sensor sampling failed in base_height_fall; "
                "continuing without height scanner. error=%s",
                err,
            )
            _FALL_LOG_FLAGS["height_sensor_sample_failed"] = True
        return None


# =============================================================================
# [Geometry Termination]
# =============================================================================
def base_height_fall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    minimum_height: float = BASE_HEIGHT_MIN,
    use_terrain_height: bool = True,
    contact_threshold: float = CONTACT_THRESHOLD_N,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    contact_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    [Stability] Grid-Based Belly-Aware Height Check with Hybrid Terrain Support.
    [Optimization] env_ids is None일 때 torch.arange 생성을 회피함.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    foot_indices = robot_cfg.body_ids

    # 1. 데이터 추출 (Slicing vs Indexing 분기)
    if env_ids is None:
        # [Zero-Copy] 전체 환경: 원본 뷰 사용
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        # body_pos_w: (num_envs, num_bodies, 3) -> (num_envs, num_feet, 3)
        feet_pos_w = robot.data.body_pos_w[:, foot_indices, :3]
    else:
        # [Indexed] 부분 환경: 필요한 만큼만 추출
        root_pos_w = robot.data.root_pos_w[env_ids]
        root_quat_w = robot.data.root_quat_w[env_ids]
        # Advanced Indexing for specific envs and bodies
        feet_pos_w = robot.data.body_pos_w[env_ids[:, None], foot_indices, :3]

    # Nan Handling
    root_pos = torch.nan_to_num(root_pos_w, nan=minimum_height)
    root_z = root_pos[:, 2]

    # 2. 지형 높이 샘플링
    terrain_z_ref = torch.full_like(root_z, INVALID_TERRAIN_HEIGHT)

    if use_terrain_height and hasattr(env.scene, "terrain") and env.scene.terrain is not None:
        is_plane = False
        if hasattr(env.scene.terrain, "cfg") and hasattr(env.scene.terrain.cfg, "terrain_type"):
            if env.scene.terrain.cfg.terrain_type == "plane":
                is_plane = True
        
        if is_plane:
            terrain_z_ref[:] = 0.0
        else:
            sample_height_fn = getattr(env.scene.terrain, "sample_height", None)
            if sample_height_fn is None or not callable(sample_height_fn):
                # TerrainImporter in this Isaac Lab version does not expose sample_height().
                # Rely on height-scanner and contact-based ground estimate below.
                sample_height_fn = None
            try:
                if sample_height_fn is not None:
                    # A. 섀시 하부 스캔
                    offsets_local = torch.tensor(
                        CHASSIS_SCAN_OFFSETS,
                        device=env.device,
                        dtype=root_pos.dtype,
                    )
                    num_points = offsets_local.shape[0]

                    # quat_apply() does not support broadcasting between env and sample axes.
                    # Build tensors with matched leading dimensions: (num_envs, num_points, *).
                    num_envs_active = root_quat_w.shape[0]
                    offsets_3d = offsets_local.unsqueeze(0).expand(num_envs_active, -1, -1).contiguous()
                    robot_quat = root_quat_w.unsqueeze(1).expand(-1, num_points, -1).contiguous()
                    offsets_w = math_utils.quat_apply(robot_quat, offsets_3d)

                    scan_points_3d = root_pos.unsqueeze(1) + offsets_w
                    scan_points = scan_points_3d[..., :2].view(-1, 2)

                    feet_xy = feet_pos_w[..., :2].view(-1, 2)

                    # Terrain sampling
                    chassis_t = sample_height_fn(scan_points).view(-1, len(CHASSIS_SCAN_OFFSETS))
                    num_foot_bodies = feet_pos_w.shape[1]
                    feet_t = sample_height_fn(feet_xy).view(-1, num_foot_bodies)

                    mean_chassis_t = torch.mean(chassis_t, dim=1)
                    max_feet_t, _ = torch.max(feet_t, dim=1)

                    terrain_z_ref = torch.maximum(mean_chassis_t, max_feet_t)
                    terrain_z_ref = torch.nan_to_num(terrain_z_ref, nan=INVALID_TERRAIN_HEIGHT)
            except Exception as err:
                if not _FALL_LOG_FLAGS["terrain_sample_failed"]:
                    logging.warning(
                        "[Fall] Terrain height sampling failed in base_height_fall; "
                        "falling back to invalid terrain reference. error=%s",
                        err,
                    )
                    _FALL_LOG_FLAGS["terrain_sample_failed"] = True

    # Height scanner 보조 채널 결합 (height_sensor_cfg를 실제로 사용)
    if use_terrain_height:
        sensor_ground_z = _sample_ground_from_height_sensor(env, env_ids, height_sensor_cfg)
        if sensor_ground_z is not None:
            terrain_z_ref = torch.maximum(terrain_z_ref, sensor_ground_z)

    # 3. 접촉 센서 기반 Grounded 여부 판단
    sensor: ContactSensor = env.scene[contact_sensor_cfg.name]
    
    # [Optimization] force data access
    if env_ids is None:
        raw_forces = sensor.data.net_forces_w
    else:
        raw_forces = sensor.data.net_forces_w[env_ids]

    # -------------------------------------------------------------------------
    # [FIX] Tensor Shape Mismatch Resolver
    # -------------------------------------------------------------------------
    # feet_pos_w는 foot_indices(5개?)만 뽑았는데, raw_forces는 전체(19개)를 가져올 수 있음.
    # 두 텐서의 차원을 일치시키기 위해 센서 데이터도 foot_indices로 슬라이싱해야 함.
    
    # 센서가 전체 바디를 추적 중이라면 슬라이싱 적용
    if raw_forces.shape[1] >= robot.num_bodies:
         contact_forces = raw_forces[:, foot_indices, :]
    else:
         # 만약 센서 자체가 이미 body_ids로 필터링되어 있다면 그대로 사용하거나 config 따름
         if contact_sensor_cfg.body_ids is not None:
            contact_forces = raw_forces[:, contact_sensor_cfg.body_ids, :]
         else:
            contact_forces = raw_forces
            
    # 최종 안전장치: 그래도 개수가 안 맞으면 강제로 맞춤 (Broadcast Error 방지)
    # 현재 에러: feet_pos_w(5) vs contact_forces(19) -> 이 코드가 실행되면 해결됨.
    if contact_forces.shape[1] != feet_pos_w.shape[1]:
        # raw_forces가 충분히 크면 인덱싱으로 줄임
        if raw_forces.shape[1] > feet_pos_w.shape[1]:
             # foot_indices 범위가 raw_forces 안에 있는지 체크
             if _body_ids_in_bounds(foot_indices, raw_forces.shape[1]):
                contact_forces = raw_forces[:, foot_indices, :]
             else:
                # 인덱스 초과 시 앞부분만 자름 (비상용)
                contact_forces = raw_forces[:, :feet_pos_w.shape[1], :]
    # -------------------------------------------------------------------------

    contact_forces = torch.nan_to_num(contact_forces, nan=0.0)
    is_grounded = torch.norm(contact_forces, dim=-1) > contact_threshold

    # 4. 실제 접촉 발 기반 지면 높이 계산
    feet_z = feet_pos_w[..., 2]

    # 이제 is_grounded와 feet_z의 차원이 일치하므로 에러 없음
    grounded_feet_z = torch.where(
        is_grounded,
        feet_z,
        torch.tensor(-1000.0, device=env.device)
    )

    max_grounded_z, _ = torch.max(grounded_feet_z, dim=1)
    ground_ref_z = torch.maximum(terrain_z_ref, max_grounded_z)

    # 5. Flying Blind 보호 로직
    is_flying_blind = (
        (max_grounded_z < FLYING_BLIND_HEIGHT_THRESHOLD)
        & (terrain_z_ref <= INVALID_TERRAIN_HEIGHT)
    )
    safe_ground_z = root_z - (minimum_height * FLYING_SAFE_MARGIN_FACTOR)
    final_ground_z = torch.where(is_flying_blind, safe_ground_z, ground_ref_z)

    relative_height = root_z - final_ground_z
    return relative_height < minimum_height


# =============================================================================
# [Orientation Termination]
# =============================================================================
def orientation_fall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    limit_angle_deg: float = 60.0,
    up_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """[Stability] Orientation Angle Check"""
    robot: Articulation = env.scene[asset_cfg.name]
    
    # [Optimization] Slicing
    if env_ids is None:
        gravity_vec = robot.data.projected_gravity_b
    else:
        gravity_vec = robot.data.projected_gravity_b[env_ids]

    ux, uy, uz = up_vector
    alignment = (gravity_vec[:, 0] * ux + gravity_vec[:, 1] * uy + gravity_vec[:, 2] * uz)
    alignment = torch.nan_to_num(alignment, nan=1.0)

    cos_theta = -alignment
    cos_theta = torch.clamp(cos_theta, -1.0 + ACOS_EPS, 1.0 - ACOS_EPS)

    tilt_deg = torch.acos(cos_theta) * RAD_TO_DEG
    return tilt_deg > limit_angle_deg


# =============================================================================
# [Collision Termination]
# =============================================================================
def chassis_impact_fall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    threshold: float = 200.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """[Collision] Weighted Moving Average Impact Detection"""
    sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [Optimization] Access Data Logic
    use_history = hasattr(sensor.data, "net_forces_w_history") and sensor.data.net_forces_w_history is not None

    if use_history:
        # History Shape: (env, history, bodies, 3)
        if env_ids is None:
            raw_history = sensor.data.net_forces_w_history
        else:
            raw_history = sensor.data.net_forces_w_history[env_ids]

        if sensor_cfg.body_ids is not None:
            target_history = raw_history[:, :, sensor_cfg.body_ids, :]
        else:
            target_history = raw_history
        
        target_history = torch.nan_to_num(target_history, nan=0.0)
        history_norm = torch.norm(target_history, dim=-1) # (env, history, bodies)
        steps = history_norm.shape[1]

        # WMA Calculation
        if steps >= MIN_HISTORY_FOR_WMA:
            weights = torch.arange(steps, 0, -1, device=env.device, dtype=torch.float32)
            weights = (weights / weights.sum()).view(1, steps, 1)
            filtered_forces = torch.sum(history_norm * weights, dim=1)
        else:
            filtered_forces = torch.mean(history_norm, dim=1)
        
        final_impact = filtered_forces
    else:
        # Current Frame Only
        if env_ids is None:
            current_forces = sensor.data.net_forces_w
        else:
            current_forces = sensor.data.net_forces_w[env_ids]

        if sensor_cfg.body_ids is not None:
            target_forces = current_forces[:, sensor_cfg.body_ids, :]
        else:
            target_forces = current_forces
            
        target_forces = torch.nan_to_num(target_forces, nan=0.0)
        final_impact = torch.norm(target_forces, dim=-1)

    max_impact, _ = torch.max(final_impact, dim=1)
    return max_impact > threshold


# =============================================================================
# [Illegal Contact Termination]
# =============================================================================
def illegal_contact_fall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    threshold: float = CONTACT_THRESHOLD_N,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """[Collision] Illegal Body Contact Detection"""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    # [Optimization]
    if env_ids is None:
        raw_forces = sensor.data.net_forces_w
    else:
        raw_forces = sensor.data.net_forces_w[env_ids]

    if sensor_cfg.body_ids is not None:
        # Boundary Check
        num_bodies = raw_forces.shape[1]
        if not _body_ids_in_bounds(sensor_cfg.body_ids, num_bodies):
            logging.error(
                "Body IDs %s out of bounds for num_bodies=%d.",
                sensor_cfg.body_ids,
                num_bodies,
            )
            # Fallback: Return boolean tensor of correct size
            num_targets = raw_forces.shape[0] # num_envs
            return torch.ones(num_targets, dtype=torch.bool, device=env.device)
            
        target_forces = raw_forces[:, sensor_cfg.body_ids, :]
    else:
        target_forces = raw_forces

    target_forces = torch.nan_to_num(target_forces, nan=0.0)
    forces_norm = torch.norm(target_forces, dim=-1)
    return torch.any(forces_norm > threshold, dim=1)


# =============================================================================
# [Non-Foot Contact Termination]
# =============================================================================
def non_foot_contact_fall(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """
    [Custom Termination] 발(Foot)이 아닌 부위가 땅에 닿으면 즉시 종료.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    
    if env_ids is None:
        raw_forces = sensor.data.net_forces_w
    else:
        raw_forces = sensor.data.net_forces_w[env_ids]

    if sensor_cfg.body_ids is not None:
        if not _body_ids_in_bounds(sensor_cfg.body_ids, raw_forces.shape[1]):
            return torch.zeros(raw_forces.shape[0], dtype=torch.bool, device=env.device)
        target_forces = raw_forces[:, sensor_cfg.body_ids, :]
    else:
        target_forces = raw_forces

    return torch.any(torch.norm(target_forces, dim=-1) > threshold, dim=1)
