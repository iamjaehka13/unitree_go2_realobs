# =============================================================================
# unitree_go2_phm/mdp/observations/contact.py
# Contact-derived PHM observation helpers for CoP, impact, and vibration metrics.
# =============================================================================
# Purpose:
#   Compute PHM (Physical Health Monitoring) metrics using foot contact data.
#   Data sources are intentionally separated:
#     1. CoP/locomotion -> force_matrix_w (filtered ground contacts only)
#     2. Impact/damage  -> net_forces_w (total mechanical load)
# =============================================================================

from __future__ import annotations

import torch
import logging
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import ContactSensorData


# =============================================================================
# PHM Constants
# =============================================================================
try:
    from ...phm.constants import (
        NORMAL_AXIS,            # Vertical axis index (Z = 2)
        CONTACT_THRESHOLD_N,    # Contact detection threshold [N]
        EPS,                    # Numerical stability
        NOMINAL_LOAD_N,         # Nominal load per foot [N]
        IMPACT_NOISE_THRESHOLD, # Minimum meaningful impact [N/s]
        GRAVITY_VAL,            # [m/s^2] Gravity acceleration (SSOT)
    )
except ImportError:
    NORMAL_AXIS = 2
    CONTACT_THRESHOLD_N = 5.0
    EPS = 1e-6
    NOMINAL_LOAD_N = 120.0
    IMPACT_NOISE_THRESHOLD = 10.0
    GRAVITY_VAL = 9.81


# =============================================================================
# Logging Guards
# =============================================================================
_LOG_FLAGS = {
    "force_matrix": False,
    "contact_points": False,
    "missing_imu": False,
    "short_history": False,
}


# =============================================================================
# Helper: World -> Base Force Transformation
# =============================================================================
def _get_total_forces_in_base_frame(
    env: ManagerBasedEnv,
    sensor: ContactSensor,
    body_indices: torch.Tensor | list | slice,
) -> torch.Tensor:
    """
    [PHM Metric Only]
    Transform TOTAL contact forces (net_forces_w) to Base frame.
    Used for mechanical load/stress calculation where source of force 
    (ground vs self-collision) does not matter.
    """
    # [PHM Truth] 기계적 부하는 필터링되지 않은 총 외력(net_forces)을 따른다.
    forces_w = sensor.data.net_forces_w[:, body_indices, :]

    robot = env.scene["robot"]
    root_quat_w = robot.data.root_quat_w  # (env, 4)

    num_bodies = forces_w.shape[1]
    quat_expanded = root_quat_w.unsqueeze(1).expand(-1, num_bodies, -1)

    forces_b = math_utils.quat_apply_inverse(quat_expanded, forces_w)
    return forces_b


# =============================================================================
# CoP (Center of Pressure) - Locomotion Metric
# =============================================================================
def _calculate_cop_body_frame_vectorized(
    env: ManagerBasedEnv,
    sensor: ContactSensor,
    body_indices: torch.Tensor | slice = slice(None),
) -> torch.Tensor:
    """
    Compute Center of Pressure (CoP) in the robot Base frame.
    
    [STRICT MODE]
    Only uses 'force_matrix_w' (filtered ground contact).
    Does NOT fallback to 'net_forces_w' to prevent 'Ghost Forces' 
    (e.g., self-collisions) from corrupting stability metrics.
    """
    data: ContactSensorData = sensor.data
    idx_len = sensor.num_bodies if isinstance(body_indices, slice) else len(body_indices)

    # 1. Contact Position Check
    if data.contact_pos_w is None or data.contact_pos_w.numel() == 0:
        if not _LOG_FLAGS["contact_points"]:
            logging.error(
                "[PHM Critical] 'contact_pos_w' empty. "
                "Set 'track_contact_points=True' in ContactSensorCfg."
            )
            _LOG_FLAGS["contact_points"] = True
        return torch.zeros((env.num_envs, idx_len, 3), device=env.device)

    # 2. Strict Data Source Enforcement
    # CoP is undefined without filtered ground-contact forces.
    # Return zeros instead of falling back to net_forces_w.
    if not hasattr(data, "force_matrix_w") or data.force_matrix_w is None:
        if not _LOG_FLAGS["force_matrix"]:
            logging.error(
                "[PHM Strict] 'force_matrix_w' missing. "
                "Cannot compute CoP safely. Check 'filter_prim_paths_expr' in Cfg. "
                "Ignoring CoP calculation to prevent ghost forces."
            )
            _LOG_FLAGS["force_matrix"] = True
        return torch.zeros((env.num_envs, idx_len, 3), device=env.device)

    # 3. Calculation using ONLY valid ground contacts
    safe_contact_pos_w = torch.nan_to_num(data.contact_pos_w, nan=0.0)
    
    # (Env, Body, Filtered_Collisions, Axis)
    weights = data.force_matrix_w[:, body_indices, :, NORMAL_AXIS]

    weights = torch.nan_to_num(weights, nan=0.0)
    weights = torch.clamp(weights, min=0.0)

    total_fz = weights.sum(dim=2)
    denom = total_fz.unsqueeze(-1) + EPS

    # Weighted sum of contact positions
    cop_w = (
        safe_contact_pos_w[:, body_indices, :, :] * weights.unsqueeze(-1)
    ).sum(dim=2) / denom

    robot = env.scene["robot"]
    cop_b, _ = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.unsqueeze(1),
        robot.data.root_quat_w.unsqueeze(1),
        cop_w,
    )

    is_contact = (total_fz > CONTACT_THRESHOLD_N).unsqueeze(-1).float()
    return cop_b * is_contact


def contact_positions_body_frame(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """RL Observation: CoP of each foot in Base frame."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    cop_b = _calculate_cop_body_frame_vectorized(
        env, sensor, sensor_cfg.body_ids
    )
    return cop_b.view(env.num_envs, -1)


# =============================================================================
# Impact (Landing Shock) - PHM Metric
# =============================================================================
def contact_impact(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    Compute landing impact as positive force derivative (loading shock).
    
    [PHM Truth]
    Uses 'net_forces_w' because mechanical damage is caused by 
    the MAGNITUDE of force, regardless of the source (Ground vs Self).
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    data: ContactSensorData = sensor.data
    body_ids = sensor_cfg.body_ids

    if (
        not hasattr(data, "net_forces_w_history")
        or data.net_forces_w_history is None
        or data.net_forces_w_history.shape[1] < 2
    ):
        if not _LOG_FLAGS["short_history"]:
            logging.warning(
                "[PHM Critical] Contact history < 2. "
                "Set 'history_length >= 2' in ContactSensorCfg."
            )
            _LOG_FLAGS["short_history"] = True

        return torch.zeros((env.num_envs, len(body_ids)), device=env.device)

    # Use Total Forces (Impact is Impact)
    forces_w_t0 = data.net_forces_w_history[:, 0, body_ids, :]
    forces_w_t1 = data.net_forces_w_history[:, 1, body_ids, :]

    robot = env.scene["robot"]
    quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)

    fz_t0 = math_utils.quat_apply_inverse(quat, forces_w_t0)[..., NORMAL_AXIS]
    fz_t1 = math_utils.quat_apply_inverse(quat, forces_w_t1)[..., NORMAL_AXIS]

    impact = torch.relu(fz_t0 - fz_t1) / max(env.physics_dt, EPS)

    impact = torch.where(
        impact > IMPACT_NOISE_THRESHOLD,
        impact,
        torch.zeros_like(impact),
    )

    return impact.view(env.num_envs, -1)


# =============================================================================
# Weighted Contact Vibration (IMU-based) - Hybrid Metric
# =============================================================================
def weighted_contact_acceleration(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    imu_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    PHM Metric: |IMU dynamic acceleration| × contact load ratio
    
    Uses 'net_forces_w' for load ratio because self-collisions 
    also contribute to stress that amplifies vibration damage.
    """
    if (not hasattr(imu_cfg, "name")) or imu_cfg.name == "0" or imu_cfg.name == 0:
        raise RuntimeError(
            f"Invalid imu_cfg passed to weighted_contact_acceleration. "
            f"imu_cfg={imu_cfg!r}, sensor_cfg={sensor_cfg!r}"
        )

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    try:
        imu_sensor: Imu = env.scene[imu_cfg.name]
    except KeyError:
        if not _LOG_FLAGS["missing_imu"]:
            logging.error(
                f"[PHM Error] IMU sensor '{imu_cfg.name}' not found in Scene."
            )
            _LOG_FLAGS["missing_imu"] = True

        return torch.zeros((env.num_envs, len(sensor_cfg.body_ids)), device=env.device)

    # projected_gravity_b 기반 중력 제거로 기울어진 자세에서도 동적 진동만 남긴다.
    # projected_gravity_b ≈ [0,0,-1] (단위 벡터), lin_acc_b ≈ [0,0,+9.81] (정지 시)
    # 정지 시 결과가 0이 되려면: imu + (pg * g) = [0,0,+9.81] + [0,0,-9.81] = 0 ✓
    # (utils.py의 compute_kinematic_accel과 동일한 수식)
    robot = env.scene["robot"]
    gravity_in_body = robot.data.projected_gravity_b * GRAVITY_VAL  # (N, 3)
    dynamic_accel = imu_sensor.data.lin_acc_b + gravity_in_body  # (N, 3)
    dynamic_vibration = torch.norm(dynamic_accel, dim=-1, keepdim=True)  # (N, 1)

    # Use Total Forces for Load Calculation
    forces_b = _get_total_forces_in_base_frame(
        env, contact_sensor, sensor_cfg.body_ids
    )
    fz_b = forces_b[..., NORMAL_AXIS]

    load_ratio = fz_b / (NOMINAL_LOAD_N + EPS)

    metric = (
        dynamic_vibration
        * load_ratio
        * (fz_b > CONTACT_THRESHOLD_N).float()
    )

    return metric.view(env.num_envs, -1)
