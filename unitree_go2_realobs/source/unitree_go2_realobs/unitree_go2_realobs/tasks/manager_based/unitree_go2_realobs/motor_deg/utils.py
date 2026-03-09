# =============================================================================
# unitree_go2_realobs/motor_deg/utils.py
# Shared physics, math, and lifecycle helpers used across MotorDeg modules.
# =============================================================================
from __future__ import annotations
import torch
import logging
from typing import Optional, Tuple, Union

# Isaac Lab Core Imports
from isaaclab.sensors import Imu
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv

# Shared constants import with conservative fallbacks for isolated execution.
try:
    from .constants import (
        # Electrical & Mechanical
        R_NOMINAL, ALPHA_CU, GEAR_RATIO, EPS, KT_NOMINAL,
        R_MOSFET, V_DROP,
        GEAR_FRICTION_COULOMB, GEAR_FRICTION_VISCOUS,
        K_HYST, K_EDDY, B_VISCOUS,
        GRAVITY_VAL,
        # Thermal Parameters
        T_AMB,
        # Battery Parameters
        BATTERY_OCV_BASE_V, BATTERY_OCV_SOC_SLOPE_V,
        BATTERY_INTERNAL_RESISTANCE, BATTERY_MAX_SAG_RATIO,
        # Regen Defaults
        REGEN_PEAK_EFFICIENCY, REGEN_OPTIMAL_SPEED, REGEN_WIDTH
    )
except ImportError:
    logging.warning("[MotorDeg Utils] Constants not found. Using conservative defaults.")
    R_NOMINAL = 0.15; ALPHA_CU = 0.00393; GEAR_RATIO = 6.33; EPS = 1e-6; KT_NOMINAL = 0.08
    R_MOSFET = 0.01; V_DROP = 0.7
    GEAR_FRICTION_COULOMB = 0.1; GEAR_FRICTION_VISCOUS = 0.01
    K_HYST = 0.01; K_EDDY = 0.0001; B_VISCOUS = 0.001
    GRAVITY_VAL = 9.81
    T_AMB = 25.0
    BATTERY_OCV_BASE_V = 26.8; BATTERY_OCV_SOC_SLOPE_V = 4.1
    BATTERY_INTERNAL_RESISTANCE = 0.06; BATTERY_MAX_SAG_RATIO = 0.40
    REGEN_PEAK_EFFICIENCY = 0.60; REGEN_OPTIMAL_SPEED = 12.0; REGEN_WIDTH = 10.0

logger = logging.getLogger("MotorDeg_Core")

# =============================================================================
# 1. Tensor Safety & Access Helpers
# =============================================================================

def safe_tensor(data: torch.Tensor, default_val: float = 0.0) -> torch.Tensor:
    """[Safety] Replaces NaNs/Infs with a default value to prevent crash."""
    mask = torch.isnan(data) | torch.isinf(data)
    if torch.any(mask):
        return torch.where(mask, torch.tensor(default_val, device=data.device), data)
    return data

def get_tensor_data(entity: object, attr_path: str, env_ids: torch.Tensor) -> torch.Tensor:
    """[Helper] Retrieves tensor data from nested attributes safely."""
    obj = entity
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    return obj[env_ids]

# =============================================================================
# 2. Physics Core (Thermal & Electrical)
# =============================================================================

def compute_battery_voltage(
    soc: torch.Tensor,
    load_watts: torch.Tensor,
    internal_resistance: float = BATTERY_INTERNAL_RESISTANCE,
    max_sag_ratio: float = BATTERY_MAX_SAG_RATIO
) -> torch.Tensor:
    """
    [Physics] SOC와 부하를 고려한 배터리 단자 전압 계산 (Soft BMS 적용).
    """
    soc_clamped = torch.clamp(soc, 0.0, 1.0)
    v_open = BATTERY_OCV_BASE_V + (BATTERY_OCV_SOC_SLOPE_V * soc_clamped)
    
    current_est = load_watts / (v_open + EPS)
    # Keep electrical model consistent with training assumption:
    # regen is not credited into SOC/pack charging in this project,
    # so negative load should not create artificial voltage boost.
    discharge_current = torch.clamp(current_est, min=0.0)
    v_drop = discharge_current * internal_resistance
    
    # Voltage sag limiter to prevent unrealistic pack collapse.
    max_allowed_drop = v_open * max_sag_ratio
    v_drop_clamped = torch.min(v_drop, max_allowed_drop)

    v_terminal = v_open - v_drop_clamped
    # Low-end clamp remains relaxed so policies can experience a broad low-voltage regime
    # during simulation (even if operational safety thresholds are higher).
    return torch.clamp(v_terminal, min=18.0, max=33.6)

def compute_regenerative_efficiency(
    vel: torch.Tensor,
    peak_eff: float = REGEN_PEAK_EFFICIENCY,
    opt_speed: float = REGEN_OPTIMAL_SPEED,
    width: float = REGEN_WIDTH
) -> torch.Tensor:
    """[Physics] 회생 제동 효율 곡선."""
    speed_abs = torch.abs(vel)
    efficiency = peak_eff * torch.exp(-torch.square((speed_abs - opt_speed) / (width + EPS)))
    return torch.where(speed_abs < 0.1, torch.zeros_like(efficiency), efficiency)

def compute_component_losses(
    torque: torch.Tensor,
    velocity: torch.Tensor,
    gear_ratio_tensor: Optional[torch.Tensor] = None,
    temp: Union[float, torch.Tensor] = T_AMB,
    external_friction_power: Optional[torch.Tensor] = None 
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[Physics] 상세 에너지 손실 모델 (Copper, Inverter, Mechanical)."""
    torque_abs = torch.abs(torque)
    vel_abs = torch.abs(velocity)
    ratio = GEAR_RATIO if gear_ratio_tensor is None else gear_ratio_tensor

    # 1. Copper Loss (Temp-dependent)
    current_est = torque_abs / (KT_NOMINAL * ratio + EPS)
    r_effective = R_NOMINAL * (1.0 + ALPHA_CU * (temp - T_AMB))
    p_copper = torch.square(current_est) * r_effective

    # 2. Inverter Loss
    p_inverter = (R_MOSFET * torch.square(current_est)) + (V_DROP * current_est)

    # 3. Mechanical Loss (Viscous + Core + Stiction)
    # Core loss(히스테리시스+와전류)는 모터 측 각속도에서 발생하므로
    # joint velocity가 아닌 motor velocity(= joint_vel x GEAR_RATIO)를 사용한다.
    omega_motor = vel_abs * ratio
    p_core = (K_HYST * omega_motor) + (K_EDDY * torch.square(omega_motor))
    
    if external_friction_power is not None:
        p_mech = external_friction_power + p_core
    else:
        # Gear friction과 viscous loss는 joint 측 속도(vel_abs) 기준
        p_friction = (GEAR_FRICTION_COULOMB + GEAR_FRICTION_VISCOUS * vel_abs) * vel_abs
        p_mech = p_friction + p_core + (B_VISCOUS * torch.square(vel_abs))

    return safe_tensor(p_copper), safe_tensor(p_inverter), safe_tensor(p_mech)

# =============================================================================
# 3. Kinematics Standard (IMU & Gravity Cancellation)
# =============================================================================

def compute_kinematic_accel(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    imu_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [Physics Standard] IMU 계측값(Proper Accel)에서 중력을 역산하여 제거.
    Isaac Lab의 Imu 센서는 기본적으로 중력이 포함된 proper acceleration을 반환합니다.
    projected_gravity_b(단위 벡터)에 g를 곱하여 더하면 중력 성분이 상쇄됩니다.
    """
    # 1. IMU Proper Acceleration (+g included)
    sensor: Imu = env.scene[imu_cfg.name]
    imu_accel_b = sensor.data.lin_acc_b[env_ids] 

    # 2. Gravity Direction in Body Frame
    robot: Articulation = env.scene[asset_cfg.name]
    grav_dir_b = robot.data.projected_gravity_b[env_ids]

    # 3. Mathematical Cancellation: (+9.81) + (-1.0 * 9.81) = 0.0
    g_val = torch.tensor(GRAVITY_VAL, device=env.device, dtype=imu_accel_b.dtype)
    kinematic_accel = imu_accel_b + (grav_dir_b * g_val)
    
    return safe_tensor(kinematic_accel)

def compute_load_ratio(torque: torch.Tensor, effort_limits: torch.Tensor) -> torch.Tensor:
    """[Analytics] 모터 정격 대비 부하율."""
    safe_limits = torch.where(effort_limits < EPS, torch.ones_like(effort_limits) * EPS, effort_limits)
    return torch.abs(torque) / safe_limits
