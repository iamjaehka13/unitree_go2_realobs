# =============================================================================
# unitree_go2_realobs/mdp/terminations/health.py
# =============================================================================

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""Termination checks that read MotorDeg state without mutating simulation state."""

# =============================================================================
# 1. Electrical Safety (전기적 보호)
# =============================================================================

def torque_overload(
    env: ManagerBasedRLEnv,
    limit_percentage: float = 0.95,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [Stress] 제어기 요구 토크(Commanded Torque) 과부하 감지.
    물리 엔진이 적용한 토크(applied)와 제어기가 요구한 토크(computed) 중 큰 값을 감시합니다.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    effort_limits = asset.data.joint_effort_limits

    # Per-environment torque source selection:
    # Use computed_torque where non-zero (controller active), else applied_torque
    computed = asset.data.computed_torque
    applied = asset.data.applied_torque
    has_computed = torch.any(computed != 0.0, dim=1, keepdim=True)  # (N, 1)
    monitoring_torque = torch.where(has_computed, torch.abs(computed), torch.abs(applied))

    # 0으로 나누기 방지
    effort_limits = torch.clamp(effort_limits, min=1e-6)
    
    violation = monitoring_torque > (effort_limits * limit_percentage)
    return torch.any(violation, dim=1)


def motor_stall(
    env: ManagerBasedRLEnv,
    stall_time_threshold_s: float = 0.2
) -> torch.Tensor:
    """
    [Critical MotorDeg] 모터 구속(Stall) 지속 감지 (Judge Only).
    
    Note:
        이 함수는 물리적인 스톨 판정(속도 < 0.1 rad/s, 토크 > 90% 등)을 수행하지 않습니다.
        물리적 판정은 'motor_deg/interface.py'에서 매 스텝 수행되어 
        'env.motor_deg_state.stall_timer'에 누적됩니다.
        
        여기서는 오직 '누적된 시간'이 허용치(stall_time_threshold_s)를 
        초과했는지만 심판합니다.
        
        * 스톨의 물리적 기준 변경: motor_deg/constants.py -> MIN_STALL_VELOCITY 수정
    """
    # 1. 안전 장치: MotorDeg State가 초기화되지 않았으면 False 반환
    if not hasattr(env, "motor_deg_state") or not hasattr(env.motor_deg_state, "stall_timer"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 2. 판정: stall_timer가 (N, J) per-motor이면 어떤 모터라도 초과 시 종료.
    stall_exceeded = env.motor_deg_state.stall_timer > stall_time_threshold_s
    if stall_exceeded.dim() > 1:
        return torch.any(stall_exceeded, dim=1)
    return stall_exceeded


def thermal_runaway(
    env: ManagerBasedRLEnv,
    threshold_temp: float = 90.0,
    use_case_proxy: bool = False,
    coil_to_case_delta_c: float = 5.0,
) -> torch.Tensor:
    """
    [Thermal Safety] 온도 임계값 초과 감지.
    기본은 coil_temp를 사용하며, use_case_proxy=True이면
    case/housing 온도(또는 coil-delta proxy)를 사용합니다.
    """
    if not hasattr(env, "motor_deg_state") or not hasattr(env.motor_deg_state, "coil_temp"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    temps = env.motor_deg_state.coil_temp
    if use_case_proxy:
        case_like = None
        for name in (
            "motor_case_temp",
            "case_temp",
            "motor_temp_case",
            "housing_temp",
            "motor_housing_temp",
        ):
            if hasattr(env.motor_deg_state, name):
                val = getattr(env.motor_deg_state, name)
                if isinstance(val, torch.Tensor):
                    case_like = val
                    break
        if case_like is None:
            case_like = env.motor_deg_state.coil_temp - float(coil_to_case_delta_c)
        temps = case_like

    return torch.any(temps > threshold_temp, dim=1)


# =============================================================================
# 2. Mechanical Integrity (기계적 건전성)
# =============================================================================

def joint_limit_violation(
    env: ManagerBasedRLEnv,
    buffer_rad: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [Mechanism] 관절 가동 범위 이탈 방지.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos

    # Soft limit 우선, 없으면 Default limit 사용
    if hasattr(asset.data, "soft_joint_pos_limits"):
        limits = asset.data.soft_joint_pos_limits
    else:
        limits = asset.data.default_joint_pos_limits

    lower_limits = limits[..., 0]
    upper_limits = limits[..., 1]

    out_of_lower = joint_pos < (lower_limits - buffer_rad)
    out_of_upper = joint_pos > (upper_limits + buffer_rad)

    return torch.any(out_of_lower | out_of_upper, dim=1)


def joint_velocity_limit(
    env: ManagerBasedRLEnv,
    limit_percentage: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Integrity] 모터/베어링 허용 속도 초과 감지."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel)
    vel_limits = asset.data.joint_vel_limits

    violation = joint_vel > (vel_limits * limit_percentage)
    return torch.any(violation, dim=1)


def foot_force_overload(
    env: ManagerBasedRLEnv,
    limit_force_n: float = 400.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces")
) -> torch.Tensor:
    """
    [Structural Safety] 발(Foot)에 가해지는 충격 하중 과부하 감지.
    """
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w
    
    # Filter: 이름에 'foot'이 포함된 링크만 검사
    if hasattr(sensor, "body_names"):
        foot_indices = [i for i, name in enumerate(sensor.body_names) if "foot" in name]
        if len(foot_indices) > 0:
            forces = forces[:, foot_indices, :]
    
    impact_magnitude = torch.norm(forces, dim=-1)
    overload = torch.any(impact_magnitude > limit_force_n, dim=1)
    return overload


# =============================================================================
# 3. Control Stability (제어 안정성)
# =============================================================================

def large_tracking_error(
    env: ManagerBasedRLEnv,
    threshold_rad: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Control Safety] 목표 위치와 실제 위치 간의 추종 오차 과다 감지."""
    asset: Articulation = env.scene[asset_cfg.name]
    current_pos = asset.data.joint_pos
    target_pos = asset.data.joint_pos_target

    error = torch.abs(target_pos - current_pos)
    violation = error > threshold_rad
    has_violation = torch.any(violation, dim=1)

    # Per-env guard: skip envs where all targets are zero (init phase)
    target_active = torch.any(target_pos != 0.0, dim=1)
    return has_violation & target_active


def energy_depletion(
    env: ManagerBasedRLEnv,
    max_energy_j: float = 10000.0
) -> torch.Tensor:
    """
    [Mission Fail] 누적 에너지 소비량이 허용 예산을 초과했는지 판정.
    """
    if not hasattr(env, "motor_deg_state"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # cumulative_energy는 state.py에서 항상 (N,) shape으로 정의됨
    current_energy = env.motor_deg_state.cumulative_energy
        
    return current_energy > max_energy_j
