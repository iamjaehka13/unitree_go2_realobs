# =============================================================================
# unitree_go2_realobs/motor_deg/models/degradation.py
# Fatigue accumulation helpers for MotorDeg motor health tracking.
# =============================================================================

from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Optional
from ..constants import (
    FATIGUE_EXPONENT,
    FATIGUE_SCALE,
    VIBRATION_FATIGUE_SCALE,
    EPS,
    TEMP_WARN_THRESHOLD,
    RATED_LOAD_FACTOR,
    THERMAL_STRESS_SCALE,
    MIN_STALL_VELOCITY
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation

def update_fatigue_index(env: ManagerBasedRLEnv, dt: float, env_ids: Optional[torch.Tensor] = None):
    """
    [MotorDeg Core] 기계적 피로 누적도 계산 (Torque + Vibration + Thermal Stress).
    """
    if env_ids is None:
        env_ids = slice(None)

    robot: Articulation = env.scene["robot"]

    # -------------------------------------------------------------------------
    # 1. 물리 데이터 추출 (구동 관절만 타겟팅)
    # -------------------------------------------------------------------------
    # 구동 조인트 인덱스 가져오기 (설정되지 않았으면 전체)
    joint_indices = getattr(env, "motor_deg_joint_indices", slice(None))

    # 실제 인가된 토크와 속도 추출 (절댓값 사용)
    # applied_torque: 물리 엔진이 계산한 순수 토크
    torque = torch.abs(robot.data.applied_torque[env_ids][:, joint_indices])
    velocity = torch.abs(robot.data.joint_vel[env_ids][:, joint_indices])

    # 추가 스트레스 요인: 진동 & 온도
    # vibration_g: (N,) -> (N, 1) - 로봇 바디 전체 진동 (IMU 기반)
    # jitter_intensity: (N, J) - 관절별 가속도 jitter (ShortTermBuffer 기반)
    # coil_temp: (N, J) - 각 관절별 코일 온도
    # 바디 진동과 관절별 jitter를 결합해 모터별 진동 스트레스를 분리한다.
    body_vibration = env.motor_deg_state.vibration_g[env_ids].unsqueeze(-1)  # (N, 1)
    joint_jitter = env.motor_deg_state.jitter_intensity[env_ids]             # (N, J)
    vibration_combined = body_vibration + joint_jitter                  # (N, J)
    temperature = env.motor_deg_state.coil_temp[env_ids]

    # -------------------------------------------------------------------------
    # 2. 정규화된 부하 (Normalized Load with Nominal Limits)
    # -------------------------------------------------------------------------
    # 정격 부하는 스펙 기준 한계를 사용한다.
    # runtime-degraded 값을 사용하면 열화와 부하율 사이에 과한 양의 피드백이 생긴다.
    if hasattr(env, "_nominal_effort_limits") and env._nominal_effort_limits is not None:
        effort_limits = env._nominal_effort_limits[env_ids][:, joint_indices]
    else:
        effort_limits = robot.data.joint_effort_limits[env_ids][:, joint_indices]
    
    # 정격 부하 계산 (Stall Torque * 안전 계수)
    rated_load = (effort_limits * RATED_LOAD_FACTOR) + EPS
    
    # 정격 대비 부하율 계산
    normalized_load = torque / rated_load

    # -------------------------------------------------------------------------
    # 3. 종합 스트레스 계수 (Vibration + Thermal Coupling)
    # -------------------------------------------------------------------------
    # 열적 스트레스: (현재온도 - 경고온도) / 스케일
    thermal_stress = torch.relu(
        (temperature - TEMP_WARN_THRESHOLD) / THERMAL_STRESS_SCALE
    )
    # 진동 스트레스: 바디 진동 + 관절별 jitter 결합 (N, J)
    vibration_stress = vibration_combined * VIBRATION_FATIGUE_SCALE
    
    # 진동 스트레스 (N, J) + 열적 스트레스 (N, J) -> (N, J)
    # 진동과 열은 기본 피로도(1.0)에 가산됨
    stress_factor = 1.0 + vibration_stress + thermal_stress

    # -------------------------------------------------------------------------
    # 4. 순간 피로 발생률 (L10 Life Theory 기반 변형 모델)
    # -------------------------------------------------------------------------
    # [Physics] 속도가 0이어도 토크가 높으면 베어링에 응력이 집중되므로 
    # 최소 속도(MIN_STALL_VELOCITY)를 보장하여 마모 계산 (Fretting corrosion 모사)
    safe_velocity = torch.clamp(velocity, min=MIN_STALL_VELOCITY)

    # 기본 마모율 ~ (부하/정격)^3 * 속도
    base_rate = torch.pow(normalized_load, FATIGUE_EXPONENT) * safe_velocity

    # 최종 마모율: 스트레스 계수(진동, 온도) 반영
    total_fatigue_rate = base_rate * stress_factor

    # -------------------------------------------------------------------------
    # 5. 누적 업데이트 (Integration)
    # -------------------------------------------------------------------------
    # 시간 적분 (Rate * dt * Scale)
    delta_fatigue = total_fatigue_rate * FATIGUE_SCALE * dt

    # 누적 피로도 업데이트
    env.motor_deg_state.fatigue_index[env_ids] += delta_fatigue
    
    # [Numeric Safety] NaN 방지 (학습 안정성)
    # Note: in-place nan_to_num_() on advanced-indexed tensor is a no-op for tensor env_ids
    env.motor_deg_state.fatigue_index[env_ids] = torch.nan_to_num(
        env.motor_deg_state.fatigue_index[env_ids], nan=0.0
    )

    # 현재 피로율 상태 저장 (Observation 및 디버깅용)
    env.motor_deg_state.fatigue_rate[env_ids] = total_fatigue_rate


def get_mechanical_health_index(env: ManagerBasedRLEnv, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    [Observation] 잔여 수명 비율 (SOH: State of Health) 조회.
    
    Returns:
        Tensor (N, J): 1.0 (Healthy) -> 0.0 (Broken/Fail)
    """
    if env_ids is None:
        env_ids = slice(None)

    # 초기 SOH (도메인 랜덤화에 의해 각기 다를 수 있음)
    initial_health = env.motor_deg_state.motor_health_capacity[env_ids]

    # 현재 SOH = 초기값 - 누적된 피로도
    current_health = initial_health - env.motor_deg_state.fatigue_index[env_ids]

    # SOH는 0.0 ~ 1.0 사이로 클램핑
    return torch.clamp(current_health, 0.0, 1.0)
