# =============================================================================
# unitree_go2_realobs/mdp/rewards/degradation.py
# =============================================================================

from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# =============================================================================
# [Electrical Health] 에너지 효율 (Average Power 기반)
# =============================================================================
def electrical_energy_reward(
    env: ManagerBasedRLEnv, 
    std: float,
    allow_regen_bonus: bool = False,  
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [MotorDeg-Aware Energy Reward]
    소비 전력을 최소화하되, 불필요한 회생 제동 유도(Fighting)를 방지하는 보상 함수.
    
    Args:
        env: Isaac Lab 환경 객체
        std: Gaussian Kernel의 민감도 (표준편차)
        allow_regen_bonus: True일 경우 회생제동을 에너지 절감으로 인정 (권장하지 않음)
    
    Returns:
        reward: [num_envs] (0.0 ~ 1.0)
    """
    # 1. MotorDeg 데이터 유효성 검사 (Safety Guard)
    # env.motor_deg_state가 초기화되지 않았거나 로그가 없을 경우 0을 반환하여 학습 터짐 방지
    if not hasattr(env, "motor_deg_state") or not hasattr(env.motor_deg_state, "avg_power_log"):
        return torch.zeros(env.num_envs, device=env.device)
    
    # (num_envs, num_joints) - 관절별 평균 전력 [단위: W]
    # 이 데이터는 앞서 interface.py에서 계산된 물리적으로 정확한 값이 넘어와야 합니다.
    avg_power_per_joint = env.motor_deg_state.avg_power_log
    
    # 2. Net-Zero Exploit 방지 로직
    if not allow_regen_bonus:
        # [핵심] 관절 단위 Clamp
        # 발전(-W)하는 관절을 0으로 처리하여, 소모(+W)하는 관절의 비용을 상쇄하지 못하게 함.
        # 예: Joint A(+50W) + Joint B(-50W) 
        # -> 기존: 0W (완벽함) -> 꼼수 발생
        # -> 수정: 50W + 0W = 50W (페널티) -> 꼼수 차단
        consumed_power = torch.clamp(avg_power_per_joint, min=0.0)
    else:
        # 회생 제동을 허용하더라도, 실제 배터리 충전 효율(0.6) 등을 고려한 값이 들어와야 함
        consumed_power = avg_power_per_joint

    # 3. 로봇 전체 전력 합산 (Total Power Load)
    total_avg_power = torch.sum(consumed_power, dim=1)

    # 4. (Optional) 보상 함수의 안정성 확보
    # 만약 allow_regen_bonus=True 상태에서 전체 합이 음수(발전)가 되면
    # exp(-(-val)) = exp(+val)로 보상이 1.0을 초과하여 폭주할 수 있음.
    # 이를 방지하기 위해 최소 0.0으로 한 번 더 잠금 장치를 둡니다.
    total_avg_power = torch.clamp(total_avg_power, min=0.0)

    # 5. Gaussian Kernel 보상 계산
    # 전력 소모가 0에 수렴할수록 1.0 반환
    reward = torch.exp(-total_avg_power / (std**2))
    
    return reward

# =============================================================================
# [Mechanical Health] 베어링 수명 (Fatigue Rate 기반)
# =============================================================================
def bearing_life_reward(
    env: ManagerBasedRLEnv, 
    threshold: float = 1.0,
    scale_factor: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Longevity] 베어링 피로도 증가율 억제."""
    if not hasattr(env, "motor_deg_state") or not hasattr(env.motor_deg_state, "fatigue_rate"):
        return torch.zeros(env.num_envs, device=env.device)

    fatigue_rate = env.motor_deg_state.fatigue_rate
    max_rate, _ = torch.max(fatigue_rate, dim=1)

    violation = torch.relu(max_rate - threshold)
    reward = 1.0 / (1.0 + (violation * scale_factor))

    return reward

# =============================================================================
# [Thermal Health] 온도 예측 제어 (Look-ahead Safety)
# =============================================================================
def thermal_predictive_reward(
    env: ManagerBasedRLEnv, 
    std: float,
    limit_temp: float = 90.0,
    horizon_dt: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Safety] 온도 상승률을 고려한 미래 온도 예측 페널티."""
    if not hasattr(env, "motor_deg_state") or not hasattr(env.motor_deg_state, "temp_derivative"):
        return torch.zeros(env.num_envs, device=env.device)

    # 미래 온도 예측: T_future = T_now + (dT/dt * horizon)
    temps = env.motor_deg_state.coil_temp
    temp_dot = env.motor_deg_state.temp_derivative
    pred_temp = temps + (temp_dot * horizon_dt)
    
    # 임계값의 95% 지점부터 페널티 부여 시작
    threshold = limit_temp * 0.95
    violation = torch.clamp(pred_temp - threshold, min=0.0)
    
    # 최대 위반값과 평균 위반값을 조합하여 보상 산출
    max_viol, _ = torch.max(violation, dim=1)
    mean_viol = torch.mean(violation, dim=1)
    total_cost = max_viol + (0.5 * mean_viol)
    
    return torch.exp(-total_cost / (std**2))


def thermal_margin_reward_realobs(
    env: ManagerBasedRLEnv,
    std: float,
    limit_temp: float = 70.0,
    warn_temp: float | None = 65.0,
    coil_to_case_delta_c: float = 5.0,
    margin_clip: float = 1.0,
    alpha_mean: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """[RealObs Safety] Case/housing temperature reward aligned to warn-crit normalized stress."""
    if not hasattr(env, "motor_deg_state"):
        return torch.zeros(env.num_envs, device=env.device)

    deg_state = env.motor_deg_state
    temps = None
    for name in (
        "motor_case_temp",
        "case_temp",
        "motor_temp_case",
        "housing_temp",
        "motor_housing_temp",
    ):
        if hasattr(deg_state, name):
            v = getattr(deg_state, name)
            if isinstance(v, torch.Tensor):
                temps = v
                break

    if temps is None:
        if not hasattr(deg_state, "coil_temp"):
            return torch.zeros(env.num_envs, device=env.device)
        temps = deg_state.coil_temp - float(coil_to_case_delta_c)

    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is None:
        joint_ids = slice(None)
    temps = temps[:, joint_ids]

    # RealObs reward shares the same warn-crit coordinate system as thermal_stress_realobs.
    crit_temp = float(limit_temp)
    if warn_temp is None:
        warn_temp = crit_temp * 0.95
    else:
        warn_temp = float(warn_temp)
    warn_temp = min(warn_temp, crit_temp - 1e-6)

    denom = max(crit_temp - warn_temp, 1e-6)
    margin = torch.clamp((temps - warn_temp) / denom, min=0.0, max=float(margin_clip))
    max_margin = torch.max(margin, dim=1).values
    mean_margin = torch.mean(margin, dim=1)
    total_cost = max_margin + (float(alpha_mean) * mean_margin)
    return torch.exp(-total_cost / (std**2))

# =============================================================================
# [Actuator Integrity] 포화 및 스톨(Stall) 감지
# =============================================================================
def actuator_saturation_reward(
    env: ManagerBasedRLEnv, 
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Integrity] 토크 포화 및 저속 고토크(Stall) 상황 억제."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is None:
        joint_ids = slice(None)
    
    # [Fix #9] motor_deg_state.torque_saturation 사용 (전체 substep max-pooling 결과).
    # 이전: asset.data.applied_torque를 직접 읽어 마지막 substep 토크만 반영.
    # substep 1~2에서 순간 포화가 발생해도 보상 함수가 이를 감지하지 못했음.
    # interface.py가 매 substep마다 max-pooling하므로 이 값이 정확한 SSOT.
    if hasattr(env, "motor_deg_state") and hasattr(env.motor_deg_state, "torque_saturation"):
        saturation_ratio = env.motor_deg_state.torque_saturation
        if isinstance(joint_ids, slice):
            saturation_ratio = saturation_ratio[:, joint_ids]
        else:
            saturation_ratio = saturation_ratio[:, joint_ids]
    else:
        # Fallback: MotorDeg 미초기화 시 기존 방식 사용
        if hasattr(env, "_nominal_effort_limits") and env._nominal_effort_limits is not None:
            effort_limits = env._nominal_effort_limits[:, joint_ids]
        else:
            effort_limits = asset.data.joint_effort_limits[:, joint_ids]
        current_torque = torch.abs(asset.data.applied_torque[:, joint_ids])
        saturation_ratio = current_torque / (effort_limits + 1e-6)

    is_saturated = saturation_ratio > 0.99
    
    # 스톨(Stall) 상황: 토크는 센데 움직임이 없을 때 기계적 스트레스 최대
    joint_vel = torch.abs(asset.data.joint_vel[:, joint_ids])
    stall_severity = torch.clamp(0.5 - joint_vel, min=0.0)
    
    # 포화 상황에서 속도가 느릴수록 더 큰 페널티
    saturation_metric = is_saturated.float() * (1.0 + stall_severity)
    max_metric, _ = torch.max(saturation_metric, dim=1)
    
    return torch.exp(-max_metric / (std**2))
