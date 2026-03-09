# =============================================================================
# unitree_go2_realobs/mdp/observations/observable_signals.py
# Raw observable-signal helpers for power, thermal, and voltage-budget channels.
# =============================================================================
# Description:
# Real-observable measurable signal layer.
#
# [Key Features]
# 1. SSOT Enforced: 전력 데이터를 Interface 적분값(avg_power_log)에서 직접 조회.
# 2. Risk-conditioned masking: Risk Factor에 따라 가용 전압을 예산(Budget) 형태로 제공.
# =============================================================================

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# [Core] Shared Utilities & Constants
from ...motor_deg.utils import safe_tensor
from ...motor_deg import constants as motor_deg_const

# =============================================================================
# 1. Component Loss Analysis (Data Fetchers)
# =============================================================================

def energy_consumption_raw(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    [Watts] 시스템 총 전력 소모량 조회 (SSOT: Interface-Computed).
    
    [Architecture Note]:
    이 함수는 더 이상 전력을 직접 계산하지 않습니다.
    'interface.py'의 물리 루프(Physics Loop)에서 고주파로 적분(Integration)된
    평균 전력 값(env.motor_deg_state.avg_power_log)을 반환합니다.
    """
    if isinstance(env_ids, slice):
        num_envs = env.num_envs
    else:
        num_envs = int(env_ids.numel()) if isinstance(env_ids, torch.Tensor) else len(env_ids)

    # 1. Safety Check: MotorDeg State 존재 여부 확인
    if not hasattr(env, "motor_deg_state") or not hasattr(env.motor_deg_state, "avg_power_log"):
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
        if joint_ids is None or isinstance(joint_ids, slice):
            num_joints = asset.data.joint_pos.shape[1]
        else:
            num_joints = len(joint_ids)
        return torch.zeros((num_envs, num_joints), device=env.device)

    # 2. Fetch Truth Data
    p_logged = env.motor_deg_state.avg_power_log[env_ids]
    
    return p_logged


# =============================================================================
# 2. Voltage Budget and Health Metrics
# =============================================================================

def available_voltage_budget(
    env: ManagerBasedEnv, 
    cutoff_voltage: float = 24.5
) -> torch.Tensor:
    """
    [Risk-Conditioned Observation] 위험 감수 계수 기반 가용 에너지 예산.
    
    Formula:
        Budget = (Current_Voltage - Cutoff) * Risk_Factor
    
    Interpretation:
        - Risk=1.0: 배터리 잔량을 있는 그대로 보여줌 (풀 파워 사용 허용).
        - Risk=0.0: 배터리가 많아도 '0'으로 보여줌 (최소 전력 모드 강제).
        - 에이전트는 이 'Budget'이 0이 되면 전압 컷오프(사망)가 임박했다고 착각하게 됨.
    """
    # 1. 현재 배터리 전압 조회
    # [Fix #8] Hidden-state budget 경로 기본값은 BMS 전압 예측값(bms_voltage_pred) 사용.
    # (brownout source는 env cfg에서 변경 가능)
    # 이전: battery_voltage(biased)를 사용하여 brownout/전략 관측 채널이 괴리됨.
    if not hasattr(env, "motor_deg_state"):
        return torch.zeros((env.num_envs, 1), device=env.device)

    if hasattr(env.motor_deg_state, "bms_voltage_pred"):
        current_v = env.motor_deg_state.bms_voltage_pred.unsqueeze(-1)
    elif hasattr(env.motor_deg_state, "battery_voltage"):
        current_v = env.motor_deg_state.battery_voltage.unsqueeze(-1)
    else:
        return torch.zeros((env.num_envs, 1), device=env.device)
    
    # 2. Risk Factor Command 조회
    # Command Manager에서 (N, 1) 형태의 명령을 가져옴. 
    # env_cfg.py에 'risk_factor' 커맨드가 등록되어 있어야 함.
    risk_cmd = env.command_manager.get_command("risk_factor")
    if risk_cmd is None:
        # 커맨드가 없으면 기본값 1.0 (Aggressive) 가정
        risk_factor = torch.ones((env.num_envs, 1), device=env.device)
    else:
        risk_factor = risk_cmd
    
    # 3. 물리적 가용 전압 (Headroom)
    # 실제 전압이 Cutoff보다 낮으면 0
    true_headroom = torch.clamp(current_v - cutoff_voltage, min=0.0)
    
    # 4. Risk-conditioned masking
    # Risk가 낮을수록 Headroom이 적은 것처럼 속임
    perceived_budget = true_headroom * risk_factor
    
    # 5. Normalization 
    # Max headroom = Full Voltage(33.6V) - Cutoff
    # 0.0 ~ 1.0 사이로 정규화하여 네트워크 입력 분포 안정화
    max_headroom = 33.6 - cutoff_voltage
    return torch.clamp(perceived_budget / max(max_headroom, 1e-6), 0.0, 1.0)


def available_voltage_budget_realobs(
    env: ManagerBasedEnv,
    cutoff_voltage: float = 24.5,
) -> torch.Tensor:
    """
    [Real-Observable Observation] Measured voltage headroom only.

    RealObs policy should not depend on hidden model-predicted channels such as
    `bms_voltage_pred`. Prefer measured sensor channels (`battery_voltage`).
    """
    if not hasattr(env, "motor_deg_state"):
        return torch.zeros((env.num_envs, 1), device=env.device)

    if hasattr(env.motor_deg_state, "battery_voltage"):
        current_v = env.motor_deg_state.battery_voltage.unsqueeze(-1)
    elif hasattr(env.motor_deg_state, "battery_voltage_true"):
        # Fallback for environments that only expose true-voltage channel.
        current_v = env.motor_deg_state.battery_voltage_true.unsqueeze(-1)
    else:
        return torch.zeros((env.num_envs, 1), device=env.device)

    headroom = torch.clamp(current_v - cutoff_voltage, min=0.0)
    max_headroom = 33.6 - cutoff_voltage
    return torch.clamp(headroom / max(max_headroom, 1e-6), 0.0, 1.0)


def thermal_stress_realobs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    warn_temp: float = 65.0,
    crit_temp: float = 70.0,
    coil_to_case_delta_c: float = 5.0,
) -> torch.Tensor:
    """
    [Real-Observable Observation] Case/housing-temperature stress proxy.

    Priority:
    1) explicit case/housing temperature tensor, if present
    2) coil temperature with fixed offset proxy (coil - delta)
    """
    if not hasattr(env, "motor_deg_state"):
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
        if joint_ids is None or isinstance(joint_ids, slice):
            num_joints = asset.data.joint_pos.shape[1]
        else:
            num_joints = len(joint_ids)
        return torch.zeros((env.num_envs, num_joints), device=env.device)

    deg_state = env.motor_deg_state
    temp = None
    for name in (
        "motor_case_temp",
        "case_temp",
        "motor_temp_case",
        "housing_temp",
        "motor_housing_temp",
    ):
        if hasattr(deg_state, name):
            val = getattr(deg_state, name)
            if isinstance(val, torch.Tensor):
                temp = val
                break

    if temp is None:
        if not hasattr(deg_state, "coil_temp"):
            asset: Articulation = env.scene[asset_cfg.name]
            return torch.zeros((env.num_envs, asset.data.joint_pos.shape[1]), device=env.device)
        temp = deg_state.coil_temp - float(coil_to_case_delta_c)

    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is None:
        joint_ids = slice(None)
    temp = temp[:, joint_ids]
    temp = torch.clamp(temp, min=motor_deg_const.T_AMB)

    denom = max(float(crit_temp) - float(warn_temp), 1e-6)
    stress = (temp - float(warn_temp)) / denom
    return safe_tensor(torch.clamp(stress, 0.0, 1.0), default_val=0.0)


def thermal_rate_realobs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rate_scale_c_per_s: float = 3.0,
    use_case_proxy: bool = True,
) -> torch.Tensor:
    """
    [Real-Observable Observation] Temperature-rate proxy (dT/dt).

    Priority:
    1) case/housing derivative channel (if available and use_case_proxy=True)
    2) coil derivative channel
    """
    if not hasattr(env, "motor_deg_state"):
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
        if joint_ids is None or isinstance(joint_ids, slice):
            num_joints = asset.data.joint_pos.shape[1]
        else:
            num_joints = len(joint_ids)
        return torch.zeros((env.num_envs, num_joints), device=env.device)

    deg_state = env.motor_deg_state
    rate = None
    if use_case_proxy:
        for name in ("case_temp_derivative", "motor_case_temp_derivative", "housing_temp_derivative"):
            if hasattr(deg_state, name):
                val = getattr(deg_state, name)
                if isinstance(val, torch.Tensor):
                    rate = val
                    break
    if rate is None:
        if hasattr(deg_state, "temp_derivative"):
            rate = deg_state.temp_derivative
        else:
            asset: Articulation = env.scene[asset_cfg.name]
            return torch.zeros((env.num_envs, asset.data.joint_pos.shape[1]), device=env.device)

    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is None:
        joint_ids = slice(None)
    rate = rate[:, joint_ids]

    denom = max(float(rate_scale_c_per_s), 1e-6)
    rate_norm = torch.clamp(rate / denom, -1.0, 1.0)
    return safe_tensor(rate_norm, default_val=0.0)


# =============================================================================
# 3. Advanced MotorDeg Metrics (Read-Only Logic)
# =============================================================================

def integrated_fatigue_score(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    fatigue_scale: float = 1.0,
    fatigue_exponent: float = 1.0, 
    **kwargs
) -> torch.Tensor:
    """
    [Observation Metric] 누적 피로도 점수 조회.
    """
    if not hasattr(env, "motor_deg_state"):
        num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        asset: Articulation = env.scene[asset_cfg.name]
        joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
        if joint_ids is None or isinstance(joint_ids, slice):
            num_joints = asset.data.joint_pos.shape[1]
        else:
            num_joints = len(joint_ids)
        return torch.zeros((num_envs, num_joints), device=env.device)

    # 1. Get Accumulated Fatigue from Single Source of Truth
    raw_fatigue = env.motor_deg_state.fatigue_index[env_ids]

    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is None:
        joint_ids = slice(None)
    raw_fatigue = raw_fatigue[:, joint_ids]

    # 2. Apply Observation Shaping
    score = fatigue_scale * torch.pow(raw_fatigue, fatigue_exponent)
    
    return safe_tensor(score, default_val=0.0)


def torque_saturation_error(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Safety] Saturation Error or Tracking Error Proxy."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is None:
        joint_ids = slice(None)
    
    applied = asset.data.applied_torque[env_ids][:, joint_ids]
    effort_limits = asset.data.joint_effort_limits[env_ids][:, joint_ids]

    effort_limits = torch.clamp(effort_limits, min=1.0)

    if hasattr(asset.data, "computed_torque") and asset.data.computed_torque is not None:
        computed = asset.data.computed_torque[env_ids][:, joint_ids]
        error = torch.abs(computed - applied)
        metric = error / effort_limits
    else:
        # Fallback: Tracking Error Proxy
        pos_target = asset.data.joint_pos_target[env_ids][:, joint_ids]
        pos_current = asset.data.joint_pos[env_ids][:, joint_ids]
        metric = torch.abs(pos_target - pos_current) * 10.0

    return metric


# =============================================================================
# 4. Long-Term Health Trend Observations (Fix #6, #12)
# =============================================================================

def degradation_slope(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    [MotorDeg Insight] 피로도 변화 기울기 (장기 버퍼 기반).
    
    LongTermHealthBuffer.get_degradation_slope()를 관측으로 노출.
    Returns:
        Tensor (N, J): 피로도 증가 속도. 값이 클수록 빠르게 열화 중.
    """
    if not hasattr(env, "motor_deg_long_term_buffer"):
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.zeros((env.num_envs, asset.data.joint_pos.shape[1]), device=env.device)
    
    slope = env.motor_deg_long_term_buffer.get_degradation_slope()
    
    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is not None and not isinstance(joint_ids, slice):
        slope = slope[:, joint_ids]
    
    return safe_tensor(slope, default_val=0.0)


def thermal_overload_duration_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 0.01,
) -> torch.Tensor:
    """
    [MotorDeg Insight] 누적 열 과부하 시간 관측.
    
    LongTermHealthBuffer.thermal_overload_duration을 관측으로 노출.
    Returns:
        Tensor (N, J): 스케일링된 누적 과열 시간 [s * scale].
    """
    if not hasattr(env, "motor_deg_long_term_buffer"):
        asset: Articulation = env.scene[asset_cfg.name]
        return torch.zeros((env.num_envs, asset.data.joint_pos.shape[1]), device=env.device)
    
    duration = env.motor_deg_long_term_buffer.thermal_overload_duration
    
    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    if joint_ids is not None and not isinstance(joint_ids, slice):
        duration = duration[:, joint_ids]
    
    return safe_tensor(duration * scale, default_val=0.0)


# =============================================================================
# [Debug] High-Performance MotorDeg Dashboard
# =============================================================================
def debug_motor_deg_energy_dashboard(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """[Dashboard] Fleet Stats & Accumulated Fatigue Check."""
    if not bool(getattr(env, "enable_observation_debug_prints", False)):
        return torch.zeros((env.num_envs, 0), device=env.device)

    step = getattr(env, "_sim_step_counter", getattr(env, "common_step_counter", 0))

    if step % 500 != 0:
        return torch.zeros((env.num_envs, 0), device=env.device)

    p_total = energy_consumption_raw(env, slice(None), asset_cfg)
    avg_power = torch.mean(torch.sum(p_total, dim=-1)).item()

    if hasattr(env, "motor_deg_state"):
        raw_fatigue = env.motor_deg_state.fatigue_index
        max_fatigue_val = torch.max(raw_fatigue).item()
        worst_env_idx = torch.argmax(torch.max(raw_fatigue, dim=-1)[0]).item()
        
        # [Debug] Voltage & Budget Check
        avg_v = torch.mean(env.motor_deg_state.battery_voltage).item()
        # Risk Factor Command Check
        risk_cmd = env.command_manager.get_command("risk_factor")
        avg_risk = torch.mean(risk_cmd).item() if risk_cmd is not None else 1.0
    else:
        max_fatigue_val = 0.0
        worst_env_idx = -1
        avg_v = 0.0
        avg_risk = 0.0

    sat_error = torque_saturation_error(env, slice(None), asset_cfg)
    avg_sat = torch.mean(sat_error).item()

    log_msg = (
        f"\n{'='*60}\n"
        f"[MotorDeg Dashboard] Step: {step}\n"
        f"------------------------------------------------------------\n"
        f"Power (Integrator)    | Avg: {avg_power:8.2f} W\n"
        f"Voltage (Avg)         | Val: {avg_v:8.2f} V (Risk: {avg_risk:.2f})\n"
        f"Accumulated Fatigue   | Max: {max_fatigue_val:8.4f} (Env {worst_env_idx})\n"
        f"Saturation Error      | Avg: {avg_sat:8.4f}\n"
        f"{'='*60}\n"
    )
    print(log_msg)

    return torch.zeros((env.num_envs, 0), device=env.device)
