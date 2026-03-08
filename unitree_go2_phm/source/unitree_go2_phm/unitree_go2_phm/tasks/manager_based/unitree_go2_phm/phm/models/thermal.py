# =============================================================================
# unitree_go2_phm/phm/models/thermal.py
# Thermal state update helpers for the Go2 PHM model.
# =============================================================================

from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Optional

# [Single Source of Truth] 물리 상수
# 불필요한 마찰 관련 상수를 제거했습니다. (Interface에서 계산됨)
from ..constants import (
    C_THERMAL_COIL,         # [J/K] coil thermal capacity
    C_THERMAL_CASE,         # [J/K] case thermal capacity
    K_COIL_TO_CASE,         # [W/K] coil->case thermal coupling
    K_CASE_COOLING,         # [W/K] case natural cooling
    K_CASE_WIND,            # [W/(K*rad/s)] case forced convection
    C_THERMAL,              # legacy single-node alias
    K_COOLING,              # legacy single-node alias
    K_WIND,                 # legacy single-node alias
    T_AMB,                  # [°C] ambient temperature
    TEMP_CRITICAL_THRESHOLD,# [°C] critical temperature
    EPS,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def update_motor_temperature(
    env: ManagerBasedRLEnv,
    dt: float,
    env_ids: Optional[torch.Tensor] = None,
    # [Interface Contract] 
    # 이 인자는 반드시 (전기적 손실 + 실제 마찰열)이 합산된 'Total Heat Source'여야 합니다.
    p_loss_watts: Optional[torch.Tensor] = None,
    # [Thermal Model v2]
    # Split heat sources:
    # - p_coil_watts: heat directly generated in winding/coils (e.g., copper loss)
    # - p_case_watts: heat generated near housing/driver (e.g., inverter/friction/core)
    p_coil_watts: Optional[torch.Tensor] = None,
    p_case_watts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    [PHM Core] 모터 열 적분기.
    
    Role:
        Interface에서 계산된 총 발열량(Q_in)을 받아 열평형 방정식에 따라 온도를 갱신합니다.
        내부에서 마찰열을 재계산하거나 공칭 손실을 차감하지 않습니다.
    
    Physics Model:
      - Preferred (v2): 2-node RC model (coil <-> case <-> ambient)
      - Fallback (legacy): 1-node coil model
        
    Args:
        p_loss_watts: total heat (legacy single-node path)
        p_coil_watts: coil-side heat source
        p_case_watts: case-side heat source
    """
    if env_ids is None:
        env_ids = slice(None)

    # Current thermal states
    coil_temp = env.phm_state.coil_temp[env_ids]

    # Joint velocity drives forced convection on the case node.
    joint_indices = getattr(env, "phm_joint_indices", slice(None))
    joint_vel = env.scene["robot"].data.joint_vel[env_ids][:, joint_indices]
    joint_vel_abs = torch.abs(joint_vel)

    # ------------------------------------------------------------------
    # Preferred path: 2-node RC model (coil <-> case <-> ambient)
    # ------------------------------------------------------------------
    if (p_coil_watts is not None) or (p_case_watts is not None):
        if p_coil_watts is None:
            p_coil_watts = torch.zeros_like(coil_temp)
        if p_case_watts is None:
            p_case_watts = torch.zeros_like(coil_temp)

        if hasattr(env.phm_state, "motor_case_temp"):
            case_temp = env.phm_state.motor_case_temp[env_ids]
        elif hasattr(env.phm_state, "case_temp"):
            case_temp = env.phm_state.case_temp[env_ids]
        else:
            # Conservative fallback if state was created without case channel.
            case_temp = torch.clamp(coil_temp - 5.0, min=T_AMB)

        p_coil_watts = torch.nan_to_num(p_coil_watts, nan=0.0, posinf=0.0, neginf=0.0)
        p_case_watts = torch.nan_to_num(p_case_watts, nan=0.0, posinf=0.0, neginf=0.0)
        case_temp = torch.nan_to_num(case_temp, nan=T_AMB).clamp(min=T_AMB, max=200.0)

        # Heat transfer: coil -> case.
        q_couple = K_COIL_TO_CASE * (coil_temp - case_temp)

        # Case cooling: ambient + velocity-dependent convection.
        k_case_effective = K_CASE_COOLING + (joint_vel_abs * K_CASE_WIND)
        q_case_cooling = k_case_effective * (case_temp - T_AMB)

        dcoil_dt = (p_coil_watts - q_couple) / (C_THERMAL_COIL + EPS)
        dcase_dt = (p_case_watts + q_couple - q_case_cooling) / (C_THERMAL_CASE + EPS)

        new_coil = torch.clamp(coil_temp + dcoil_dt * dt, min=T_AMB, max=200.0)
        new_case = torch.clamp(case_temp + dcase_dt * dt, min=T_AMB, max=200.0)

        if hasattr(env.phm_state, "motor_case_temp"):
            env.phm_state.motor_case_temp[env_ids] = new_case
        if hasattr(env.phm_state, "case_temp"):
            env.phm_state.case_temp[env_ids] = new_case
        if hasattr(env.phm_state, "case_temp_derivative"):
            safe_dt = dt if dt > 1e-6 else 1e-6
            env.phm_state.case_temp_derivative[env_ids] = (new_case - case_temp) / safe_dt

        return new_coil

    # ------------------------------------------------------------------
    # Fallback path: legacy single-node coil model
    # ------------------------------------------------------------------
    if p_loss_watts is None:
        total_heat_in = torch.zeros_like(coil_temp)
    else:
        total_heat_in = p_loss_watts

    k_effective = K_COOLING + (joint_vel_abs * K_WIND)
    t_delta = coil_temp - T_AMB
    p_cooling = k_effective * t_delta
    net_heat_flow = total_heat_in - p_cooling
    temp_rise = (net_heat_flow * dt) / (C_THERMAL + EPS)
    new_temp = torch.clamp(coil_temp + temp_rise, min=T_AMB, max=200.0)

    # Keep case proxy coherent even in legacy mode.
    if hasattr(env.phm_state, "motor_case_temp"):
        env.phm_state.motor_case_temp[env_ids] = torch.clamp(new_temp - 5.0, min=T_AMB)
    if hasattr(env.phm_state, "case_temp"):
        env.phm_state.case_temp[env_ids] = torch.clamp(new_temp - 5.0, min=T_AMB)
    if hasattr(env.phm_state, "case_temp_derivative"):
        env.phm_state.case_temp_derivative[env_ids] = 0.0

    return new_temp


def get_thermal_stress_index(env: ManagerBasedRLEnv, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    [Observation Helper] 열적 스트레스 지수 정규화 (0.0 ~ 1.0).
    """
    if env_ids is None:
        env_ids = slice(None)

    current_temp = env.phm_state.coil_temp[env_ids]
    
    # (현재온도 - 대기온도) / (임계온도 - 대기온도)
    denom = TEMP_CRITICAL_THRESHOLD - T_AMB + EPS
    stress = (current_temp - T_AMB) / denom
    
    return torch.clamp(stress, 0.0, 1.0)
