# unitree_go2_phm/mdp/observations/proprioception.py
#
# PHM-enhanced proprioceptive observation helpers.
# Joint velocity channels reuse the same measurement noise cached in `PHMState`
# so the actuator path and the policy observe the same corrupted sensor stream.

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

# [Isaac Lab Core]
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# [Fix #6] __all__을 명시하여 wildcard import 시 Isaac Lab 표준 함수
# (base_lin_vel, base_ang_vel, projected_gravity, last_action)를 덮어쓰지 않도록 방지.
# 이 모듈의 PHM-전용 함수만 export합니다.
__all__ = [
    "joint_pos_rel_phm",
    "joint_vel_phm",
    "joint_acc",
    "joint_pos_limit_normalized",
    "joint_torques_applied",
    "debug_proprioception_phm",
]

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
try:
    from ...phm.constants import (
        NOMINAL_VELOCITY,
        EPS,
        ACC_SCALE,
    )
except ImportError:
    NOMINAL_VELOCITY = 30.0
    EPS = 1e-6
    ACC_SCALE = 0.01


def joint_pos_rel_phm(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice = slice(None), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    [PHM Enhanced] 중앙 PHM 상태(phm_state)와 동기화된 관절 위치 관측.
    
    Physics:
        Measured_Pos = True_Pos + Saved_Offset + Saved_Noise
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    
    # 1. Ground Truth
    true_pos_abs = asset.data.joint_pos[env_ids][:, joint_ids]
    
    # 2. PHM State 기반 오차 주입 (SSOT)
    if hasattr(env, "phm_state") and hasattr(env.phm_state, "encoder_meas_pos"):
        measured_pos_abs = env.phm_state.encoder_meas_pos[env_ids][:, joint_ids]
    elif hasattr(env, "phm_state"):
        cached_offset = env.phm_state.encoder_offset[env_ids]
        cached_noise = env.phm_state.encoder_noise[env_ids]
        measured_pos_abs = true_pos_abs + cached_offset[:, joint_ids] + cached_noise[:, joint_ids]
    else:
        measured_pos_abs = true_pos_abs

    # 3. Relative conversion
    default_pos = asset.data.default_joint_pos[env_ids][:, joint_ids]
    
    return measured_pos_abs - default_pos


def joint_vel_phm(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice = slice(None), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    [PHM Enhanced] 관절 각속도 (Normalized).
    
    Actuator와 동일한 속도 노이즈(D-term noise)를 적용하는 SSOT 경로.
    """
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    true_vel = asset.data.joint_vel[env_ids][:, joint_ids]
    
    if hasattr(env, "phm_state") and hasattr(env.phm_state, "encoder_meas_vel"):
        measured_vel = env.phm_state.encoder_meas_vel[env_ids][:, joint_ids]
    elif hasattr(env, "phm_state"):
        # Actuator가 보는 것과 동일한 속도 노이즈 주입
        vel_noise = getattr(env.phm_state, "encoder_vel_noise", 0.0)
        if isinstance(vel_noise, torch.Tensor):
            vel_noise = vel_noise[env_ids][:, joint_ids]
            
        measured_vel = true_vel + vel_noise
    else:
        measured_vel = true_vel
        
    return measured_vel / NOMINAL_VELOCITY


def joint_acc(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice = slice(None), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """[PHM Critical] 관절 각가속도 (Normalized by ACC_SCALE)."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    return asset.data.joint_acc[env_ids][:, joint_ids] * ACC_SCALE


def joint_pos_limit_normalized(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice = slice(None), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """[Safety] 관절 가동 범위 정규화 (-1.0 ~ 1.0)."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    
    # Safety Layer는 보통 True State보다는 센서 값(Measured)을 기준으로 판단하지만,
    # 학습 안정성을 위해 여기서는 True Pos를 유지하는 것이 일반적입니다.
    pos = asset.data.joint_pos[env_ids][:, joint_ids]
    limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
    
    lower = limits[..., 0]
    upper = limits[..., 1]
    range_span = (upper - lower) + EPS
    
    normalized_pos = 2.0 * (pos - lower) / range_span - 1.0
    return torch.clamp(normalized_pos, -1.0, 1.0)


def joint_torques_applied(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice = slice(None), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """[PHM Critical] 실제 인가된 토크 / Effort Limits."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        joint_ids = slice(None)
    
    raw_torque = asset.data.applied_torque[env_ids][:, joint_ids]
    effort_limits = asset.data.joint_effort_limits[env_ids][:, joint_ids] + EPS
    
    return raw_torque / effort_limits


# =============================================================================
# [Debug] Real-time PHM Monitoring
# =============================================================================
def debug_proprioception_phm(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice = slice(None), asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    [PHM Debug Dashboard]
    SSOT 검증: phm_state의 노이즈가 실제로 관측값에 반영되었는지 육안 확인 가능.
    """
    if not bool(getattr(env, "enable_observation_debug_prints", False)):
        num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        return torch.zeros((num_envs, 0), device=env.device)

    step = getattr(env, "_sim_step_counter", getattr(env, "common_step_counter", 0))
    if step % 500 != 0:
        num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        return torch.zeros((num_envs, 0), device=env.device)

    asset = env.scene[asset_cfg.name]
    
    # 0번 Env 데이터
    true_pos = asset.data.joint_pos[0]
    true_vel = asset.data.joint_vel[0]
    
    # PHM 상태 확인
    offset_val = torch.zeros_like(true_pos)
    pos_noise = torch.zeros_like(true_pos)
    vel_noise = torch.zeros_like(true_vel)
    
    if hasattr(env, "phm_state"):
        offset_val = env.phm_state.encoder_offset[0]
        pos_noise = env.phm_state.encoder_noise[0]
        vel_noise = getattr(env.phm_state, "encoder_vel_noise", torch.zeros_like(true_vel))[0]
    
    measured_pos = true_pos + offset_val + pos_noise
    measured_vel = true_vel + vel_noise

    joint_names = asset.joint_names if hasattr(asset, "joint_names") else [f"J{i}" for i in range(len(true_pos))]
    
    print(f"\n[Isaac Lab PHM Proprioception] Step: {step}")
    print(f"{'Joint':<8} | {'T_Pos':<8} {'M_Pos':<8} {'Diff':<8} | {'T_Vel':<8} {'M_Vel':<8} {'V_Diff':<8}")
    print("-" * 80)
    
    for i in range(len(joint_names)):
        print(f"{joint_names[i]:<8} | {true_pos[i]:8.4f} {measured_pos[i]:8.4f} {pos_noise[i]:8.4f} | {true_vel[i]:8.4f} {measured_vel[i]:8.4f} {vel_noise[i]:8.4f}")
            
    print("-" * 80 + "\n")
    
    num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
    return torch.zeros((num_envs, 0), device=env.device)
