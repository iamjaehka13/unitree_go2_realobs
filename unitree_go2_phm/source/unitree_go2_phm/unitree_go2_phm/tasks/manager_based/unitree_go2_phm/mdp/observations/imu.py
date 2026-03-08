# unitree_go2_phm/mdp/observations/imu.py
#
# IMU observation helpers with optional PHM-aware drift modeling.
# Angular velocity and linear acceleration channels can inject
# temperature-conditioned bias while reusing shared kinematic helpers.
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Imu

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# [Architecture] Import Core Constants & Utils
from ...phm.utils import compute_kinematic_accel
try:
    from ...phm.constants import GRAVITY_VAL
except ImportError:
    GRAVITY_VAL = 9.81  # Fallback

# [Physics Constants for Sensor Model]
TEMP_CALIBRATION = 25.0       # [°C] 센서 캘리브레이션 기준 온도 (상온)
DRIFT_SENSITIVITY = 0.002     # [G/°C] 온도 1도 상승 당 바이어스 이동량 (MPU6050 Worst case 유사)
GYRO_DRIFT_SENSITIVITY = 0.0015  # [rad/s/°C] 온도 유도형 자이로 바이어스

# =============================================================================
# 1. Standard State Observations
# =============================================================================

def base_ang_vel_phm(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_imu")
) -> torch.Tensor:
    """[Sim-to-Real] 로봇 바디 프레임 기준 각속도 (Gyroscope)."""
    # Isaac Lab IMU sensor lookup.
    sensor: Imu = env.scene.sensors.get(sensor_cfg.name)
    if sensor is None:
        raise ValueError(f"[Config Error] Imu sensor '{sensor_cfg.name}' missing.")

    ang_vel = sensor.data.ang_vel_b[env_ids]

    # Temperature-induced gyroscope drift.
    if hasattr(env, "phm_state"):
        avg_temp = torch.mean(env.phm_state.coil_temp[env_ids], dim=1, keepdim=True)
        temp_delta = avg_temp - TEMP_CALIBRATION

        if isinstance(env_ids, slice):
            env_id_vals = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
        else:
            env_id_vals = (
                env_ids.to(device=env.device, dtype=torch.float32)
                if isinstance(env_ids, torch.Tensor)
                else torch.tensor(env_ids, device=env.device, dtype=torch.float32)
            )

        drift_dir = torch.stack(
            [
                torch.cos(env_id_vals * 0.37),
                torch.sin(env_id_vals * 0.61),
                torch.cos(env_id_vals * 1.07),
            ],
            dim=-1,
        )
        drift_dir = torch.nn.functional.normalize(drift_dir, p=2.0, dim=-1)
        gyro_sens = float(getattr(getattr(env, "cfg", None), "imu_gyro_drift_sensitivity", 0.0008))
        gyro_bias = drift_dir * temp_delta * gyro_sens
        ang_vel = ang_vel + gyro_bias

    return ang_vel

def projected_gravity_phm(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Control] 중력 벡터의 바디 프레임 투영 (Projected Gravity)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b[env_ids]

# =============================================================================
# 2. PHM Optimized Observations (Kinematic Acceleration)
# =============================================================================

def base_lin_accel_phm(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_imu"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [PHM Standardized] 중력을 제거한 '순수 동적 가속도(Dynamic Acceleration)' 관측.
    Returns:
        Tensor (N, 3): Unit in G-Force (1G = 9.81 m/s^2)
    """
    # 1. Physics Standardization (m/s^2)
    # utils.py가 IMU 바이어스와 중력 벡터 보정을 전담합니다.
    kinematic_accel_mps2 = compute_kinematic_accel(env, env_ids, sensor_cfg, asset_cfg)
    
    # 2. Scaling to G-Force (m/s^2 -> G)
    accel_g = kinematic_accel_mps2 / GRAVITY_VAL
    
    return accel_g.clone()

def base_lin_accel_phm_with_drift(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_imu"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    [Sensor Integrity] 온도 유도형 센서 편향(Temperature Drift)이 포함된 가속도.
    
    Physics:
        Observed = True_Accel + Bias(T) + Noise
        Bias(T) = Direction * (T_curr - T_calib) * Sensitivity
        
    Mechanism:
        - Bias Direction은 로봇(env_id)마다 고유하며 변하지 않는다고 가정합니다.
        - 온도가 오를수록 0점이 틀어지는 현상을 모사합니다.
        - RL 에이전트는 고온 상태에서 가속도계 신뢰도가 낮아짐을 학습해야 합니다.
    """
    # 1. 기본 가속도 계산 (Ideal + Random Noise included in sensor model)
    accel_g = base_lin_accel_phm(env, env_ids, sensor_cfg, asset_cfg)
    
    # 2. 열적 편향(Drift) 주입
    if hasattr(env, "phm_state"):
        # 로봇 모터들의 평균 온도를 '바디 온도'로 근사 (Heat Soak Effect)
        # shape: (N, 12) -> (N, 1)
        avg_temp = torch.mean(env.phm_state.coil_temp[env_ids], dim=1, keepdim=True)
        
        # 온도 편차 (Delta T)
        # 캘리브레이션 온도(25도)보다 높을수록 편차가 커짐
        temp_delta = avg_temp - TEMP_CALIBRATION
        
        # 3. 편향 방향 벡터 생성 (Deterministic per Env)
        # 각 로봇마다 고유한 결함 방향을 가짐 (Pseudo-random based on env_ids).
        # 별도의 State 저장 없이 env_id만으로 일관된 랜덤 방향을 만듭니다.
        # [Visualizing Vector Generation]
        # x, y, z 방향으로 서로 다른 주기의 사인파를 사용하여 3D 벡터 생성
        if isinstance(env_ids, slice):
            env_id_vals = torch.arange(env.num_envs, device=env.device, dtype=torch.float32)
        else:
            env_id_vals = env_ids.to(device=env.device, dtype=torch.float32) if isinstance(env_ids, torch.Tensor) else torch.tensor(env_ids, device=env.device, dtype=torch.float32)

        drift_dir_x = torch.sin(env_id_vals * 0.45)
        drift_dir_y = torch.cos(env_id_vals * 0.89)
        drift_dir_z = torch.sin(env_id_vals * 1.23)
        
        # (N, 3) 벡터 생성 및 단위 벡터화 (Normalize)
        drift_vec = torch.stack([drift_dir_x, drift_dir_y, drift_dir_z], dim=-1)
        drift_vec = torch.nn.functional.normalize(drift_vec, p=2.0, dim=-1)
        
        # 4. 최종 편향 값 계산
        # Bias = Direction * Delta_T * Sensitivity
        # 예: 65도(델타 40도) * 0.002 = 0.08G의 오차 발생
        accel_sens = float(getattr(getattr(env, "cfg", None), "imu_accel_drift_sensitivity", 0.0012))
        bias_drift = drift_vec * temp_delta * accel_sens
        
        # 5. 관측값 오염 (Corrupt Observation)
        accel_g += bias_drift
        
    return accel_g

def orientation_error_phm(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Energy Efficiency] 이상적 수평 자세 대비 현재 기울기 오차 (RMSE-like)."""
    asset: Articulation = env.scene[asset_cfg.name]
    # Projected Gravity의 XY 성분은 기울어질수록 커짐 (수직일 때 0, 0)
    gravity_vec = asset.data.projected_gravity_b[env_ids]
    return torch.norm(gravity_vec[:, :2], dim=-1, keepdim=True)

# =============================================================================
# 3. Debugging & PHM Analysis Tool
# =============================================================================

def debug_imu_precision(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice = slice(None),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_imu"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """[Debug] IMU 0점 조절 상태 확인 (System Integrity Check)."""
    if not bool(getattr(env, "enable_observation_debug_prints", False)):
        num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        return torch.zeros((num_envs, 0), device=env.device)

    step = getattr(env, "_sim_step_counter", getattr(env, "common_step_counter", 0))
    if step % 500 != 0:
        num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
        return torch.zeros((num_envs, 0), device=env.device)

    # 1. 드리프트가 포함된 현재 가속도 조회
    drifted_accel = base_lin_accel_phm_with_drift(env, env_ids, sensor_cfg, asset_cfg)
    
    # 2. 오차 크기 측정 (정지 상태 가정 시 0이어야 함)
    # 실제로는 움직이고 있으므로 절대적인 0점은 아니지만, Bias의 경향성을 볼 수 있음
    bias_magnitude = torch.norm(drifted_accel, dim=-1)
    avg_bias = torch.mean(bias_magnitude).item()
    max_bias = torch.max(bias_magnitude).item()
    
    # 3. 온도 상태 확인
    if hasattr(env, "phm_state"):
        avg_temp = torch.mean(env.phm_state.coil_temp[env_ids]).item()
        temp_status = f"Avg Temp: {avg_temp:.1f}°C"
    else:
        temp_status = "Temp: N/A"

    status = "[STABLE]" if max_bias < 0.1 else "[DRIFT DETECTED]"
    
    print(f"\n[PHM IMU INTEGRITY] Step {step} | {temp_status}")
    print(f" > Mean Accel Mag (Drift+Motion): {avg_bias:.4f} G")
    print(f" > Max Observed Drift Impact: {max_bias:.4f} G")
    print(f" > System Status: {status}")
    print("-" * 50 + "\n")
    
    num_envs = env.num_envs if isinstance(env_ids, slice) else len(env_ids)
    return torch.zeros((num_envs, 0), device=env.device)
