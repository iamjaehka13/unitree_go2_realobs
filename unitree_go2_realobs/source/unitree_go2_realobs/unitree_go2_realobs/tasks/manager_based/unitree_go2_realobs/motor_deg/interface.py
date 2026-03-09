# =============================================================================
# unitree_go2_realobs/motor_deg/interface.py
# MotorDeg reset / dynamics / fault-sampling helpers shared by the Go2 tasks.
# =============================================================================
from __future__ import annotations

import logging
import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.managers import SceneEntityCfg

# [Relative Import]
from .state import MotorDegState
from .utils import (
    compute_kinematic_accel,
    compute_component_losses,
    compute_battery_voltage,
    compute_regenerative_efficiency
)

# [Buffers]
from .buffers.long_term import LongTermHealthBuffer
from .buffers.short_term import ShortTermHealthBuffer
from .models.thermal import update_motor_temperature
from .models.degradation import update_fatigue_index

# [Constants] 중앙 관리되는 상수 직접 호출
from .constants import (
    T_AMB,
    NUM_MOTORS,
    GRAVITY_VAL,
    TEMP_WARN_THRESHOLD,
    TEMP_CRITICAL_THRESHOLD,
    GEAR_RATIO,
    KT_NOMINAL,
    EPS,
    # [Step 2] Stall Constants
    RATED_LOAD_FACTOR,
    MIN_STALL_VELOCITY,
    # [Step 3] Friction & Wear Constants
    B_VISCOUS,
    WEAR_FRICTION_GAIN,
    STICTION_NOMINAL,
    STICTION_WEAR_FACTOR,
    JOINT_NAMES,
    FRICTION_HEAT_EFF,
    BATTERY_CAPACITY_J,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


_MOTOR_DEG_LOG_FLAGS = {
    "friction_source_read_failed": False,
    "friction_source_reset_failed": False,
    "thermal_reset_params_read_failed": False,
    "fault_mode_invalid_warned": False,
    "fault_fixed_id_invalid_warned": False,
    "fault_pair_uniform_unsupported_warned": False,
    "fault_focus_prob_invalid_warned": False,
    "fault_focus_invalid_entry_warned": False,
    "fault_focus_empty_warned": False,
    "fault_pair_target_weights_invalid_warned": False,
    "fault_pair_prob_range_invalid_warned": False,
    "fault_pair_weighted_requires_pair_uniform_warned": False,
    "fault_pair_adaptive_requires_weighted_warned": False,
    "fault_pair_adaptive_requires_pair_uniform_warned": False,
    "forced_scenario_invalid_warned": False,
    "forced_scenario_info_logged": False,
}

_FAULT_MIRROR_PAIRS_12 = (
    (0, 3),
    (1, 4),
    (2, 5),
    (6, 9),
    (7, 10),
    (8, 11),
)


def thermal_termination_params_from_cfg(
    cfg_obj,
    default_threshold_temp: float = 90.0,
    default_use_case_proxy: bool = False,
    default_coil_to_case_delta_c: float = 5.0,
) -> tuple[float | None, bool, float]:
    """
    Resolve thermal termination params from cfg.

    Returns:
      (threshold_temp_or_none, use_case_proxy, coil_to_case_delta_c)
    """
    threshold_temp: float | None = None
    use_case_proxy = bool(default_use_case_proxy)
    coil_to_case_delta_c = float(default_coil_to_case_delta_c)

    term_cfg = getattr(cfg_obj, "terminations", None)
    thermal_failure = getattr(term_cfg, "thermal_failure", None) if term_cfg is not None else None
    params = getattr(thermal_failure, "params", None)
    if isinstance(params, dict):
        threshold_temp = float(params.get("threshold_temp", float(default_threshold_temp)))
        use_case_proxy = bool(params.get("use_case_proxy", use_case_proxy))
        coil_to_case_delta_c = float(params.get("coil_to_case_delta_c", coil_to_case_delta_c))

    return threshold_temp, use_case_proxy, coil_to_case_delta_c


def case_proxy_safe_coil_max_for_reset(
    threshold_temp: float,
    coil_to_case_delta_c: float,
    *,
    case_delta_low_c: float = 3.5,
    immediate_term_margin_c: float = 0.5,
    proxy_margin_c: float = 1.0,
    ambient_temp_c: float = T_AMB,
    min_coil_above_ambient_c: float = 8.0,
) -> float:
    """
    Compute a conservative coil-temperature upper bound for reset-time sampling
    under case-proxy thermal termination.
    """
    safe_coil_max_from_case = float(threshold_temp) + float(case_delta_low_c) - float(immediate_term_margin_c)
    safe_coil_max_from_proxy = float(threshold_temp) + float(coil_to_case_delta_c) - float(proxy_margin_c)
    return max(
        float(ambient_temp_c) + float(min_coil_above_ambient_c),
        min(safe_coil_max_from_case, safe_coil_max_from_proxy),
    )


def _quantize_channel(x: torch.Tensor, step: float) -> torch.Tensor:
    """Quantize a sensor-like channel with fixed step size."""
    if step <= 0.0:
        return x
    return torch.round(x / step) * step


# =============================================================================
# MotorDeg 시스템 초기화
# =============================================================================
def init_motor_deg_interface(env: ManagerBasedRLEnv):
    """
    [Isaac Lab 2.1] MotorDeg 시스템 초기화.
    필수 State와 Buffer만 신속하게 생성합니다.
    """
    num_envs = env.num_envs
    device = env.device

    # 1. MotorDeg State 생성 (Core)
    # MotorDegState.__init__() creates all fields (friction_power, stall_timer, etc.)
    env.motor_deg_state = MotorDegState(num_envs, NUM_MOTORS, device)

    # 2. IMU Sensor Linking
    if hasattr(env.scene, "sensors") and "base_imu" in env.scene.sensors:
        env.motor_deg_imu_sensor = env.scene.sensors["base_imu"]
        env.motor_deg_imu_cfg = SceneEntityCfg("base_imu")
    else:
        env.motor_deg_imu_sensor = None
        env.motor_deg_imu_cfg = None

    # 5. Robot Config & Indexing
    robot: Articulation = env.scene["robot"]
    env.motor_deg_robot_cfg = SceneEntityCfg("robot")

    joint_indices, _ = robot.find_joints(JOINT_NAMES)
    env.motor_deg_joint_indices = joint_indices

    # 6. Optimization Constants
    env.motor_deg_const_gravity = torch.tensor(GRAVITY_VAL, device=device)

    # 7. Initialize Buffers
    env.motor_deg_long_term_buffer = LongTermHealthBuffer(
        num_envs=num_envs,
        num_joints=NUM_MOTORS,
        device=device
    )

    env.motor_deg_short_term_buffer = ShortTermHealthBuffer(
        env=env,
        num_envs=num_envs,
        device=device,
        robot_cfg=env.motor_deg_robot_cfg,
        contact_sensor_cfg=SceneEntityCfg("contact_forces"),
        joint_indices=env.motor_deg_joint_indices
    )

    # 8. External Sync (User Data)
    # [Fix] Isaac Lab 표준: env.data 대신 env.extras 사용
    if "joint_temp" not in env.extras:
        env.extras["joint_temp"] = env.motor_deg_state.coil_temp
    if hasattr(env.motor_deg_state, "motor_case_temp") and ("joint_case_temp" not in env.extras):
        env.extras["joint_case_temp"] = env.motor_deg_state.motor_case_temp


# =============================================================================
# 스텝별 메트릭 초기화 (Accumulator Reset)
# =============================================================================
def clear_motor_deg_step_metrics(env: ManagerBasedRLEnv):
    """
    [Critical Logic] RL Control Step 시작 전 호출.
    
    이전 Decimation Loop에서 누적된 에너지 및 전력 로그를 0으로 초기화합니다.
    """
    if hasattr(env.motor_deg_state, "step_energy_log"):
        env.motor_deg_state.step_energy_log.zero_()
        
    if hasattr(env.motor_deg_state, "avg_power_log"):
        env.motor_deg_state.avg_power_log.zero_()
        
    if hasattr(env.motor_deg_state, "torque_saturation"):
        env.motor_deg_state.torque_saturation.zero_()

    if hasattr(env.motor_deg_state, "jitter_intensity"):
        env.motor_deg_state.jitter_intensity.zero_()


# =============================================================================
# 센서 노이즈 갱신 (Static Noise Paradox 해결)
# =============================================================================
def refresh_motor_deg_sensors(env: ManagerBasedRLEnv, env_ids: Optional[torch.Tensor] = None):
    """
    매 제어 스텝마다 센서 노이즈를 재샘플링 (White Noise).
    만약 이를 수행하지 않으면 노이즈가 Static Bias(영구적 오차)가 되어
    RL 에이전트가 잘못된 패턴을 학습하게 됩니다.

    Args:
        env_ids: 특정 환경만 갱신할 경우 지정. None이면 전체 환경 갱신.
    """
    # 1. Position Noise (Encoder Error)
    if hasattr(env.motor_deg_state, "encoder_noise"):
        # Gaussian Noise Resampling (Mean=0, Std=0.01 rad)
        noise_std_pos = float(getattr(getattr(env, "cfg", None), "encoder_pos_noise_std_rad", 0.005))
        if env_ids is None:
            env.motor_deg_state.encoder_noise.normal_(mean=0.0, std=noise_std_pos)
        else:
            env.motor_deg_state.encoder_noise[env_ids] = torch.randn_like(
                env.motor_deg_state.encoder_noise[env_ids]
            ) * noise_std_pos

    # 2. Velocity Noise Injection (D-term Noise) [NEW]
    # 속도 노이즈는 미분 특성상 위치 노이즈보다 주파수가 높고 큼 (예: 0.05 ~ 0.1 rad/s)
    if hasattr(env.motor_deg_state, "encoder_vel_noise"):
        noise_std_vel = float(getattr(getattr(env, "cfg", None), "encoder_vel_noise_std_rads", 0.03))
        if env_ids is None:
            env.motor_deg_state.encoder_vel_noise.normal_(mean=0.0, std=noise_std_vel)
        else:
            env.motor_deg_state.encoder_vel_noise[env_ids] = torch.randn_like(
                env.motor_deg_state.encoder_vel_noise[env_ids]
            ) * noise_std_vel

    # 3. Sample-and-hold measurement update (sensor latency / packet staleness model)
    if hasattr(env.motor_deg_state, "encoder_meas_pos") and hasattr(env.motor_deg_state, "encoder_meas_vel"):
        ids = slice(None) if env_ids is None else env_ids
        robot: Articulation = env.scene["robot"]
        joint_idx = getattr(env, "motor_deg_joint_indices", slice(None))

        true_pos = robot.data.joint_pos[ids][:, joint_idx]
        true_vel = robot.data.joint_vel[ids][:, joint_idx]
        offset = env.motor_deg_state.encoder_offset[ids]
        pos_noise = env.motor_deg_state.encoder_noise[ids]
        vel_noise = env.motor_deg_state.encoder_vel_noise[ids]

        proposed_pos = true_pos + offset + pos_noise
        proposed_vel = true_vel + vel_noise

        hold_prob = float(getattr(getattr(env, "cfg", None), "encoder_sample_hold_prob", 0.0))
        num_rows = proposed_pos.shape[0]
        if hold_prob > 0.0:
            hold_mask = (torch.rand((num_rows, 1), device=env.device) < hold_prob)
        else:
            hold_mask = torch.zeros((num_rows, 1), dtype=torch.bool, device=env.device)

        prev_pos = env.motor_deg_state.encoder_meas_pos[ids]
        prev_vel = env.motor_deg_state.encoder_meas_vel[ids]
        env.motor_deg_state.encoder_meas_pos[ids] = torch.where(hold_mask, prev_pos, proposed_pos)
        env.motor_deg_state.encoder_meas_vel[ids] = torch.where(hold_mask, prev_vel, proposed_vel)

        if hasattr(env.motor_deg_state, "encoder_hold_flag"):
            env.motor_deg_state.encoder_hold_flag[ids] = hold_mask.squeeze(-1).float()


# =============================================================================
# MotorDeg 동역학 업데이트 (Physics Step 직후 호출)
# =============================================================================
def update_motor_deg_dynamics(env: ManagerBasedRLEnv, dt: float, env_ids: Optional[torch.Tensor] = None):
    """
    물리 상태 -> 피로도 -> 마찰전력 -> 열/에너지/배터리 업데이트.
    물리 루프(Fast Loop) 내에서 호출됩니다.
    
    [Fix Details]:
    1. Input Sanitization: 물리 엔진에서 오는 Torque/Vel에 NaN/Inf가 있어도 0으로 치환.
    2. Output Clamping: 전력, 온도 등 계산 결과가 물리적 한계를 넘지 않도록 강제 제한.
    3. Safe Logic: 0으로 나누기 등 수학적 오류 원천 차단.
    """
    if env_ids is None:
        env_ids = slice(None)

    robot: Articulation = env.scene["robot"]

    # -------------------------------------------------------------------------
    # 1. 물리 데이터 추출 & 세탁 (Sanitization)
    # -------------------------------------------------------------------------
    # [Critical Fix] 물리 엔진이 불안정할 때 NaN이나 1e38 같은 값이 들어올 수 있음 -> 0으로 뭉갬
    raw_torque = robot.data.applied_torque[env_ids][:, env.motor_deg_joint_indices]
    raw_vel = robot.data.joint_vel[env_ids][:, env.motor_deg_joint_indices]

    applied_torque = torch.nan_to_num(raw_torque, nan=0.0, posinf=0.0, neginf=0.0)
    joint_vel = torch.nan_to_num(raw_vel, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 너무 큰 값은 물리적으로 말이 안 되므로 컷 (Safety Clamp)
    # Go2 Max Torque ~24Nm, Velocity ~30 rad/s
    applied_torque = torch.clamp(applied_torque, -50.0, 50.0)
    joint_vel = torch.clamp(joint_vel, -100.0, 100.0)

    env.motor_deg_state.applied_torque[env_ids] = applied_torque

    # -------------------------------------------------------------------------
    # 2. 마모(Degradation) 상태 업데이트
    # -------------------------------------------------------------------------
    # 내부 로직 보호를 위해 try-except 처럼 동작하도록 모델 함수들도 점검 필요하지만,
    # 일단 여기서 호출 전후로 보호.
    update_fatigue_index(env, dt, env_ids)
    
    # 혹시 모를 NaN 방지
    env.motor_deg_state.fatigue_index[env_ids] = torch.nan_to_num(
        env.motor_deg_state.fatigue_index[env_ids], nan=0.0
    ).clamp(min=0.0, max=10.0)

    # -------------------------------------------------------------------------
    # 3. 진동 계산 (Safe Norm)
    # -------------------------------------------------------------------------
    if env.motor_deg_imu_sensor is not None:
        kinematic_accel = compute_kinematic_accel(
            env,
            env_ids,
            env.motor_deg_imu_cfg,
            env.motor_deg_robot_cfg
        )
        # NaN 방지
        kinematic_accel = torch.nan_to_num(kinematic_accel, nan=0.0)
        
        # 중력 가속도가 0일 리는 없지만 안전장치
        g_val = env.motor_deg_const_gravity if env.motor_deg_const_gravity > 1e-6 else 9.81
        vibration = (torch.norm(kinematic_accel, dim=-1, keepdim=True) / g_val).squeeze(-1)
        
        env.motor_deg_state.vibration_g[env_ids] = torch.clamp(vibration, 0.0, 100.0)

    # -------------------------------------------------------------------------
    # 4. [CRITICAL] 마찰 전력(Friction Power) 계산
    # -------------------------------------------------------------------------
    fatigue = env.motor_deg_state.fatigue_index[env_ids] # 위에서 이미 clamp 함
    
    friction_bias_full = getattr(env.motor_deg_state, "friction_bias", None)
    if friction_bias_full is not None:
        friction_bias = friction_bias_full[env_ids]
    else:
        friction_bias = torch.ones_like(fatigue)
    friction_bias = torch.nan_to_num(friction_bias, nan=1.0).clamp(0.5, 2.0)

    friction_source = None

    if hasattr(robot.data, "joint_friction_coeff"):
        t = robot.data.joint_friction_coeff
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            friction_source = t

    if friction_source is None and hasattr(robot.data, "default_joint_friction_coeff"):
        t = robot.data.default_joint_friction_coeff
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            friction_source = t

    base_friction_torque = None

    if friction_source is not None:
        try:
            num_source_envs, num_source_joints = friction_source.shape[:2]

            if isinstance(env_ids, slice) or env_ids.max().item() < num_source_envs:

                if isinstance(env.motor_deg_joint_indices, slice):
                    base_friction_torque = friction_source[env_ids]
                else:
                    safe_indices = [
                        i for i in env.motor_deg_joint_indices
                        if 0 <= i < num_source_joints
                    ]

                    if len(safe_indices) == len(env.motor_deg_joint_indices):
                        base_friction_torque = friction_source[env_ids][:, safe_indices]
        except Exception as err:
            if not _MOTOR_DEG_LOG_FLAGS["friction_source_read_failed"]:
                logging.warning(
                    "[MotorDeg] Failed to read runtime friction coefficients; "
                    "falling back to nominal stiction. error=%s",
                    err,
                )
                _MOTOR_DEG_LOG_FLAGS["friction_source_read_failed"] = True
            base_friction_torque = None

    if base_friction_torque is None:
        if isinstance(env.motor_deg_joint_indices, slice):
            target_dim = env.scene["robot"].num_joints
        else:
            target_dim = len(env.motor_deg_joint_indices)

        if isinstance(env_ids, slice):
            num_rows = env.num_envs
        else:
            num_rows = len(env_ids)

        base_friction_torque = torch.full(
            (num_rows, target_dim),
            STICTION_NOMINAL,
            device=env.device
        )

    # [Fix] base_friction_torque를 MotorDegState에 캐싱 (SSOT: env.py에서도 동일 소스 참조)
    if hasattr(env.motor_deg_state, "base_friction_torque"):
        env.motor_deg_state.base_friction_torque[env_ids] = base_friction_torque

    # 계산 폭주 방지
    viscous_coeff = B_VISCOUS * (1.0 + fatigue * WEAR_FRICTION_GAIN) * friction_bias
    viscous_loss_torque = viscous_coeff * torch.abs(joint_vel)
    
    stiction_val = base_friction_torque * (1.0 + fatigue * STICTION_WEAR_FACTOR) * friction_bias
    
    total_friction_torque = viscous_loss_torque + stiction_val
    real_friction_power = total_friction_torque * torch.abs(joint_vel)
    
    # [Safety] 마찰 전력이 무한대로 발산하지 않도록 제한 (예: 모터당 500W 이상은 비정상)
    real_friction_power = torch.clamp(real_friction_power, 0.0, 2000.0)
    env.motor_deg_state.friction_power[env_ids] = real_friction_power

    # -------------------------------------------------------------------------
    # 5. 전기적/열적 손실 통합 계산
    # -------------------------------------------------------------------------
    current_temp = env.motor_deg_state.coil_temp[env_ids]
    # 온도도 NaN이면 실온으로 초기화
    current_temp = torch.nan_to_num(current_temp, nan=T_AMB).clamp(min=-50.0, max=300.0)
    
    delta_t = current_temp - T_AMB
    env.motor_deg_state.delta_t[env_ids] = delta_t

    current_est = torch.abs(applied_torque) / (KT_NOMINAL * GEAR_RATIO + EPS)
    env.motor_deg_state.real_current[env_ids] = current_est

    # 외부 함수 호출 (NaN 위험 구역)
    p_copper, p_inverter, p_mech_total = compute_component_losses(
        torque=applied_torque,
        velocity=joint_vel,
        temp=current_temp,
        external_friction_power=real_friction_power 
    )
    
    # 결과값 세탁
    p_copper = torch.nan_to_num(p_copper, nan=0.0).clamp(min=0.0, max=5000.0)
    p_inverter = torch.nan_to_num(p_inverter, nan=0.0).clamp(min=0.0, max=5000.0)
    p_mech_total = torch.nan_to_num(p_mech_total, nan=0.0).clamp(min=0.0, max=5000.0)

    # 1. 순수 기계적 출력 계산 (마찰 전력 제외)
    # [Fix] applied_torque * joint_vel은 마찰 토크를 포함한 총 축 출력이므로
    # p_mech_total에 이미 포함된 real_friction_power를 빼서 이중 계상 방지.
    # 모터 에너지 흐름: 전기입력 = 손실(copper+inverter+mech) + 순수 기계출력
    gross_mechanical_power = applied_torque * joint_vel
    net_mechanical_power = gross_mechanical_power - real_friction_power
    
    # 2. 회생 효율 적용 (속도 의존 Gaussian 곡선)
    # [Fix] 고정값 0.6 대신 속도에 따른 물리적 회생 효율 모델 사용.
    # 저속/고속에서 효율이 떨어지고 최적 속도 부근에서 최대 효율.
    regen_eff = compute_regenerative_efficiency(joint_vel)
    mechanical_load = torch.where(
        net_mechanical_power < 0,
        net_mechanical_power * regen_eff,
        net_mechanical_power
    )

    # 3. 총 전력 부하
    # = 전기적 손실 + 기계적 손실(마찰+코어) + 순수 기계출력(회생 반영)
    joint_power_load = p_copper + p_inverter + p_mech_total + mechanical_load
    joint_power_load = torch.nan_to_num(joint_power_load, nan=0.0).clamp(-5000.0, 5000.0) # Safety

    # [Thermal Source]
    # Split heat by physical location (2-node model):
    # - Coil node: copper loss (winding I^2R)
    # - Case node: inverter + friction + core + non-recovered regen
    # This improves RealObs consistency because case temperature is dynamic.
    # [Fix] 회생 모드에서 회수되지 못한 에너지는 열로 변환.
    regen_heat_loss = torch.where(
        net_mechanical_power < 0,
        (1.0 - regen_eff) * torch.abs(net_mechanical_power),
        torch.zeros_like(net_mechanical_power)
    )
    # [Fix] FRICTION_HEAT_EFF 적용: p_mech_total 내 real_friction_power에 마찰→열 변환 효율 반영.
    # p_mech_total = real_friction_power + p_core이므로,
    # 마찰열 = real_friction_power * FRICTION_HEAT_EFF + p_core (코어 손실은 100% 열)
    friction_heat = real_friction_power * FRICTION_HEAT_EFF
    p_core_only = torch.clamp(p_mech_total - real_friction_power, min=0.0)
    coil_heat_watts = p_copper
    case_heat_watts = p_inverter + friction_heat + p_core_only + regen_heat_loss
    total_heat_watts = coil_heat_watts + case_heat_watts

    # [1] 순간 전력
    if hasattr(env.motor_deg_state, "instant_power"):
        env.motor_deg_state.instant_power[env_ids] = joint_power_load

    # [2] 평균 전력 누적
    if hasattr(env.motor_deg_state, "avg_power_log"):
        if hasattr(env, "step_dt"):
            control_dt = env.step_dt
        else:
            control_dt = dt * getattr(env.cfg, "decimation", 4)
        
        # dt가 0이면 1로 바꿔서 나눗셈 방지
        control_dt = control_dt if control_dt > 1e-6 else 0.02
        ratio = dt / control_dt
        
        env.motor_deg_state.avg_power_log[env_ids] += joint_power_load * ratio

    # 에너지 적분
    total_power_env = torch.sum(joint_power_load, dim=1, keepdim=True)
    step_energy = total_power_env * dt
    env.motor_deg_state.cumulative_energy[env_ids] += step_energy.squeeze(-1)

    # Reward Source
    if hasattr(env.motor_deg_state, "step_energy_log"):
        env.motor_deg_state.step_energy_log[env_ids] += step_energy.squeeze(-1)
        
    # A. SOC Update
    # [Fix] soc_drop을 min=0으로 클램프하여 회생제동(음의 에너지)이 SOC를 증가시키는 것을 방지.
    # 보상 함수(electrical_energy_reward)는 회생 이득을 차단하지만,
    # SOC 증가 → voltage_budget 관측값 상승으로 인한 간접 인센티브 모순을 제거.
    soc_drop = torch.clamp(step_energy.squeeze(-1) / BATTERY_CAPACITY_J, min=0.0)
    if hasattr(env.motor_deg_state, "soc"):
        env.motor_deg_state.soc[env_ids] = torch.clamp(
            env.motor_deg_state.soc[env_ids] - soc_drop, 
            min=0.0, max=1.0
        )

    # B. Voltage Write-back
    if hasattr(env.motor_deg_state, "battery_voltage"):
        current_soc = env.motor_deg_state.soc[env_ids]
        current_load_watts = total_power_env.squeeze(-1)
        
        new_voltage = compute_battery_voltage(current_soc, current_load_watts)
        # [Fix #8] Ground truth voltage 별도 저장 (로깅/brownout 기준)
        true_voltage = torch.nan_to_num(new_voltage, nan=25.0).clamp(0.0, 60.0)
        if hasattr(env.motor_deg_state, "battery_voltage_true"):
            env.motor_deg_state.battery_voltage_true[env_ids] = true_voltage

        if hasattr(env.motor_deg_state, "voltage_sensor_bias"):
            sensor_bias = env.motor_deg_state.voltage_sensor_bias[env_ids]
        else:
            sensor_bias = torch.zeros_like(new_voltage)
        
        # battery_voltage = biased (에이전트 관측용)
        final_voltage = true_voltage + sensor_bias
        v_quant_step = float(getattr(getattr(env, "cfg", None), "battery_voltage_quant_step_v", 0.0))
        final_voltage = _quantize_channel(final_voltage, v_quant_step)
        env.motor_deg_state.battery_voltage[env_ids] = torch.clamp(final_voltage, 0.0, 60.0)

        # Optional 8-cell observable channel (for replay governor parity with real logs).
        if hasattr(env.motor_deg_state, "cell_voltage"):
            cell_ocv_bias = getattr(env.motor_deg_state, "cell_ocv_bias", None)
            if not isinstance(cell_ocv_bias, torch.Tensor):
                cell_ocv_bias = torch.zeros((len(current_soc), 8), device=env.device, dtype=torch.float32)
            else:
                cell_ocv_bias = cell_ocv_bias[env_ids]

            cell_ir = getattr(env.motor_deg_state, "cell_internal_resistance", None)
            if not isinstance(cell_ir, torch.Tensor):
                cell_ir = torch.full((len(current_soc), 8), 0.0045, device=env.device, dtype=torch.float32)
            else:
                cell_ir = cell_ir[env_ids]

            cell_sensor_bias = getattr(env.motor_deg_state, "cell_sensor_bias", None)
            if not isinstance(cell_sensor_bias, torch.Tensor):
                cell_sensor_bias = torch.zeros((len(current_soc), 8), device=env.device, dtype=torch.float32)
            else:
                cell_sensor_bias = cell_sensor_bias[env_ids]

            pack_current_est = torch.clamp(
                current_load_watts / torch.clamp(true_voltage, min=1.0),
                min=0.0,
            )
            cell_base = true_voltage.unsqueeze(-1) / 8.0 + cell_ocv_bias
            cell_true = cell_base - (cell_ir * pack_current_est.unsqueeze(-1))
            cell_meas = cell_true + cell_sensor_bias
            cell_quant_step = float(getattr(getattr(env, "cfg", None), "cell_voltage_quant_step_v", 0.0))
            cell_meas = _quantize_channel(cell_meas, cell_quant_step)
            env.motor_deg_state.cell_voltage[env_ids] = torch.clamp(cell_meas, 2.5, 4.25)

    # -------------------------------------------------------------------------
    # 5.5. 토크 포화도 기록 & 모터 구속(Stall) 상태
    # -------------------------------------------------------------------------
    # [Fix] Saturation: Runtime effort limits 사용 (실제 운용 한계 대비 근접도).
    # Nominal limits를 쓰면, degraded motor가 한계에 도달해도 ratio < 0.99로
    # 보상 함수(actuator_saturation_reward)가 포화 상태를 감지하지 못합니다.
    runtime_limits = robot.data.joint_effort_limits[env_ids][:, env.motor_deg_joint_indices]
    safe_runtime_limits = torch.clamp(runtime_limits, min=1e-6)
    saturation_ratio = torch.abs(applied_torque) / safe_runtime_limits
    if hasattr(env.motor_deg_state, "torque_saturation"):
        env.motor_deg_state.torque_saturation[env_ids] = torch.maximum(
            env.motor_deg_state.torque_saturation[env_ids], saturation_ratio
        )

    # Stall detection: Nominal limits 사용 (설계 스펙 대비 절대 부하 수준).
    if hasattr(env, "_nominal_effort_limits") and env._nominal_effort_limits is not None:
        nominal_limits = env._nominal_effort_limits[env_ids][:, env.motor_deg_joint_indices]
    else:
        nominal_limits = runtime_limits
    is_high_torque = torch.abs(applied_torque) > (nominal_limits * RATED_LOAD_FACTOR)
    is_stuck = torch.abs(joint_vel) < MIN_STALL_VELOCITY
    
    # [Fix #7] 모터별(per-joint) stall 추적: (N, J) boolean
    # 이전: torch.any(dim=1)로 축소하여 어떤 모터가 stall인지 식별 불가.
    momentary_stall = is_high_torque & is_stuck  # (N, J)

    if hasattr(env.motor_deg_state, "stall_timer"):
        # [Fix #8] Instant reset → Linear decay.
        # 이전: stall이 1 substep이라도 풀리면 타이머가 즉시 0으로 리셋되어
        # 간헐적 stall(미세 진동으로 잠깐 풀렸다 다시 걸리는 경우) 감지 불가.
        # 수정: 비-스톨 시 누적량의 2배 속도로 선형 감쇠하여
        # 짧은 해제 구간에서도 타이머가 유지되고, 지속적 해제 시 자연스럽게 0으로 수렴.
        STALL_DECAY_RATE = 2.0  # 누적(+dt) 대비 감쇠(-2*dt) 비율
        env.motor_deg_state.stall_timer[env_ids] = torch.where(
            momentary_stall,
            env.motor_deg_state.stall_timer[env_ids] + dt,
            torch.clamp(env.motor_deg_state.stall_timer[env_ids] - STALL_DECAY_RATE * dt, min=0.0)
        )

    # -------------------------------------------------------------------------
    # 6. 열(Thermal) 모델 업데이트
    # -------------------------------------------------------------------------
    new_temp = update_motor_temperature(
        env,
        dt,
        env_ids,
        p_loss_watts=total_heat_watts,
        p_coil_watts=coil_heat_watts,
        p_case_watts=case_heat_watts,
    )
    
    # [Final Safety] 온도가 NaN이 되면 물리엔진 붕괴의 주범이 됨.
    new_temp = torch.nan_to_num(new_temp, nan=T_AMB).clamp(min=T_AMB, max=200.0)

    if hasattr(env.motor_deg_state, "temp_derivative"):
        # dt가 너무 작으면 튀니까 보호
        safe_dt = dt if dt > 1e-6 else 1e-6
        temp_rate = (new_temp - current_temp) / safe_dt
        env.motor_deg_state.temp_derivative[env_ids] = temp_rate

    env.motor_deg_state.coil_temp[env_ids] = new_temp

    # Real-observable motor temperature is integer-level on hardware.
    case_quant_step = float(getattr(getattr(env, "cfg", None), "case_temp_quant_step_c", 0.0))
    if case_quant_step > 0.0 and hasattr(env.motor_deg_state, "motor_case_temp"):
        quant_case = _quantize_channel(env.motor_deg_state.motor_case_temp[env_ids], case_quant_step)
        env.motor_deg_state.motor_case_temp[env_ids] = torch.clamp(quant_case, min=T_AMB, max=200.0)
        if hasattr(env.motor_deg_state, "case_temp"):
            env.motor_deg_state.case_temp[env_ids] = env.motor_deg_state.motor_case_temp[env_ids]

    warn_mask = new_temp > TEMP_WARN_THRESHOLD
    crit_mask = new_temp > TEMP_CRITICAL_THRESHOLD
    status_vals = torch.zeros_like(new_temp, dtype=torch.float32)
    status_vals = torch.where(warn_mask, 1.0, status_vals)
    status_vals = torch.where(crit_mask, 2.0, status_vals)
    env.motor_deg_state.thermal_status[env_ids] = status_vals

    # -------------------------------------------------------------------------
    # 7. 버퍼 및 동기화
    # -------------------------------------------------------------------------
    if hasattr(env, "motor_deg_short_term_buffer"):
        # [Fix] env_ids를 전달하여 해당 환경만 EMA/충격량 갱신.
        # 반환값은 이미 env_ids 범위에 해당하는 크기이므로 재슬라이싱 불필요.
        metrics = env.motor_deg_short_term_buffer.update(env, env_ids=env_ids)
        
        if hasattr(env.motor_deg_state, "jitter_intensity"):
            jitter_data = metrics["jitter"]
            val = torch.maximum(env.motor_deg_state.jitter_intensity[env_ids], jitter_data)
            env.motor_deg_state.jitter_intensity[env_ids] = torch.nan_to_num(val, nan=0.0)

        if hasattr(env.motor_deg_state, "impact_intensity") and metrics["impact_jerk"].numel() > 0:
            impact_data = metrics["impact_jerk"]
            max_impact, _ = torch.max(impact_data, dim=1)
            env.motor_deg_state.impact_intensity[env_ids] = torch.nan_to_num(max_impact, nan=0.0)

    if hasattr(env, "motor_deg_long_term_buffer"):
        # [Fix] env_ids를 전달하여 해당 환경만 열 누적/타이머/스냅샷 갱신.
        env.motor_deg_long_term_buffer.update(env, dt, env_ids=env_ids)

    # env.extras 업데이트
    if "joint_temp" in env.extras:
        env.extras["joint_temp"][env_ids] = env.motor_deg_state.coil_temp[env_ids]
    if "joint_case_temp" in env.extras and hasattr(env.motor_deg_state, "motor_case_temp"):
        env.extras["joint_case_temp"][env_ids] = env.motor_deg_state.motor_case_temp[env_ids]


# =============================================================================
# 커리큘럼 유틸 (Time Ramp, optional Performance Gate)
# =============================================================================
def _ramp01(x: float, x0: float, x1: float) -> float:
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    return (x - x0) / max(x1 - x0, 1e-6)


def _curriculum_landmarks(env: ManagerBasedRLEnv) -> tuple[float, float, float, float, float]:
    """
    Curriculum step landmarks for MotorDeg scenario mix.

    Primary path (recommended):
      - iteration milestones from cfg:
        used_start -> used_end -> aged_end -> critical_end -> final_end
      - converted to simulation steps via motor_deg_curriculum_steps_per_iter.

    Backward-compat fallback:
      - legacy ratio schedule from curriculum_total_steps.
    """
    cfg = getattr(env, "cfg", None)
    # Preferred: explicit iteration milestones.
    if cfg is not None and hasattr(cfg, "motor_deg_curriculum_used_start_iter"):
        try:
            steps_per_iter = max(float(getattr(cfg, "motor_deg_curriculum_steps_per_iter", 24)), 1.0)
            i0 = int(getattr(cfg, "motor_deg_curriculum_used_start_iter", 1000))
            i1 = int(getattr(cfg, "motor_deg_curriculum_used_end_iter", 1400))
            i2 = int(getattr(cfg, "motor_deg_curriculum_aged_end_iter", 1900))
            i3 = int(getattr(cfg, "motor_deg_curriculum_critical_end_iter", 2400))
            i4 = int(getattr(cfg, "motor_deg_curriculum_final_end_iter", 3000))

            # Ensure strictly increasing boundaries.
            i0 = max(i0, 0)
            i1 = max(i1, i0 + 1)
            i2 = max(i2, i1 + 1)
            i3 = max(i3, i2 + 1)
            i4 = max(i4, i3 + 1)

            return (
                float(i0) * steps_per_iter,
                float(i1) * steps_per_iter,
                float(i2) * steps_per_iter,
                float(i3) * steps_per_iter,
                float(i4) * steps_per_iter,
            )
        except Exception:
            pass

    # Fallback: legacy ratio schedule.
    total_steps = 72_000.0
    try:
        cfg_total = getattr(cfg, "curriculum_total_steps", None) if cfg is not None else None
        if cfg_total is not None:
            total_steps = max(float(cfg_total), 10_000.0)
    except Exception:
        total_steps = 72_000.0

    s0 = total_steps * 0.20
    s1 = total_steps * 0.4666666667
    s2 = total_steps * 0.7333333333
    s3 = total_steps * 0.9333333333
    s4 = total_steps
    return s0, s1, s2, s3, s4


def _curriculum_mix_from_step(
    step: float,
    landmarks: tuple[float, float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Smooth curriculum schedule over training step.
    `landmarks` = (S0, S1, S2, S3, S4), where:
      S0: used-start
      S1: used-target end
      S2: aged-target end
      S3: critical-partial end
      S4: final-target end
    """
    s = float(max(step, 0.0))
    S0, S1, S2, S3, S4 = landmarks

    if s <= S0:
        # Fresh 100%
        p_fresh = 1.0
        p_used = 0.0
        p_aged = 0.0
        p_crit = 0.0
    elif s <= S1:
        # Used ramp:
        # fresh 1.00 -> 0.60, used 0.00 -> 0.40
        t = _ramp01(s, S0, S1)
        p_fresh = 1.0 + (0.60 - 1.0) * t
        p_used = 0.40 * t
        p_aged = 0.0
        p_crit = 0.0
    elif s <= S2:
        # Aged ramp:
        # fresh 0.60 -> 0.25, used 0.40 -> 0.45, aged 0.00 -> 0.30
        t = _ramp01(s, S1, S2)
        p_fresh = 0.60 + (0.25 - 0.60) * t
        p_used = 0.40 + (0.45 - 0.40) * t
        p_aged = 0.3 * t
        p_crit = 0.0
    elif s <= S3:
        # Critical ramp (phase 1):
        # fresh 0.25 -> 0.15, used 0.45 -> 0.40, aged 0.30 fixed, critical 0.00 -> 0.15
        t = _ramp01(s, S2, S3)
        p_fresh = 0.25 + (0.15 - 0.25) * t
        p_used = 0.45 + (0.40 - 0.45) * t
        p_aged = 0.3
        p_crit = 0.15 * t
    elif s <= S4:
        # Final settling:
        # fresh 0.15 -> 0.20, used 0.40 fixed, aged 0.30 -> 0.25, critical 0.15 fixed
        t = _ramp01(s, S3, S4)
        p_fresh = 0.15 + (0.20 - 0.15) * t
        p_used = 0.40
        p_aged = 0.30 + (0.25 - 0.30) * t
        p_crit = 0.15
    else:
        # Final target mixture
        p_fresh = 0.2
        p_used = 0.4
        p_aged = 0.25
        p_crit = 0.15

    probs = torch.tensor([p_fresh, p_used, p_aged, p_crit], dtype=torch.float32)
    probs = torch.clamp(probs, min=0.0)
    probs = probs / torch.clamp(torch.sum(probs), min=1e-6)
    return float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])


def _curriculum_effective_step_with_gate(env: ManagerBasedRLEnv, current_step: int, num_resets: int) -> tuple[float, float, float, float]:
    """
    Compute effective curriculum step.
    - Default: pure time-step progression (gate off).
    - Optional: walking-performance gate if cfg.motor_deg_curriculum_use_performance_gate=True.

    Gate logic:
    - stable   : termination_ema<=0.10 and tracking_ema<=0.35 -> advance 1.0x
    - caution  : termination_ema<=0.15 and tracking_ema<=0.50 -> advance 0.3x
    - unstable : otherwise                                   -> freeze 0.0x

    tracking_ema is normalized velocity tracking error:
      ||v_cmd - v|| / (||v_cmd|| + 0.20)
    """
    use_gate = bool(getattr(getattr(env, "cfg", None), "motor_deg_curriculum_use_performance_gate", False))
    if not use_gate:
        if hasattr(env, "extras"):
            env.extras["motor_deg/curriculum_effective_step"] = torch.tensor(float(max(current_step, 0)), device=env.device)
            env.extras["motor_deg/curriculum_term_ema"] = torch.tensor(0.0, device=env.device)
            env.extras["motor_deg/curriculum_track_ema"] = torch.tensor(0.0, device=env.device)
            env.extras["motor_deg/curriculum_advance_gain"] = torch.tensor(1.0, device=env.device)
        return float(max(current_step, 0)), 0.0, 0.0, 1.0

    if not hasattr(env, "_curriculum_last_step"):
        env._curriculum_last_step = int(current_step)
    if not hasattr(env, "_curriculum_effective_step"):
        env._curriculum_effective_step = float(current_step)
    if not hasattr(env, "_curriculum_term_ema"):
        env._curriculum_term_ema = 0.0
    if not hasattr(env, "_curriculum_track_ema"):
        env._curriculum_track_ema = 0.0

    delta = max(int(current_step) - int(env._curriculum_last_step), 0)
    env._curriculum_last_step = int(current_step)

    # Prefer non-timeout resets as instability proxy; timeout resets are normal rollovers.
    non_timeout_resets = None
    if hasattr(env, "_curriculum_non_timeout_resets") and hasattr(env, "_curriculum_reset_count"):
        try:
            if int(getattr(env, "_curriculum_reset_count")) == int(num_resets):
                non_timeout_resets = float(getattr(env, "_curriculum_non_timeout_resets"))
        except Exception:
            non_timeout_resets = None
    if non_timeout_resets is None:
        non_timeout_resets = float(num_resets)

    term_rate = float(non_timeout_resets) / max(float(getattr(env, "num_envs", 1)), 1.0)
    track_err = float("nan")
    try:
        robot = env.scene["robot"]
        cmd_xy = env.command_manager.get_command("base_velocity")[:, :2]
        vel_xy = robot.data.root_lin_vel_b[:, :2]
        cmd_norm = torch.norm(cmd_xy, dim=1)
        abs_err = torch.norm(cmd_xy - vel_xy, dim=1)
        rel_err = abs_err / torch.clamp(cmd_norm + 0.20, min=0.20)
        track_err = float(torch.mean(rel_err).detach().cpu().item())
    except Exception:
        # command manager 초기화 타이밍 이슈가 있어도 커리큘럼이 깨지지 않도록 안전 폴백
        track_err = env._curriculum_track_ema

    beta = 0.05
    env._curriculum_term_ema = (1.0 - beta) * float(env._curriculum_term_ema) + beta * term_rate
    env._curriculum_track_ema = (1.0 - beta) * float(env._curriculum_track_ema) + beta * float(track_err)

    term_ema = float(env._curriculum_term_ema)
    track_ema = float(env._curriculum_track_ema)
    if term_ema <= 0.10 and track_ema <= 0.35:
        advance_gain = 1.0
    elif term_ema <= 0.15 and track_ema <= 0.50:
        advance_gain = 0.3
    else:
        advance_gain = 0.0

    env._curriculum_effective_step = float(env._curriculum_effective_step) + float(delta) * float(advance_gain)
    env._curriculum_effective_step = min(float(env._curriculum_effective_step), float(current_step))
    env._curriculum_effective_step = max(float(env._curriculum_effective_step), 0.0)

    if hasattr(env, "extras"):
        env.extras["motor_deg/curriculum_effective_step"] = torch.tensor(float(env._curriculum_effective_step), device=env.device)
        env.extras["motor_deg/curriculum_term_ema"] = torch.tensor(term_ema, device=env.device)
        env.extras["motor_deg/curriculum_track_ema"] = torch.tensor(track_ema, device=env.device)
        env.extras["motor_deg/curriculum_advance_gain"] = torch.tensor(float(advance_gain), device=env.device)

    return float(env._curriculum_effective_step), term_ema, track_ema, float(advance_gain)


# =============================================================================
# 에피소드 리셋 (Modified for Curriculum Learning)
# =============================================================================
def reset_motor_deg_interface(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    [MotorDeg Reset Logic: Hybrid Curriculum Randomization]
    1. 시간 기반 부드러운 시나리오 혼합 램프
    2. (옵션) 성능 게이트(termination/tracking EMA)로 난이도 상승 속도 제어
    """
    device = env.device
    num_resets = len(env_ids)
    if (not hasattr(env, "_motor_deg_scenario_id")) or int(getattr(env._motor_deg_scenario_id, "numel", lambda: 0)()) != int(env.num_envs):
        env._motor_deg_scenario_id = torch.zeros((env.num_envs,), dtype=torch.long, device=device)

    # Snapshot terminal-step episode diagnostics before state reset.
    # These are used to estimate per-motor/pair difficulty for adaptive sampling.
    prev_fault_motor_id = torch.full((num_resets,), -1, device=device, dtype=torch.long)
    if hasattr(env.motor_deg_state, "fault_motor_id"):
        try:
            prev_fault_motor_id = env.motor_deg_state.fault_motor_id[env_ids].to(torch.long).clone()
        except Exception:
            pass

    prev_time_out_mask = torch.zeros((num_resets,), device=device, dtype=torch.bool)
    if hasattr(env, "termination_manager") and hasattr(env.termination_manager, "time_outs"):
        try:
            prev_time_out_mask = env.termination_manager.time_outs[env_ids].to(torch.bool).clone()
        except Exception:
            pass

    prev_sat_ratio = torch.zeros((num_resets,), device=device, dtype=torch.float32)
    if hasattr(env, "_crit_sat_ratio") and isinstance(getattr(env, "_crit_sat_ratio"), torch.Tensor):
        try:
            prev_sat_ratio = env._crit_sat_ratio[env_ids].to(torch.float32).clone()
        except Exception:
            pass

    prev_latched_flag = torch.zeros((num_resets,), device=device, dtype=torch.float32)
    if hasattr(env, "_crit_latch_steps_remaining") and hasattr(env, "_crit_need_unlatch"):
        try:
            prev_latched = (
                (env._crit_latch_steps_remaining[env_ids] > 0) | env._crit_need_unlatch[env_ids]
            ).to(torch.float32)
            prev_latched_flag = prev_latched.clone()
        except Exception:
            pass

    # 1. 상태 및 적분기 초기화 (기존 코드 유지)
    env.motor_deg_state.reset(env_ids)
    
    # ---------------------------------------------------------------------
    # Nominal Physics Restoration (Prevent Collapse)
    # ---------------------------------------------------------------------
    if hasattr(env.motor_deg_state, "degraded_stiffness") and hasattr(env.motor_deg_state, "degraded_damping"):
        joint_idx = getattr(env, "motor_deg_joint_indices", slice(None))
        if (
            hasattr(env, "_nominal_stiffness")
            and env._nominal_stiffness is not None
            and hasattr(env, "_nominal_damping")
            and env._nominal_damping is not None
        ):
            nom_stiff_subset = env._nominal_stiffness[env_ids]
            nom_damp_subset = env._nominal_damping[env_ids]

            if isinstance(joint_idx, slice):
                env.motor_deg_state.degraded_stiffness[env_ids] = nom_stiff_subset
                env.motor_deg_state.degraded_damping[env_ids] = nom_damp_subset
            else:
                env.motor_deg_state.degraded_stiffness[env_ids] = nom_stiff_subset[:, joint_idx]
                env.motor_deg_state.degraded_damping[env_ids] = nom_damp_subset[:, joint_idx]
        else:
            raise RuntimeError(
                "[MotorDeg Error] '_nominal_stiffness' not found in Env. "
                "Ensure actutators are configured correctly in EnvCfg."
            )

    # 버퍼 초기화
    if hasattr(env, "motor_deg_long_term_buffer"):
        env.motor_deg_long_term_buffer.reset(env_ids)
    if hasattr(env, "motor_deg_short_term_buffer"):
        env.motor_deg_short_term_buffer.reset(env_ids)

    # ---------------------------------------------------------------------
    # [Priority 4] Systematic Bias Injection (Sim-to-Real Gap Modeling)
    # ---------------------------------------------------------------------
    # Symmetric bias around 1.0 to avoid one-sided "always-more-friction" domain skew.
    friction_bias_range = getattr(getattr(env, "cfg", None), "friction_bias_range", (0.95, 1.05))
    try:
        f_min = float(friction_bias_range[0])
        f_max = float(friction_bias_range[1])
    except Exception:
        f_min, f_max = 0.95, 1.05
    if f_max < f_min:
        f_min, f_max = f_max, f_min
    friction_noise = torch.rand((num_resets, NUM_MOTORS), device=device) * (f_max - f_min) + f_min
    env.motor_deg_state.friction_bias[env_ids] = friction_noise

    sensor_bias_range = getattr(getattr(env, "cfg", None), "voltage_sensor_bias_range_v", (-0.12, 0.12))
    try:
        v_bias_min = float(sensor_bias_range[0])
        v_bias_max = float(sensor_bias_range[1])
    except Exception:
        v_bias_min, v_bias_max = -0.2, 0.2
    if v_bias_max < v_bias_min:
        v_bias_min, v_bias_max = v_bias_max, v_bias_min
    sensor_noise = torch.rand(num_resets, device=device) * (v_bias_max - v_bias_min) + v_bias_min
    env.motor_deg_state.voltage_sensor_bias[env_ids] = sensor_noise

    # Cell-level DR parameters (for measurable min-cell voltage channel realism).
    cfg_obj = getattr(env, "cfg", None)
    cell_bias_range = getattr(cfg_obj, "cell_ocv_bias_range_v", (-0.015, 0.015))
    cell_ir_range = getattr(cfg_obj, "cell_ir_range_ohm", (0.0035, 0.0065))
    cell_sensor_bias_range = getattr(cfg_obj, "cell_sensor_bias_range_v", (-0.010, 0.010))

    if hasattr(env.motor_deg_state, "cell_ocv_bias"):
        lo, hi = float(cell_bias_range[0]), float(cell_bias_range[1])
        if hi < lo:
            lo, hi = hi, lo
        env.motor_deg_state.cell_ocv_bias[env_ids] = torch.rand((num_resets, 8), device=device) * (hi - lo) + lo

    if hasattr(env.motor_deg_state, "cell_internal_resistance"):
        lo, hi = float(cell_ir_range[0]), float(cell_ir_range[1])
        if hi < lo:
            lo, hi = hi, lo
        env.motor_deg_state.cell_internal_resistance[env_ids] = torch.rand((num_resets, 8), device=device) * (hi - lo) + lo

    if hasattr(env.motor_deg_state, "cell_sensor_bias"):
        lo, hi = float(cell_sensor_bias_range[0]), float(cell_sensor_bias_range[1])
        if hi < lo:
            lo, hi = hi, lo
        env.motor_deg_state.cell_sensor_bias[env_ids] = torch.rand((num_resets, 8), device=device) * (hi - lo) + lo

    # [Fix] Encoder offset randomization (was always zero — dead feature)
    # Simulates per-robot fixed calibration error (~0.005 rad ≈ 0.3°)
    if hasattr(env.motor_deg_state, "encoder_offset"):
        env.motor_deg_state.encoder_offset[env_ids] = (
            torch.rand((num_resets, NUM_MOTORS), device=device) - 0.5
        ) * 0.01
    
    if hasattr(env.motor_deg_state, "min_voltage_log"):
         env.motor_deg_state.min_voltage_log[env_ids] = 33.6
    if hasattr(env.motor_deg_state, "brownout_latched"):
         env.motor_deg_state.brownout_latched[env_ids] = False

    # ---------------------------------------------------------------------
    # [Hybrid] Smooth curriculum mixture + performance gate
    # ---------------------------------------------------------------------
    current_step = int(env.common_step_counter)
    eff_step, term_ema, track_ema, advance_gain = _curriculum_effective_step_with_gate(
        env=env, current_step=current_step, num_resets=num_resets
    )
    landmarks = _curriculum_landmarks(env)
    p_fresh, p_used, p_aged, p_crit = _curriculum_mix_from_step(eff_step, landmarks)

    # Optional debug/play override:
    # force scenario assignment regardless of curriculum step.
    forced_scenario = str(getattr(cfg_obj, "motor_deg_force_scenario_label", "none")).strip().lower()
    if forced_scenario in {"", "none", "off"}:
        forced_scenario = "none"
    if forced_scenario not in {"none", "fresh", "used", "aged", "critical"}:
        if not _MOTOR_DEG_LOG_FLAGS["forced_scenario_invalid_warned"]:
            logging.warning(
                "[MotorDeg] Invalid motor_deg_force_scenario_label='%s'. Ignoring forced scenario.",
                forced_scenario,
            )
            _MOTOR_DEG_LOG_FLAGS["forced_scenario_invalid_warned"] = True
        forced_scenario = "none"
    if forced_scenario != "none":
        p_fresh = 1.0 if forced_scenario == "fresh" else 0.0
        p_used = 1.0 if forced_scenario == "used" else 0.0
        p_aged = 1.0 if forced_scenario == "aged" else 0.0
        p_crit = 1.0 if forced_scenario == "critical" else 0.0
        if not _MOTOR_DEG_LOG_FLAGS["forced_scenario_info_logged"]:
            logging.info("[MotorDeg] Forced reset scenario active: %s", forced_scenario)
            _MOTOR_DEG_LOG_FLAGS["forced_scenario_info_logged"] = True

    env._curriculum_p_fresh = float(p_fresh)
    env._curriculum_p_used = float(p_used)
    env._curriculum_p_aged = float(p_aged)
    env._curriculum_p_crit = float(p_crit)

    if hasattr(env, "extras"):
        env.extras["motor_deg/curriculum_p_fresh"] = torch.tensor(p_fresh, device=device)
        env.extras["motor_deg/curriculum_p_used"] = torch.tensor(p_used, device=device)
        env.extras["motor_deg/curriculum_p_aged"] = torch.tensor(p_aged, device=device)
        env.extras["motor_deg/curriculum_p_crit"] = torch.tensor(p_crit, device=device)

    rand = torch.rand(num_resets, device=device)
    c_fresh = p_fresh
    c_used = c_fresh + p_used
    c_aged = c_used + p_aged

    # Fault-injection mode:
    # - all_motors_random: legacy behavior (all 12 motors sampled)
    # - single_motor_random: one motor uniformly sampled per env reset
    # - single_motor_fixed: fixed motor index per env reset (for deterministic eval)
    fault_mode = str(getattr(cfg_obj, "motor_deg_fault_injection_mode", "single_motor_random")).strip().lower()
    if fault_mode not in {"all_motors_random", "single_motor_random", "single_motor_fixed"}:
        if not _MOTOR_DEG_LOG_FLAGS["fault_mode_invalid_warned"]:
            logging.warning(
                "[MotorDeg] Invalid motor_deg_fault_injection_mode='%s'. Fallback to single_motor_random.",
                fault_mode,
            )
            _MOTOR_DEG_LOG_FLAGS["fault_mode_invalid_warned"] = True
        fault_mode = "single_motor_random"
    fixed_motor_id = int(getattr(cfg_obj, "motor_deg_fault_fixed_motor_id", -1))
    if fault_mode == "single_motor_fixed" and not (0 <= fixed_motor_id < NUM_MOTORS):
        if not _MOTOR_DEG_LOG_FLAGS["fault_fixed_id_invalid_warned"]:
            logging.warning(
                "[MotorDeg] motor_deg_fault_fixed_motor_id=%s is invalid for mode single_motor_fixed. "
                "Fallback to single_motor_random.",
                fixed_motor_id,
            )
            _MOTOR_DEG_LOG_FLAGS["fault_fixed_id_invalid_warned"] = True
        fault_mode = "single_motor_random"
    # single_motor_random stabilization controls:
    # - mirror-uniform motor sampling (pair first, side second)
    # - hold sampled motor id for fixed step window to equalize step exposure
    pair_uniform_enabled_cfg = bool(getattr(cfg_obj, "motor_deg_fault_pair_uniform_enable", True))
    pair_uniform_enabled = pair_uniform_enabled_cfg
    if pair_uniform_enabled and NUM_MOTORS != 12:
        if not _MOTOR_DEG_LOG_FLAGS["fault_pair_uniform_unsupported_warned"]:
            logging.warning(
                "[MotorDeg] motor_deg_fault_pair_uniform_enable=True requires NUM_MOTORS=12; "
                "fallback to uniform motor sampling (NUM_MOTORS=%s).",
                NUM_MOTORS,
            )
            _MOTOR_DEG_LOG_FLAGS["fault_pair_uniform_unsupported_warned"] = True
        pair_uniform_enabled = False
    try:
        fault_hold_steps = int(getattr(cfg_obj, "motor_deg_fault_hold_steps", 1000))
    except Exception:
        fault_hold_steps = 1000
    fault_hold_steps = max(fault_hold_steps, 0)

    # Optional hard-case focus sampling (single_motor_random only).
    # - plain focus mode: `motor_deg_fault_focus_prob` is the probability of overriding
    #   a fresh draw with manual/adaptive focus motors or pairs.
    # - weighted pair mode: the same value becomes the mixing alpha between the
    #   uniform pair distribution and the manual/adaptive target distribution.
    # - motor_deg_fault_focus_motor_ids / motor_deg_fault_focus_pairs seed the manual focus set.
    focus_prob_raw = getattr(cfg_obj, "motor_deg_fault_focus_prob", 0.0)
    try:
        fault_focus_prob = float(focus_prob_raw)
    except Exception:
        if not _MOTOR_DEG_LOG_FLAGS["fault_focus_prob_invalid_warned"]:
            logging.warning(
                "[MotorDeg] Invalid motor_deg_fault_focus_prob='%s'. Using 0.0.",
                focus_prob_raw,
            )
            _MOTOR_DEG_LOG_FLAGS["fault_focus_prob_invalid_warned"] = True
        fault_focus_prob = 0.0
    fault_focus_prob = max(0.0, min(1.0, fault_focus_prob))

    focus_motor_ids_cfg = getattr(cfg_obj, "motor_deg_fault_focus_motor_ids", ())
    focus_pairs_cfg = getattr(cfg_obj, "motor_deg_fault_focus_pairs", ())
    pair_weighted_enable = bool(getattr(cfg_obj, "motor_deg_fault_pair_weighted_enable", False))
    pair_prob_floor_raw = getattr(cfg_obj, "motor_deg_fault_pair_prob_floor", 0.0)
    pair_prob_cap_raw = getattr(cfg_obj, "motor_deg_fault_pair_prob_cap", 1.0)
    pair_target_weights_raw = getattr(cfg_obj, "motor_deg_fault_pair_target_weights", ())
    pair_adaptive_enable = bool(getattr(cfg_obj, "motor_deg_fault_pair_adaptive_enable", False))
    pair_adaptive_mix_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_mix", 1.0)
    pair_adaptive_beta_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_beta", 4.0)
    pair_adaptive_ema_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_ema", 0.9)
    pair_adaptive_min_ep_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_min_episode_per_pair", 20.0)
    pair_adaptive_w_fail_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_w_fail", 0.55)
    pair_adaptive_w_sat_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_w_sat", 0.30)
    pair_adaptive_w_latch_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_w_latch", 0.15)
    pair_adaptive_sat_scale_raw = getattr(cfg_obj, "motor_deg_fault_pair_adaptive_sat_scale", 1.0)
    motor_adaptive_enable = bool(getattr(cfg_obj, "motor_deg_fault_motor_adaptive_enable", False))
    motor_adaptive_topk_raw = getattr(cfg_obj, "motor_deg_fault_motor_adaptive_topk", 3)
    motor_adaptive_min_ep_raw = getattr(cfg_obj, "motor_deg_fault_motor_adaptive_min_episode_per_motor", 20.0)

    try:
        pair_prob_floor = float(pair_prob_floor_raw)
    except Exception:
        pair_prob_floor = 0.0
    try:
        pair_prob_cap = float(pair_prob_cap_raw)
    except Exception:
        pair_prob_cap = 1.0
    pair_prob_floor = max(0.0, min(1.0, pair_prob_floor))
    pair_prob_cap = max(0.0, min(1.0, pair_prob_cap))
    if pair_prob_cap < pair_prob_floor:
        if not _MOTOR_DEG_LOG_FLAGS["fault_pair_prob_range_invalid_warned"]:
            logging.warning(
                "[MotorDeg] Invalid pair prob range floor=%.4f cap=%.4f; forcing cap=floor.",
                pair_prob_floor,
                pair_prob_cap,
            )
            _MOTOR_DEG_LOG_FLAGS["fault_pair_prob_range_invalid_warned"] = True
        pair_prob_cap = pair_prob_floor
    if pair_weighted_enable and not pair_uniform_enabled:
        if not _MOTOR_DEG_LOG_FLAGS["fault_pair_weighted_requires_pair_uniform_warned"]:
            logging.warning(
                "[MotorDeg] motor_deg_fault_pair_weighted_enable=True requires motor_deg_fault_pair_uniform_enable=True; "
                "falling back to legacy sampler."
            )
            _MOTOR_DEG_LOG_FLAGS["fault_pair_weighted_requires_pair_uniform_warned"] = True
        pair_weighted_enable = False
    if pair_adaptive_enable and not pair_uniform_enabled:
        if not _MOTOR_DEG_LOG_FLAGS["fault_pair_adaptive_requires_pair_uniform_warned"]:
            logging.warning(
                "[MotorDeg] motor_deg_fault_pair_adaptive_enable=True requires motor_deg_fault_pair_uniform_enable=True; "
                "adaptive pair sampler disabled."
            )
            _MOTOR_DEG_LOG_FLAGS["fault_pair_adaptive_requires_pair_uniform_warned"] = True
        pair_adaptive_enable = False
    if pair_adaptive_enable and not pair_weighted_enable:
        if not _MOTOR_DEG_LOG_FLAGS["fault_pair_adaptive_requires_weighted_warned"]:
            logging.warning(
                "[MotorDeg] motor_deg_fault_pair_adaptive_enable=True requires motor_deg_fault_pair_weighted_enable=True; "
                "adaptive pair sampler disabled."
            )
            _MOTOR_DEG_LOG_FLAGS["fault_pair_adaptive_requires_weighted_warned"] = True
        pair_adaptive_enable = False

    try:
        pair_adaptive_mix = float(pair_adaptive_mix_raw)
    except Exception:
        pair_adaptive_mix = 1.0
    pair_adaptive_mix = max(0.0, min(1.0, pair_adaptive_mix))
    try:
        pair_adaptive_beta = float(pair_adaptive_beta_raw)
    except Exception:
        pair_adaptive_beta = 4.0
    pair_adaptive_beta = max(0.0, pair_adaptive_beta)
    try:
        pair_adaptive_ema = float(pair_adaptive_ema_raw)
    except Exception:
        pair_adaptive_ema = 0.9
    pair_adaptive_ema = max(0.0, min(0.999, pair_adaptive_ema))
    try:
        pair_adaptive_min_ep = float(pair_adaptive_min_ep_raw)
    except Exception:
        pair_adaptive_min_ep = 20.0
    pair_adaptive_min_ep = max(1.0, pair_adaptive_min_ep)
    try:
        pair_adaptive_w_fail = float(pair_adaptive_w_fail_raw)
    except Exception:
        pair_adaptive_w_fail = 0.55
    try:
        pair_adaptive_w_sat = float(pair_adaptive_w_sat_raw)
    except Exception:
        pair_adaptive_w_sat = 0.30
    try:
        pair_adaptive_w_latch = float(pair_adaptive_w_latch_raw)
    except Exception:
        pair_adaptive_w_latch = 0.15
    pair_adaptive_w_fail = max(0.0, pair_adaptive_w_fail)
    pair_adaptive_w_sat = max(0.0, pair_adaptive_w_sat)
    pair_adaptive_w_latch = max(0.0, pair_adaptive_w_latch)
    try:
        pair_adaptive_sat_scale = float(pair_adaptive_sat_scale_raw)
    except Exception:
        pair_adaptive_sat_scale = 1.0
    pair_adaptive_sat_scale = max(1e-6, pair_adaptive_sat_scale)
    try:
        motor_adaptive_topk = int(motor_adaptive_topk_raw)
    except Exception:
        motor_adaptive_topk = 3
    motor_adaptive_topk = max(1, min(NUM_MOTORS, motor_adaptive_topk))
    try:
        motor_adaptive_min_ep = float(motor_adaptive_min_ep_raw)
    except Exception:
        motor_adaptive_min_ep = 20.0
    motor_adaptive_min_ep = max(1.0, motor_adaptive_min_ep)

    def _compute_adaptive_motor_stats() -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not hasattr(env, "_motor_deg_fault_adapt_episode_counts")
            or int(getattr(env._motor_deg_fault_adapt_episode_counts, "numel", lambda: 0)()) != int(NUM_MOTORS)
        ):
            env._motor_deg_fault_adapt_episode_counts = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
        if (
            not hasattr(env, "_motor_deg_fault_adapt_fail_counts")
            or int(getattr(env._motor_deg_fault_adapt_fail_counts, "numel", lambda: 0)()) != int(NUM_MOTORS)
        ):
            env._motor_deg_fault_adapt_fail_counts = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
        if (
            not hasattr(env, "_motor_deg_fault_adapt_sat_sum")
            or int(getattr(env._motor_deg_fault_adapt_sat_sum, "numel", lambda: 0)()) != int(NUM_MOTORS)
        ):
            env._motor_deg_fault_adapt_sat_sum = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
        if (
            not hasattr(env, "_motor_deg_fault_adapt_latch_sum")
            or int(getattr(env._motor_deg_fault_adapt_latch_sum, "numel", lambda: 0)()) != int(NUM_MOTORS)
        ):
            env._motor_deg_fault_adapt_latch_sum = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)

        valid_prev = (prev_fault_motor_id >= 0) & (prev_fault_motor_id < NUM_MOTORS)
        if torch.any(valid_prev):
            prev_motor_valid = prev_fault_motor_id[valid_prev]
            ep_add = torch.bincount(prev_motor_valid, minlength=NUM_MOTORS).to(torch.float32)
            env._motor_deg_fault_adapt_episode_counts += ep_add

            fail_mask = valid_prev & (~prev_time_out_mask)
            if torch.any(fail_mask):
                fail_add = torch.bincount(prev_fault_motor_id[fail_mask], minlength=NUM_MOTORS).to(torch.float32)
                env._motor_deg_fault_adapt_fail_counts += fail_add

            sat_acc = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
            sat_acc.index_add_(0, prev_motor_valid, torch.clamp(prev_sat_ratio[valid_prev], min=0.0))
            env._motor_deg_fault_adapt_sat_sum += sat_acc

            latch_acc = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
            latch_acc.index_add_(0, prev_motor_valid, torch.clamp(prev_latched_flag[valid_prev], min=0.0))
            env._motor_deg_fault_adapt_latch_sum += latch_acc

        motor_ep = torch.clamp(env._motor_deg_fault_adapt_episode_counts.to(torch.float32), min=1.0)
        fail_rate = torch.clamp(env._motor_deg_fault_adapt_fail_counts.to(torch.float32) / motor_ep, min=0.0, max=1.0)
        sat_mean = torch.clamp(env._motor_deg_fault_adapt_sat_sum.to(torch.float32) / motor_ep, min=0.0)
        sat_norm = torch.clamp(sat_mean / float(pair_adaptive_sat_scale), min=0.0, max=1.0)
        latch_mean = torch.clamp(env._motor_deg_fault_adapt_latch_sum.to(torch.float32) / motor_ep, min=0.0, max=1.0)
        motor_scores = (
            float(pair_adaptive_w_fail) * fail_rate
            + float(pair_adaptive_w_sat) * sat_norm
            + float(pair_adaptive_w_latch) * latch_mean
        )
        motor_conf = torch.clamp(
            env._motor_deg_fault_adapt_episode_counts.to(torch.float32) / float(motor_adaptive_min_ep),
            min=0.0,
            max=1.0,
        )
        return motor_scores, motor_conf

    focus_motor_ids_set: set[int] = set()

    def _append_focus_motor_id(x):
        try:
            v = int(x)
        except Exception:
            return
        if 0 <= v < NUM_MOTORS:
            focus_motor_ids_set.add(v)
        else:
            if not _MOTOR_DEG_LOG_FLAGS["fault_focus_invalid_entry_warned"]:
                logging.warning(
                    "[MotorDeg] Ignoring out-of-range focus motor id=%s (valid: 0..%s).",
                    v,
                    NUM_MOTORS - 1,
                )
                _MOTOR_DEG_LOG_FLAGS["fault_focus_invalid_entry_warned"] = True

    def _consume_focus_ids(raw_ids):
        if isinstance(raw_ids, str):
            # Accept CSV-like strings: "7,10"
            parts = [p.strip() for p in raw_ids.replace(";", ",").split(",") if p.strip() != ""]
            for p in parts:
                _append_focus_motor_id(p)
            return
        if isinstance(raw_ids, (list, tuple, set)):
            for item in raw_ids:
                _append_focus_motor_id(item)

    def _consume_focus_pairs(raw_pairs):
        if isinstance(raw_pairs, (list, tuple, set)):
            for pair in raw_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    continue
                _append_focus_motor_id(pair[0])
                _append_focus_motor_id(pair[1])

    _consume_focus_ids(focus_motor_ids_cfg)
    _consume_focus_pairs(focus_pairs_cfg)

    focus_motor_ids_sorted = sorted(list(focus_motor_ids_set))
    if len(focus_motor_ids_sorted) > 0:
        fault_focus_motor_choices = torch.as_tensor(focus_motor_ids_sorted, device=device, dtype=torch.long)
    else:
        fault_focus_motor_choices = torch.empty((0,), device=device, dtype=torch.long)

    if pair_uniform_enabled and NUM_MOTORS == 12 and len(focus_motor_ids_sorted) > 0:
        focus_pair_idx_list = [
            i for i, p in enumerate(_FAULT_MIRROR_PAIRS_12) if int(p[0]) in focus_motor_ids_set or int(p[1]) in focus_motor_ids_set
        ]
        fault_focus_pair_indices = torch.as_tensor(focus_pair_idx_list, device=device, dtype=torch.long)
    else:
        fault_focus_pair_indices = torch.empty((0,), device=device, dtype=torch.long)
    adaptive_focus_motor_choices = torch.empty((0,), device=device, dtype=torch.long)
    adaptive_focus_motor_target_probs = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
    adaptive_motor_scores = None
    adaptive_motor_conf = None

    mirror_pairs = None
    num_pairs = 0
    pair_uniform_probs = None
    pair_target_probs = None
    pair_target_weights_has_custom = False
    if pair_uniform_enabled and NUM_MOTORS == 12:
        mirror_pairs = torch.as_tensor(_FAULT_MIRROR_PAIRS_12, device=device, dtype=torch.long)
        num_pairs = int(mirror_pairs.shape[0])
        pair_uniform_probs = torch.full((num_pairs,), 1.0 / float(num_pairs), device=device, dtype=torch.float32)
        pair_target_probs = pair_uniform_probs.clone()

        # Parse optional target pair weights (order: _FAULT_MIRROR_PAIRS_12).
        pair_target_weight_list: list[float] = []
        if isinstance(pair_target_weights_raw, str):
            raw_vals = [
                p.strip()
                for p in pair_target_weights_raw.replace(";", ",").split(",")
                if p.strip() != ""
            ]
            for item in raw_vals:
                try:
                    pair_target_weight_list.append(float(item))
                except Exception:
                    pair_target_weight_list = []
                    break
        elif isinstance(pair_target_weights_raw, (list, tuple)):
            try:
                pair_target_weight_list = [float(x) for x in pair_target_weights_raw]
            except Exception:
                pair_target_weight_list = []

        if len(pair_target_weight_list) > 0:
            if len(pair_target_weight_list) != num_pairs or any(v < 0.0 for v in pair_target_weight_list) or sum(pair_target_weight_list) <= 0.0:
                if not _MOTOR_DEG_LOG_FLAGS["fault_pair_target_weights_invalid_warned"]:
                    logging.warning(
                        "[MotorDeg] Invalid motor_deg_fault_pair_target_weights=%s. Expected %d non-negative values with positive sum.",
                        pair_target_weights_raw,
                        num_pairs,
                    )
                    _MOTOR_DEG_LOG_FLAGS["fault_pair_target_weights_invalid_warned"] = True
            else:
                pair_target_probs = torch.as_tensor(pair_target_weight_list, device=device, dtype=torch.float32)
                pair_target_probs = pair_target_probs / torch.clamp(torch.sum(pair_target_probs), min=1e-8)
                pair_target_weights_has_custom = True

        # Fallback target distribution from focus pair selector when explicit target weights are absent.
        if not pair_target_weights_has_custom and int(fault_focus_pair_indices.numel()) > 0:
            pair_target_probs.zero_()
            pair_target_probs[fault_focus_pair_indices] = 1.0
            pair_target_probs = pair_target_probs / torch.clamp(torch.sum(pair_target_probs), min=1e-8)

        # Keep floor/cap feasible; if impossible, relax to uniform-safe defaults.
        if float(pair_prob_floor) * float(num_pairs) > 1.0:
            if not _MOTOR_DEG_LOG_FLAGS["fault_pair_prob_range_invalid_warned"]:
                logging.warning(
                    "[MotorDeg] motor_deg_fault_pair_prob_floor=%.4f is infeasible for %d pairs; clamping to uniform %.4f.",
                    pair_prob_floor,
                    num_pairs,
                    1.0 / float(num_pairs),
                )
                _MOTOR_DEG_LOG_FLAGS["fault_pair_prob_range_invalid_warned"] = True
            pair_prob_floor = 1.0 / float(num_pairs)
        if float(pair_prob_cap) * float(num_pairs) < 1.0:
            if not _MOTOR_DEG_LOG_FLAGS["fault_pair_prob_range_invalid_warned"]:
                logging.warning(
                    "[MotorDeg] motor_deg_fault_pair_prob_cap=%.4f is infeasible for %d pairs; clamping to uniform %.4f.",
                    pair_prob_cap,
                    num_pairs,
                    1.0 / float(num_pairs),
                )
                _MOTOR_DEG_LOG_FLAGS["fault_pair_prob_range_invalid_warned"] = True
            pair_prob_cap = 1.0 / float(num_pairs)

        if pair_weighted_enable:
            # Cache static target/uniform diagnostics for tensorboard logging.
            env._motor_deg_fault_pair_target_probs = pair_target_probs.clone()
            env._motor_deg_fault_pair_uniform_probs = pair_uniform_probs.clone()

    if pair_adaptive_enable or motor_adaptive_enable:
        adaptive_motor_scores, adaptive_motor_conf = _compute_adaptive_motor_stats()
        env._motor_deg_fault_adaptive_motor_scores = adaptive_motor_scores.clone()
        env._motor_deg_fault_adaptive_motor_confidence = adaptive_motor_conf.clone()

        if motor_adaptive_enable:
            motor_rank_scores = adaptive_motor_scores * adaptive_motor_conf
            positive_mask = motor_rank_scores > 1e-8
            positive_count = int(torch.sum(positive_mask).item())
            if positive_count > 0:
                topk = min(int(motor_adaptive_topk), positive_count)
                adaptive_focus_motor_choices = torch.topk(
                    motor_rank_scores, k=topk, largest=True, sorted=True
                ).indices.to(torch.long)
                adaptive_focus_motor_target_probs.zero_()
                adaptive_focus_motor_target_probs[adaptive_focus_motor_choices] = 1.0 / float(topk)
            env._motor_deg_fault_motor_adaptive_enabled = torch.tensor(1.0, device=device, dtype=torch.float32)
            env._motor_deg_fault_motor_adaptive_topk = torch.tensor(
                float(motor_adaptive_topk), device=device, dtype=torch.float32
            )
            env._motor_deg_fault_motor_adaptive_target_probs = adaptive_focus_motor_target_probs.clone()

        if pair_uniform_enabled and NUM_MOTORS == 12 and pair_adaptive_enable:
            pair_scores = 0.5 * (
                adaptive_motor_scores[mirror_pairs[:, 0]] + adaptive_motor_scores[mirror_pairs[:, 1]]
            )
            pair_ep = env._motor_deg_fault_adapt_episode_counts[mirror_pairs[:, 0]] + env._motor_deg_fault_adapt_episode_counts[
                mirror_pairs[:, 1]
            ]
            pair_conf = torch.clamp(pair_ep.to(torch.float32) / float(pair_adaptive_min_ep), min=0.0, max=1.0)
            pair_scores = pair_scores * pair_conf

            centered = pair_scores - torch.mean(pair_scores)
            adaptive_probs = torch.softmax(centered * float(pair_adaptive_beta), dim=0)
            adaptive_probs = adaptive_probs / torch.clamp(torch.sum(adaptive_probs), min=1e-8)

            if (
                not hasattr(env, "_motor_deg_fault_pair_adaptive_probs_ema")
                or int(getattr(env._motor_deg_fault_pair_adaptive_probs_ema, "numel", lambda: 0)()) != int(num_pairs)
            ):
                env._motor_deg_fault_pair_adaptive_probs_ema = adaptive_probs.clone()
            else:
                env._motor_deg_fault_pair_adaptive_probs_ema = (
                    float(pair_adaptive_ema) * env._motor_deg_fault_pair_adaptive_probs_ema.to(torch.float32)
                    + (1.0 - float(pair_adaptive_ema)) * adaptive_probs
                )
            adaptive_probs_smoothed = env._motor_deg_fault_pair_adaptive_probs_ema
            adaptive_probs_smoothed = adaptive_probs_smoothed / torch.clamp(
                torch.sum(adaptive_probs_smoothed), min=1e-8
            )

            pair_target_probs = (
                (1.0 - float(pair_adaptive_mix)) * pair_target_probs
                + float(pair_adaptive_mix) * adaptive_probs_smoothed
            )
            pair_target_probs = pair_target_probs / torch.clamp(torch.sum(pair_target_probs), min=1e-8)

            # Diagnostics for tensorboard / log consumers.
            env._motor_deg_fault_pair_adaptive_enabled = torch.tensor(1.0, device=device, dtype=torch.float32)
            env._motor_deg_fault_pair_adaptive_mix = torch.tensor(float(pair_adaptive_mix), device=device, dtype=torch.float32)
            env._motor_deg_fault_pair_adaptive_scores = pair_scores.clone()
            env._motor_deg_fault_pair_adaptive_confidence = pair_conf.clone()
            env._motor_deg_fault_pair_adaptive_target_probs = adaptive_probs_smoothed.clone()
            env._motor_deg_fault_pair_adaptive_motor_scores = adaptive_motor_scores.clone()
            env._motor_deg_fault_pair_target_probs = pair_target_probs.clone()

    focus_available = (
        int(fault_focus_motor_choices.numel()) > 0
        or int(adaptive_focus_motor_choices.numel()) > 0
        or bool(pair_target_weights_has_custom)
        or bool(pair_adaptive_enable)
    )
    if fault_focus_prob > 0.0 and not focus_available:
        if not _MOTOR_DEG_LOG_FLAGS["fault_focus_empty_warned"]:
            logging.warning(
                "[MotorDeg] motor_deg_fault_focus_prob=%.3f but no valid focus motors found. "
                "Focus sampling disabled.",
                fault_focus_prob,
            )
            _MOTOR_DEG_LOG_FLAGS["fault_focus_empty_warned"] = True
        fault_focus_prob = 0.0

    if not hasattr(env, "_motor_deg_fault_hold_motor_id") or int(getattr(env._motor_deg_fault_hold_motor_id, "numel", lambda: 0)()) != int(env.num_envs):
        env._motor_deg_fault_hold_motor_id = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)
    if not hasattr(env, "_motor_deg_fault_hold_until_step") or int(getattr(env._motor_deg_fault_hold_until_step, "numel", lambda: 0)()) != int(env.num_envs):
        env._motor_deg_fault_hold_until_step = torch.zeros((env.num_envs,), device=device, dtype=torch.long)
    # Cumulative fault sampling diagnostics (for train-time exposure verification).
    if not hasattr(env, "_motor_deg_fault_episode_counts") or int(getattr(env._motor_deg_fault_episode_counts, "numel", lambda: 0)()) != int(NUM_MOTORS):
        env._motor_deg_fault_episode_counts = torch.zeros((NUM_MOTORS,), device=device, dtype=torch.float32)
    if not hasattr(env, "_motor_deg_fault_episode_total"):
        env._motor_deg_fault_episode_total = torch.tensor(0.0, device=device, dtype=torch.float32)
    if not hasattr(env, "_motor_deg_fault_focus_draw_count"):
        env._motor_deg_fault_focus_draw_count = torch.tensor(0.0, device=device, dtype=torch.float32)
    if not hasattr(env, "_motor_deg_fault_focus_draw_total"):
        env._motor_deg_fault_focus_draw_total = torch.tensor(0.0, device=device, dtype=torch.float32)
    if pair_uniform_enabled and pair_weighted_enable and pair_uniform_probs is not None:
        env._motor_deg_fault_pair_sampling_probs = pair_uniform_probs.clone()
        env._motor_deg_fault_pair_sampling_alpha = torch.tensor(
            float(max(0.0, min(1.0, fault_focus_prob))), device=device, dtype=torch.float32
        )
    if fault_mode != "single_motor_random" or fault_hold_steps <= 0:
        env._motor_deg_fault_hold_motor_id[env_ids] = -1
        env._motor_deg_fault_hold_until_step[env_ids] = 0
    # Reset fault bookkeeping for selected envs before scenario assignment.
    if hasattr(env.motor_deg_state, "fault_mask"):
        env.motor_deg_state.fault_mask[env_ids] = 0.0
    if hasattr(env.motor_deg_state, "fault_motor_id"):
        env.motor_deg_state.fault_motor_id[env_ids] = -1
    env._motor_deg_scenario_id[env_ids] = 0

    def _set_case_from_coil(ids: torch.Tensor, delta_low: float, delta_high: float):
        if not hasattr(env.motor_deg_state, "motor_case_temp"):
            return
        if len(ids) == 0:
            return
        delta = torch.rand((len(ids), NUM_MOTORS), device=device) * (delta_high - delta_low) + delta_low
        case_temp = torch.clamp(env.motor_deg_state.coil_temp[ids] - delta, min=T_AMB)
        env.motor_deg_state.motor_case_temp[ids] = case_temp
        if hasattr(env.motor_deg_state, "case_temp"):
            env.motor_deg_state.case_temp[ids] = case_temp

    def _build_pair_sampling_probs() -> tuple[torch.Tensor, float]:
        """Build per-pair sampling probabilities using uniform/target mixing + floor/cap."""
        if pair_uniform_probs is None:
            return torch.empty((0,), device=device, dtype=torch.float32), 0.0
        if not pair_weighted_enable:
            return pair_uniform_probs.clone(), 0.0

        alpha = max(0.0, min(1.0, float(fault_focus_prob)))
        mixed = (1.0 - alpha) * pair_uniform_probs + alpha * pair_target_probs
        mixed = torch.clamp(mixed, min=float(pair_prob_floor), max=float(pair_prob_cap))
        total = torch.sum(mixed)
        if float(total.item()) <= 1e-8:
            mixed = pair_uniform_probs.clone()
        else:
            mixed = mixed / total
        return mixed, alpha

    def _sample_fault_profile(
        ids: torch.Tensor,
        fatigue_low: float,
        fatigue_high: float,
        margin_low: float,
        margin_high: float,
        coil_low: float,
        coil_high: float,
    ):
        """Sample reset-time fault profile under configured injection mode."""
        n = len(ids)
        fatigue = torch.zeros((n, NUM_MOTORS), device=device, dtype=torch.float32)
        health = torch.ones((n, NUM_MOTORS), device=device, dtype=torch.float32)
        coil = torch.full((n, NUM_MOTORS), float(T_AMB), device=device, dtype=torch.float32)
        fault_mask_local = torch.zeros((n, NUM_MOTORS), device=device, dtype=torch.float32)
        fault_motor_local = torch.full((n,), -1, device=device, dtype=torch.long)
        if n == 0:
            return fatigue, health, coil, fault_mask_local, fault_motor_local

        if fault_mode == "all_motors_random":
            fatigue = torch.rand((n, NUM_MOTORS), device=device) * (fatigue_high - fatigue_low) + fatigue_low
            margin = torch.rand((n, NUM_MOTORS), device=device) * (margin_high - margin_low) + margin_low
            health = torch.clamp(fatigue + margin, max=1.0)
            coil = coil_low + torch.rand((n, NUM_MOTORS), device=device) * (coil_high - coil_low)
            fault_mask_local.fill_(1.0)
            return fatigue, health, coil, fault_mask_local, fault_motor_local

        # Single-motor modes: only one joint per env is degraded at reset.
        if fault_mode == "single_motor_fixed":
            motor_idx = torch.full((n,), int(fixed_motor_id), device=device, dtype=torch.long)
        elif fault_mode == "single_motor_random":
            env_indices = ids.to(device=device, dtype=torch.long)
            motor_idx = torch.full((n,), -1, device=device, dtype=torch.long)

            if fault_hold_steps > 0:
                held_motor = env._motor_deg_fault_hold_motor_id[env_indices]
                held_until = env._motor_deg_fault_hold_until_step[env_indices]
                reuse_mask = (held_until > int(current_step)) & (held_motor >= 0) & (held_motor < NUM_MOTORS)
                if torch.any(reuse_mask):
                    motor_idx[reuse_mask] = held_motor[reuse_mask]
            else:
                reuse_mask = torch.zeros((n,), device=device, dtype=torch.bool)

            sample_mask = ~reuse_mask
            sample_count = int(torch.sum(sample_mask).item())
            n_focus_draw = 0
            if sample_count > 0:
                if pair_uniform_enabled:
                    if pair_weighted_enable:
                        pair_probs, alpha = _build_pair_sampling_probs()
                        pair_idx = torch.multinomial(pair_probs, num_samples=sample_count, replacement=True)
                        if int(fault_focus_pair_indices.numel()) > 0:
                            n_focus_draw = int(torch.sum(torch.isin(pair_idx, fault_focus_pair_indices)).item())
                        else:
                            n_focus_draw = int(round(float(alpha) * float(sample_count)))
                        # Export the active pair distribution for diagnostics.
                        env._motor_deg_fault_pair_sampling_probs = pair_probs.clone()
                        env._motor_deg_fault_pair_sampling_alpha = torch.tensor(
                            float(alpha), device=device, dtype=torch.float32
                        )
                    else:
                        pair_idx = torch.randint(0, int(mirror_pairs.shape[0]), (sample_count,), device=device)
                        if fault_focus_prob > 0.0 and int(fault_focus_pair_indices.numel()) > 0:
                            focus_draw = torch.rand((sample_count,), device=device) < float(fault_focus_prob)
                            n_focus = int(torch.sum(focus_draw).item())
                            n_focus_draw = n_focus
                            if n_focus > 0:
                                focus_ids = fault_focus_pair_indices[
                                    torch.randint(0, int(fault_focus_pair_indices.numel()), (n_focus,), device=device)
                                ]
                                pair_idx[focus_draw] = focus_ids
                    side_idx = torch.randint(0, 2, (sample_count,), device=device)
                    sampled_motor = mirror_pairs[pair_idx, side_idx]
                else:
                    sampled_motor = torch.randint(0, NUM_MOTORS, (sample_count,), device=device, dtype=torch.long)
                    if fault_focus_prob > 0.0 and int(fault_focus_motor_choices.numel()) > 0:
                        focus_draw = torch.rand((sample_count,), device=device) < float(fault_focus_prob)
                        n_focus = int(torch.sum(focus_draw).item())
                        n_focus_draw = n_focus
                        if n_focus > 0:
                            sampled_focus = fault_focus_motor_choices[
                                torch.randint(0, int(fault_focus_motor_choices.numel()), (n_focus,), device=device)
                            ]
                            sampled_motor[focus_draw] = sampled_focus
                if fault_focus_prob > 0.0 and int(adaptive_focus_motor_choices.numel()) > 0:
                    focus_draw = torch.rand((sample_count,), device=device) < float(fault_focus_prob)
                    n_focus = int(torch.sum(focus_draw).item())
                    n_focus_draw = n_focus
                    if n_focus > 0:
                        sampled_focus = adaptive_focus_motor_choices[
                            torch.randint(0, int(adaptive_focus_motor_choices.numel()), (n_focus,), device=device)
                        ]
                        sampled_motor[focus_draw] = sampled_focus
                motor_idx[sample_mask] = sampled_motor
                # Count only newly sampled resets (held envs are not fresh draws).
                if fault_focus_prob > 0.0 and focus_available:
                    env._motor_deg_fault_focus_draw_count += float(n_focus_draw)
                    env._motor_deg_fault_focus_draw_total += float(sample_count)

            # Update/refresh hold window for all just-reset environments.
            if fault_hold_steps > 0:
                env._motor_deg_fault_hold_motor_id[env_indices] = motor_idx
                env._motor_deg_fault_hold_until_step[env_indices] = int(current_step + fault_hold_steps)
            else:
                env._motor_deg_fault_hold_motor_id[env_indices] = -1
                env._motor_deg_fault_hold_until_step[env_indices] = 0
        else:
            motor_idx = torch.randint(0, NUM_MOTORS, (n,), device=device, dtype=torch.long)
        row_idx = torch.arange(n, device=device, dtype=torch.long)
        # Episode-level fault motor exposure counter (reset-time draws).
        if motor_idx.numel() > 0:
            binc = torch.bincount(motor_idx, minlength=NUM_MOTORS).to(torch.float32)
            env._motor_deg_fault_episode_counts += binc
            env._motor_deg_fault_episode_total += float(motor_idx.numel())

        sampled_fatigue = torch.rand((n,), device=device) * (fatigue_high - fatigue_low) + fatigue_low
        sampled_margin = torch.rand((n,), device=device) * (margin_high - margin_low) + margin_low
        sampled_coil = torch.rand((n,), device=device) * (coil_high - coil_low) + coil_low

        fatigue[row_idx, motor_idx] = sampled_fatigue
        health[row_idx, motor_idx] = torch.clamp(sampled_fatigue + sampled_margin, max=1.0)
        coil[row_idx, motor_idx] = sampled_coil
        fault_mask_local[row_idx, motor_idx] = 1.0
        fault_motor_local = motor_idx
        return fatigue, health, coil, fault_mask_local, fault_motor_local

    # Scenario 1: Fresh
    mask_fresh = rand < c_fresh
    if torch.any(mask_fresh):
        ids = env_ids[mask_fresh]
        env._motor_deg_scenario_id[ids] = 1
        env.motor_deg_state.fatigue_index[ids] = 0.0
        env.motor_deg_state.motor_health_capacity[ids] = 1.0
        env.motor_deg_state.coil_temp[ids] = T_AMB
        _set_case_from_coil(ids, 0.0, 0.5)
        if hasattr(env.motor_deg_state, "soc"):
            env.motor_deg_state.soc[ids] = 1.0

    # Scenario 2: Used
    mask_used = (rand >= c_fresh) & (rand < c_used)
    if torch.any(mask_used):
        ids = env_ids[mask_used]
        env._motor_deg_scenario_id[ids] = 2
        used_fatigue, used_health, used_coil, used_fault_mask, used_fault_motor = _sample_fault_profile(
            ids=ids,
            fatigue_low=0.1,
            fatigue_high=0.4,
            margin_low=0.30,
            margin_high=0.55,
            coil_low=T_AMB,
            coil_high=T_AMB + 10.0,
        )
        env.motor_deg_state.fatigue_index[ids] = used_fatigue
        env.motor_deg_state.motor_health_capacity[ids] = used_health
        env.motor_deg_state.coil_temp[ids] = used_coil
        _set_case_from_coil(ids, 1.0, 3.5)
        if hasattr(env.motor_deg_state, "fault_mask"):
            env.motor_deg_state.fault_mask[ids] = used_fault_mask
        if hasattr(env.motor_deg_state, "fault_motor_id"):
            env.motor_deg_state.fault_motor_id[ids] = used_fault_motor
        if hasattr(env.motor_deg_state, "soc"):
            env.motor_deg_state.soc[ids] = torch.rand(len(ids), device=device) * 0.2 + 0.8

    # Scenario 3: Aged
    mask_aged = (rand >= c_used) & (rand < c_aged)
    if torch.any(mask_aged):
        ids = env_ids[mask_aged]
        env._motor_deg_scenario_id[ids] = 3
        aged_fatigue, aged_health, aged_coil, aged_fault_mask, aged_fault_motor = _sample_fault_profile(
            ids=ids,
            fatigue_low=0.4,
            fatigue_high=0.7,
            margin_low=0.10,
            margin_high=0.25,
            coil_low=T_AMB,
            coil_high=T_AMB + 20.0,
        )
        env.motor_deg_state.fatigue_index[ids] = aged_fatigue
        env.motor_deg_state.motor_health_capacity[ids] = aged_health
        env.motor_deg_state.coil_temp[ids] = aged_coil
        _set_case_from_coil(ids, 2.5, 6.5)
        if hasattr(env.motor_deg_state, "fault_mask"):
            env.motor_deg_state.fault_mask[ids] = aged_fault_mask
        if hasattr(env.motor_deg_state, "fault_motor_id"):
            env.motor_deg_state.fault_motor_id[ids] = aged_fault_motor
        if hasattr(env.motor_deg_state, "soc"):
            env.motor_deg_state.soc[ids] = torch.rand(len(ids), device=device) * 0.4 + 0.4

    # Scenario 4: Critical
    # For case-proxy thermal tasks (e.g., RealObs), avoid overly aggressive reset
    # temperatures that trigger immediate thermal termination.
    crit_coil_min = 65.0
    crit_coil_max = 85.0
    try:
        threshold_temp, use_case_proxy, coil_to_case_delta = thermal_termination_params_from_cfg(
            getattr(env, "cfg", None)
        )
        if use_case_proxy and threshold_temp is not None:
            crit_coil_max = case_proxy_safe_coil_max_for_reset(
                threshold_temp=threshold_temp,
                coil_to_case_delta_c=coil_to_case_delta,
                case_delta_low_c=3.5,
                immediate_term_margin_c=0.5,
                proxy_margin_c=1.0,
                ambient_temp_c=T_AMB,
                min_coil_above_ambient_c=8.0,
            )
            crit_coil_min = max(T_AMB + 5.0, crit_coil_max - 10.0)
    except Exception as err:
        if not _MOTOR_DEG_LOG_FLAGS["thermal_reset_params_read_failed"]:
            logging.warning(
                "[MotorDeg] Failed to read thermal reset params for critical scenario adaptation; "
                "using default critical coil range. error=%s",
                err,
            )
            _MOTOR_DEG_LOG_FLAGS["thermal_reset_params_read_failed"] = True

    mask_crit = rand >= c_aged
    if torch.any(mask_crit):
        ids = env_ids[mask_crit]
        env._motor_deg_scenario_id[ids] = 4
        crit_fatigue, crit_health, crit_coil, crit_fault_mask, crit_fault_motor = _sample_fault_profile(
            ids=ids,
            fatigue_low=0.7,
            fatigue_high=0.95,
            margin_low=0.02,
            margin_high=0.10,
            coil_low=crit_coil_min,
            coil_high=crit_coil_max,
        )
        env.motor_deg_state.fatigue_index[ids] = crit_fatigue
        env.motor_deg_state.motor_health_capacity[ids] = crit_health
        env.motor_deg_state.coil_temp[ids] = crit_coil
        _set_case_from_coil(ids, 3.5, 8.0)
        if hasattr(env.motor_deg_state, "fault_mask"):
            env.motor_deg_state.fault_mask[ids] = crit_fault_mask
        if hasattr(env.motor_deg_state, "fault_motor_id"):
            env.motor_deg_state.fault_motor_id[ids] = crit_fault_motor
        if hasattr(env.motor_deg_state, "soc"):
            env.motor_deg_state.soc[ids] = torch.rand(len(ids), device=device) * 0.2 + 0.1

    if hasattr(env.motor_deg_state, "temp_derivative"):
        env.motor_deg_state.temp_derivative[env_ids] = 0.0
    if hasattr(env.motor_deg_state, "case_temp_derivative"):
        env.motor_deg_state.case_temp_derivative[env_ids] = 0.0
    case_quant_step = float(getattr(getattr(env, "cfg", None), "case_temp_quant_step_c", 0.0))
    if case_quant_step > 0.0 and hasattr(env.motor_deg_state, "motor_case_temp"):
        quant_case = _quantize_channel(env.motor_deg_state.motor_case_temp[env_ids], case_quant_step)
        env.motor_deg_state.motor_case_temp[env_ids] = torch.clamp(quant_case, min=T_AMB, max=200.0)
        if hasattr(env.motor_deg_state, "case_temp"):
            env.motor_deg_state.case_temp[env_ids] = env.motor_deg_state.motor_case_temp[env_ids]
    if hasattr(env.motor_deg_state, "encoder_hold_flag"):
        env.motor_deg_state.encoder_hold_flag[env_ids] = 0.0

    # Initialize voltage channels to SOC-consistent values at reset (avoid stale 33.6V mismatch).
    if hasattr(env.motor_deg_state, "soc"):
        pack_v_reset = compute_battery_voltage(
            env.motor_deg_state.soc[env_ids],
            torch.zeros((num_resets,), device=device, dtype=torch.float32),
        )
        if hasattr(env.motor_deg_state, "battery_voltage_true"):
            env.motor_deg_state.battery_voltage_true[env_ids] = pack_v_reset
        sensed_pack_v = pack_v_reset + env.motor_deg_state.voltage_sensor_bias[env_ids]
        v_quant_step = float(getattr(getattr(env, "cfg", None), "battery_voltage_quant_step_v", 0.0))
        sensed_pack_v = _quantize_channel(sensed_pack_v, v_quant_step)
        env.motor_deg_state.battery_voltage[env_ids] = torch.clamp(sensed_pack_v, 0.0, 60.0)
        if hasattr(env.motor_deg_state, "min_voltage_log"):
            env.motor_deg_state.min_voltage_log[env_ids] = env.motor_deg_state.battery_voltage_true[env_ids]

        if hasattr(env.motor_deg_state, "cell_voltage"):
            cell_base = pack_v_reset.unsqueeze(-1) / 8.0 + env.motor_deg_state.cell_ocv_bias[env_ids]
            cell_meas = cell_base + env.motor_deg_state.cell_sensor_bias[env_ids]
            c_quant_step = float(getattr(getattr(env, "cfg", None), "cell_voltage_quant_step_v", 0.0))
            cell_meas = _quantize_channel(cell_meas, c_quant_step)
            env.motor_deg_state.cell_voltage[env_ids] = torch.clamp(cell_meas, 2.5, 4.25)

    # ---------------------------------------------------------------------
    # [Fix] base_friction_torque를 articulation joint friction 계수에서 재파생.
    # state.reset()이 고정값 0.2(STICTION_NOMINAL)로 초기화하지만,
    # 런타임 joint friction tensor가 제공되는 경우 이를 반영해야
    # _apply_physical_degradation과 _compute_thermal_limits가 리셋 직후 첫 substep부터
    # 같은 마찰 기준을 공유할 수 있습니다.
    # NOTE: 이는 terrain/contact 마찰 랜덤화와는 다른 joint-side friction 소스입니다.
    # ---------------------------------------------------------------------
    robot = env.scene["robot"]
    joint_indices = getattr(env, "motor_deg_joint_indices", slice(None))
    friction_source = None
    if hasattr(robot.data, "joint_friction_coeff"):
        t = robot.data.joint_friction_coeff
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            friction_source = t
    if friction_source is None and hasattr(robot.data, "default_joint_friction_coeff"):
        t = robot.data.default_joint_friction_coeff
        if isinstance(t, torch.Tensor) and t.numel() > 0:
            friction_source = t
    if friction_source is not None:
        try:
            if isinstance(joint_indices, slice):
                env.motor_deg_state.base_friction_torque[env_ids] = friction_source[env_ids]
            else:
                env.motor_deg_state.base_friction_torque[env_ids] = friction_source[env_ids][:, joint_indices]
        except Exception as err:
            if not _MOTOR_DEG_LOG_FLAGS["friction_source_reset_failed"]:
                logging.warning(
                    "[MotorDeg] Failed to sync reset joint friction coefficients; "
                    "keeping nominal defaults. error=%s",
                    err,
                )
                _MOTOR_DEG_LOG_FLAGS["friction_source_reset_failed"] = True

    # ---------------------------------------------------------------------
    # 3. Buffer Sync & Data Export
    # ---------------------------------------------------------------------
    if hasattr(env, "motor_deg_long_term_buffer"):
        # [Fix v3.0] 초기 fatigue를 slot 0에 단일 스냅샷으로 기록하고 index를 1로 전진.
        # 이전: 전체 히스토리를 동일 값으로 채우고 is_buffer_filled=True 강제 설정.
        #   → 첫 실제 스냅샷이 oldest 위치를 덮어써 slope 부호 역전 + 1/N 왜곡.
        # 수정: fill_count=1로 시작 → 두 번째 스냅샷부터 정확한 slope 계산 가능.
        #   slope=0 구간(초반)은 fill_count<2 가드로 안전하게 처리됨.
        init_fatigue = env.motor_deg_state.fatigue_index[env_ids]  # (M, J)
        env.motor_deg_long_term_buffer.fatigue_snapshots[env_ids, 0, :] = init_fatigue
        env.motor_deg_long_term_buffer.snapshot_index[env_ids] = 1
        env.motor_deg_long_term_buffer.fill_count[env_ids] = 1

    if "joint_temp" in env.extras:
        env.extras["joint_temp"][env_ids] = env.motor_deg_state.coil_temp[env_ids]
    if "joint_case_temp" in env.extras and hasattr(env.motor_deg_state, "motor_case_temp"):
        env.extras["joint_case_temp"][env_ids] = env.motor_deg_state.motor_case_temp[env_ids]
