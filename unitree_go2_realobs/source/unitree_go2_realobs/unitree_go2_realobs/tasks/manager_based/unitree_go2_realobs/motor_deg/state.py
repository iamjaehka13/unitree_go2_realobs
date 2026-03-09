# =============================================================================
# unitree_go2_realobs/motor_deg/state.py
# Runtime MotorDeg state container shared across resets, observations, and rewards.
# =============================================================================
from __future__ import annotations
import torch
from .constants import T_AMB, CELL_INTERNAL_RESISTANCE_NOMINAL

class MotorDegState:
    """
    [MotorDeg Core] 로봇 하드웨어의 동적 건전성 상태(State of Health)를 관리하는 중앙 데이터 컨테이너.
    """

    def __init__(self, num_envs: int, num_joints: int, device: str):
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device

        # ---------------------------------------------------------------------
        # 1. 열역학 상태 (Thermal States)
        # ---------------------------------------------------------------------
        self.coil_temp = torch.full((num_envs, num_joints), T_AMB, device=device, dtype=torch.float32)
        # Real-observable proxy channel: motor case / housing temperature.
        self.motor_case_temp = torch.full((num_envs, num_joints), T_AMB, device=device, dtype=torch.float32)
        self.temp_derivative = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.case_temp_derivative = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.thermal_status = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.delta_t = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 2. 전기적 상태 (Electrical States)
        # ---------------------------------------------------------------------
        self.soc = torch.ones((num_envs,), device=device, dtype=torch.float32)
        self.battery_voltage = torch.full((num_envs,), 33.6, device=device, dtype=torch.float32)
        # [Fix #8] Ground truth voltage (sensor bias 미포함) — 로깅 및 brownout 기준용
        self.battery_voltage_true = torch.full((num_envs,), 33.6, device=device, dtype=torch.float32)
        self.instant_power = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        # 8-cell observable BMS channel (sim-side approximation).
        self.cell_voltage = torch.full((num_envs, 8), 4.2, device=device, dtype=torch.float32)
        # Backward-compatible aliases used by replay/eval scripts.
        self.cell_voltages = self.cell_voltage
        self.bms_cell_voltage = self.cell_voltage
        self.bms_cell_voltages = self.cell_voltage
        # Per-cell model parameters (episode-randomized in reset_motor_deg_interface).
        self.cell_ocv_bias = torch.zeros((num_envs, 8), device=device, dtype=torch.float32)
        self.cell_internal_resistance = torch.full(
            (num_envs, 8), CELL_INTERNAL_RESISTANCE_NOMINAL, device=device, dtype=torch.float32
        )
        self.cell_sensor_bias = torch.zeros((num_envs, 8), device=device, dtype=torch.float32)
        
        # Aliasing 방지를 위한 Control Step 동안의 평균 전력
        self.avg_power_log = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        
        self.cumulative_energy = torch.zeros((num_envs,), device=device, dtype=torch.float32)
        self.real_current = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.step_energy_log = torch.zeros((num_envs,), device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 3. 기계적 수명 상태 (Mechanical Health)
        # ---------------------------------------------------------------------
        self.fatigue_rate = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.fatigue_index = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.motor_health_capacity = torch.ones((num_envs, num_joints), device=device, dtype=torch.float32)
        # Fault bookkeeping for analysis/visualization:
        # - fault_mask: per-joint (N, J), 1.0 if the joint is selected as reset-time fault target.
        # - fault_motor_id: per-env index in [0, J-1] for single-motor modes, else -1.
        self.fault_mask = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.fault_motor_id = torch.full((num_envs,), -1, device=device, dtype=torch.long)

        # ---------------------------------------------------------------------
        # 4. 진동 및 충격 (Vibration & Impact)
        # ---------------------------------------------------------------------
        self.impact_intensity = torch.zeros((num_envs,), device=device, dtype=torch.float32)
        self.vibration_g = torch.zeros((num_envs,), device=device, dtype=torch.float32)
        self.jitter_intensity = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        
        # ---------------------------------------------------------------------
        # 5. Sim-to-Real Sensor Corruption
        # ---------------------------------------------------------------------
        self.encoder_offset = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        # 위치 노이즈
        self.encoder_noise = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        # 속도 미분 기반 측정 채널용 노이즈.
        # 실제 로봇에서는 속도 추정 노이즈가 위치 노이즈보다 더 크게 나타날 수 있다.
        self.encoder_vel_noise = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        # Sample-and-hold measurement channels (for sensor latency/drop modeling).
        self.encoder_meas_pos = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.encoder_meas_vel = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.encoder_hold_flag = torch.zeros((num_envs,), device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 6. 제어 게인 열화 (Degraded Control Gains)
        # ---------------------------------------------------------------------
        self.degraded_stiffness = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.degraded_damping = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 7. 제어 및 무결성 (Integrity)
        # ---------------------------------------------------------------------
        self.applied_torque = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.torque_saturation = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.friction_power = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        self.stall_timer = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        # [Fix] interface.py에서 결정된 base stiction torque 캐시 (SSOT 마찰 소스)
        self.base_friction_torque = torch.full((num_envs, num_joints), 0.2, device=device, dtype=torch.float32)

        # ---------------------------------------------------------------------
        # 8. Bias, brownout, and latent-state memory
        # ---------------------------------------------------------------------
        
        # (A) Systematic Bias
        self.friction_bias = torch.ones((num_envs, num_joints), device=device, dtype=torch.float32)
        self.voltage_sensor_bias = torch.zeros((num_envs,), device=device, dtype=torch.float32)

        # (B) Brownout Control State
        self.brownout_scale = torch.ones((num_envs,), device=device, dtype=torch.float32)
        # [Fix #8] Cached BMS predicted voltage for latent observation paths.
        # Brownout logic may use this or measured channels depending on env config.
        self.bms_voltage_pred = torch.full((num_envs,), 33.6, device=device, dtype=torch.float32)
        
        # (C) Min-Pooling Log
        self.min_voltage_log = torch.full((num_envs,), 33.6, device=device, dtype=torch.float32)
        self.brownout_latched = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # (D) Health Residual EMA
        self.torque_residual_ema = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)

    def reset(self, env_ids: torch.Tensor):
        """
        에피소드 초기화 시 MotorDeg 지표 리셋.
        """
        if env_ids is None or len(env_ids) == 0:
            return

        # 1. 기본 상태 리셋
        self.coil_temp[env_ids] = T_AMB
        self.motor_case_temp[env_ids] = T_AMB
        self.case_temp[env_ids] = T_AMB
        self.temp_derivative[env_ids] = 0.0
        self.case_temp_derivative[env_ids] = 0.0
        self.thermal_status[env_ids] = 0.0
        self.delta_t[env_ids] = 0.0
        
        self.soc[env_ids] = 1.0 
        self.battery_voltage[env_ids] = 33.6
        self.battery_voltage_true[env_ids] = 33.6
        self.cell_voltage[env_ids] = 4.2
        self.cell_ocv_bias[env_ids] = 0.0
        self.cell_internal_resistance[env_ids] = CELL_INTERNAL_RESISTANCE_NOMINAL
        self.cell_sensor_bias[env_ids] = 0.0
        self.instant_power[env_ids] = 0.0
        self.avg_power_log[env_ids] = 0.0
        self.cumulative_energy[env_ids] = 0.0
        self.step_energy_log[env_ids] = 0.0
        
        self.fatigue_rate[env_ids] = 0.0
        self.fatigue_index[env_ids] = 0.0
        self.motor_health_capacity[env_ids] = 1.0
        self.fault_mask[env_ids] = 0.0
        self.fault_motor_id[env_ids] = -1
        self.friction_power[env_ids] = 0.0
        self.stall_timer[env_ids] = 0.0
        self.jitter_intensity[env_ids] = 0.0
        
        self.vibration_g[env_ids] = 0.0
        self.impact_intensity[env_ids] = 0.0
        self.real_current[env_ids] = 0.0
        self.applied_torque[env_ids] = 0.0
        self.encoder_noise[env_ids] = 0.0
        self.encoder_vel_noise[env_ids] = 0.0
        self.encoder_meas_pos[env_ids] = 0.0
        self.encoder_meas_vel[env_ids] = 0.0
        self.encoder_hold_flag[env_ids] = 0.0
        # [Fix] encoder_offset도 리셋 (reset_motor_deg_interface에서 재랜덤화하지만,
        # state.reset()만 단독 호출되는 경로에서 이전 에피소드 값 잔류 방지)
        self.encoder_offset[env_ids] = 0.0

        # [Fix] degraded gains 리셋 (이후 _apply_physical_degradation에서 재계산됨)
        self.degraded_stiffness[env_ids] = 0.0
        self.degraded_damping[env_ids] = 0.0

        # [Fix] torque_saturation 리셋
        self.torque_saturation[env_ids] = 0.0

        # [Fix] base_friction_torque를 nominal 값(STICTION_NOMINAL=0.2)으로 리셋
        self.base_friction_torque[env_ids] = 0.2
        
        # 2. Bias / brownout memory reset
        self.friction_bias[env_ids] = 1.0
        self.voltage_sensor_bias[env_ids] = 0.0
        
        self.brownout_scale[env_ids] = 1.0
        self.bms_voltage_pred[env_ids] = 33.6
        self.min_voltage_log[env_ids] = 33.6
        self.brownout_latched[env_ids] = False
        
        self.torque_residual_ema[env_ids] = 0.0

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "coil_temp": self.coil_temp,
            "motor_case_temp": self.motor_case_temp,
            "soc": self.soc,
            "battery_voltage": self.battery_voltage,
            "battery_voltage_true": self.battery_voltage_true,
            "cell_voltage": self.cell_voltage,
            "fatigue_index": self.fatigue_index,
            "motor_health_capacity": self.motor_health_capacity,
            "fault_mask": self.fault_mask,
            "fault_motor_id": self.fault_motor_id,
            "avg_power_log": self.avg_power_log,
            "degraded_stiffness": self.degraded_stiffness,
            "degraded_damping": self.degraded_damping,
            "friction_bias": self.friction_bias,         
            "brownout_scale": self.brownout_scale,       
            "min_voltage_log": self.min_voltage_log,     
            "torque_residual_ema": self.torque_residual_ema, 
            "torque_saturation": self.torque_saturation,
        }

    @property
    def case_temp(self) -> torch.Tensor:
        """Compatibility alias used by observation/reward helpers."""
        return self.motor_case_temp

    @case_temp.setter
    def case_temp(self, value: torch.Tensor):
        self.motor_case_temp = value
