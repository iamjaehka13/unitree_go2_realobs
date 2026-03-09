# unitree_go2_realobs/motor_deg/buffers/short_term.py
from __future__ import annotations
import torch
import logging
from typing import TYPE_CHECKING, List, Dict
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from ..constants import EPS, NORMAL_AXIS
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
class ShortTermHealthBuffer:
    """
    [MotorDeg Core] 고주파 물리 과도 현상(Transient Physics) 추적 버퍼 (Safety Patched v3.2).
    Safe Features:
    1. [Crash Prevention] History Buffer Overflow 방지를 위한 Stride Clamping.
    2. [Physics Integrity] Body Frame 기준의 수직 충격량 계산 명시화.
    """
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        num_envs: int,
        device: str,
        robot_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        joint_indices: List[int] | torch.Tensor,
        analysis_dt: float = 0.01,
        jitter_alpha: float = 0.1
    ):
        self.num_envs = num_envs
        self.device = device
        self.jitter_alpha = jitter_alpha
        self.analysis_dt = analysis_dt
        # 1. 자산 해석
        robot_cfg.resolve(env.scene)
        self.robot_entity: Articulation = env.scene[robot_cfg.name]
        self.contact_sensor_cfg = contact_sensor_cfg
        contact_sensor_cfg.resolve(env.scene)
        target_sensor_name = str(contact_sensor_cfg.name)
        if target_sensor_name == "0":
            logging.warning("[MotorDeg Warning] Contact sensor name is '0'. Forcing 'contact_forces'.")
            target_sensor_name = "contact_forces"
        try:
            self.contact_sensor: ContactSensor | None = env.scene[target_sensor_name]
        except KeyError:
            self.contact_sensor = None
        if self.contact_sensor is None:
            logging.warning("[MotorDeg Warning] Contact sensor not found during init. Impact analysis will be empty until resolved.")
        else:
            if getattr(self.contact_sensor, "num_bodies", 0) == 0:
                logging.warning("[MotorDeg Warning] Contact sensor is not tracking any bodies. Impact analysis will be empty.")
        # 2. 조인트 인덱스 저장
        if isinstance(joint_indices, list):
            self.joint_indices = torch.tensor(joint_indices, device=device)
        else:
            self.joint_indices = joint_indices.to(device)
        # 3. EMA 통계용 상태
        num_target_joints = len(self.joint_indices)
        self.acc_mean_ema = torch.zeros((num_envs, num_target_joints), device=device)
        self.acc_var_ema = torch.zeros((num_envs, num_target_joints), device=device)
        self.is_ema_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, env: ManagerBasedRLEnv, env_ids=None) -> Dict[str, torch.Tensor]:
        """
        매 physics_step마다 호출.
        [Fix] env_ids가 지정되면 해당 환경의 EMA/충격량만 갱신하여
        미갱신 환경의 통계 오염을 방지합니다.
        반환값은 env_ids 범위에 해당하는 텐서입니다.
        """
        physics_dt = env.physics_dt

        # env_ids 정규화
        if env_ids is None or isinstance(env_ids, slice):
            ids = slice(None)
            is_full = True
        else:
            ids = env_ids
            is_full = False

        # ---------------------------------------------------------------------
        # 1. 가속도 Jitter 분석 (EMA) — env_ids 범위만 갱신
        # ---------------------------------------------------------------------
        current_acc = self.robot_entity.data.joint_acc[ids][:, self.joint_indices] if not is_full else self.robot_entity.data.joint_acc[:, self.joint_indices]

        not_init = ~self.is_ema_initialized[ids]
        if torch.any(not_init):
            if is_full:
                self.acc_mean_ema[not_init] = current_acc[not_init]
                self.is_ema_initialized[not_init] = True
            else:
                global_not_init = ids[not_init]
                self.acc_mean_ema[global_not_init] = current_acc[not_init]
                self.is_ema_initialized[global_not_init] = True

        old_mean = self.acc_mean_ema[ids]
        delta = current_acc - old_mean
        new_mean = old_mean + self.jitter_alpha * delta
        self.acc_mean_ema[ids] = new_mean

        old_var = self.acc_var_ema[ids]
        new_var = (1.0 - self.jitter_alpha) * (
            old_var + self.jitter_alpha * delta * (current_acc - new_mean)
        )
        self.acc_var_ema[ids] = new_var
        jitter_score = torch.sqrt(new_var + EPS)

        # ---------------------------------------------------------------------
        # 2. 충격량(Jerk) 분석 (Safe Stride & Body Frame)
        # ---------------------------------------------------------------------
        # Sensor History: (N, T, B, 3)
        if self.contact_sensor is None:
            self.contact_sensor_cfg.resolve(env.scene)
            target_sensor_name = str(self.contact_sensor_cfg.name)
            if target_sensor_name == "0":
                target_sensor_name = "contact_forces"
            try:
                self.contact_sensor = env.scene[target_sensor_name]
            except KeyError:
                self.contact_sensor = None
            if self.contact_sensor is None:
                num_out = env.num_envs if is_full else len(ids)
                return {
                    "jitter": jitter_score,
                    "impact_jerk": torch.zeros((num_out, 0), device=env.device),
                }

        force_history_w = self.contact_sensor.data.net_forces_w_history
        # [Critical Fix 1] Stride Calculation & Safety Clamping
        # 사용자가 설정한 history_length보다 요구하는 stride가 클 경우 크래시 방지
        target_stride = int(self.analysis_dt / physics_dt)
        available_history = force_history_w.shape[1] # T dimension
        # 유효한 인덱스는 0 ~ (available_history - 1)
        # stride는 현재(0)와 과거(stride)의 차이이므로, stride < available_history 여야 함.
        if target_stride >= available_history:
            # 안전하게 가장 오래된 데이터 사용 (Clamp)
            safe_stride = available_history - 1
            # stride가 0이 되는 것 방지 (최소 1)
            safe_stride = max(1, safe_stride)
        else:
            safe_stride = max(1, target_stride)
        # 데이터 추출 — env_ids 범위만
        force_t_w = force_history_w[ids, 0, :, :] # (N_sub, B, 3) Current
        force_prev_w = force_history_w[ids, safe_stride, :, :] # (N_sub, B, 3) Delayed
        # [Critical Fix 2] Body Frame Transformation
        # 로봇 Base(Root) 기준의 상대적 충격력 계산.
        # 이는 경사지에서 로봇이 기울어져 있을 때도, 로봇 입장에서의 '수직 충격'을 정확히 계산함.
        root_quat_w = self.robot_entity.data.root_quat_w[ids]
        inv_quat = root_quat_w.unsqueeze(1).expand(-1, force_t_w.shape[1], -1)
        force_t_b = quat_apply_inverse(inv_quat, force_t_w)
        force_prev_b = quat_apply_inverse(inv_quat, force_prev_w)
        # 수직 성분 (Normal Axis)
        fz_t = force_t_b[..., NORMAL_AXIS]
        fz_prev = force_prev_b[..., NORMAL_AXIS]
        # 충격 변화율(Jerk) 계산
        # Clamping된 stride에 맞춰 시간 간격도 재계산해야 물리적으로 정확함
        actual_time_delta = safe_stride * physics_dt
        impact_jerk = torch.abs(fz_t - fz_prev) / (actual_time_delta + EPS)
        # ---------------------------------------------------------------------
        # 3. 결과 반환 (env_ids 범위에 해당하는 텐서)
        # ---------------------------------------------------------------------
        return {
            "jitter": jitter_score, # (N_sub, Joints)
            "impact_jerk": impact_jerk # (N_sub, Bodies)
        }

    def reset(self, env_ids: torch.Tensor):
        """에피소드 리셋 시 통계 초기화."""
        if len(env_ids) == 0:
            return
        self.acc_mean_ema[env_ids] = 0.0
        self.acc_var_ema[env_ids] = 0.0
        self.is_ema_initialized[env_ids] = False
