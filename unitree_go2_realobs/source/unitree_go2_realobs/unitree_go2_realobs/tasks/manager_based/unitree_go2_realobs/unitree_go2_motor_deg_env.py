# =============================================================================
# unitree_go2_realobs/tasks/manager_based/unitree_go2_realobs/unitree_go2_motor_deg_env.py
# Motor-degradation Unitree Go2 RL environment.
#  1. Physics loop: degradation state is written into PhysX actuator state.
#  2. Control loop: critical-scenario command governor can modify commands.
#  3. Action path: optional post-unlatch smoothing reduces release spikes.
#  4. Diagnostics: optional terminal snapshots and debug monitors feed evaluation.
# =============================================================================

from __future__ import annotations

import os
import torch
import time
import logging
from typing import Any, Sequence

# [Isaac Lab Core]
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation

# [Motor-degradation components]
from .motor_deg.interface import (
    init_motor_deg_interface,
    update_motor_deg_dynamics,
    reset_motor_deg_interface,
    refresh_motor_deg_sensors,
    clear_motor_deg_step_metrics,
)
from .motor_deg.sat_latch import SatLatchCfg, SatRatioLatch

# [Motor-degradation constants]
from .motor_deg.constants import (
    T_AMB,
    BASE_HEIGHT_MIN,
    TEMP_WARN_THRESHOLD, 
    TEMP_CRITICAL_THRESHOLD,
    B_VISCOUS,
    WEAR_FRICTION_GAIN,
    ALPHA_CU,
    STICTION_NOMINAL,
    STICTION_WEAR_FACTOR,
    ALPHA_MAG,
)

# [Motor-degradation utils]
from .motor_deg.utils import compute_battery_voltage, compute_regenerative_efficiency, compute_component_losses


class UnitreeGo2MotorDegEnv(ManagerBasedRLEnv):
    """
    Real-observable motor-degradation environment for Unitree Go2.
    
    Architecture:
    - Physics Loop (200Hz): Implicit PD Control (PhysX), Degradation Injection.
    - Control Loop (50Hz): Command governor, action processing, sensor noise,
      and metric aggregation.
    """

    def load_managers(self):
        if not hasattr(self, "robot"):
            self.robot: Articulation = self.scene["robot"]
        if not hasattr(self, "motor_deg_state"):
            init_motor_deg_interface(self)

        super().load_managers()

    def __init__(self, cfg: Any, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._dbg_first_reset = bool(getattr(cfg, "debug_first_reset", False))
        self._dbg_first_step = bool(getattr(cfg, "debug_first_step", False))
        # Disabled by default to avoid heavy console I/O in large-scale training.
        self._debug_contact_force_monitor = bool(getattr(cfg, "debug_contact_force_monitor", False))
        # Disabled by default to avoid extra tensor copies in training.
        self._enable_terminal_snapshot = bool(getattr(cfg, "enable_terminal_snapshot", False))
        self._last_terminal_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._last_terminal_metrics: dict[str, torch.Tensor] = {}

        # ---------------------------------------------------------------------
        # [Lifecycle Phase 2] Degradation interface binding
        # ---------------------------------------------------------------------
        if not hasattr(self, "robot"):
            self.robot: Articulation = self.scene["robot"]
        if not hasattr(self, "motor_deg_state"):
            init_motor_deg_interface(self)
        
        if not hasattr(self, "motor_deg_joint_indices"):
            # [Fix] slice(None) fallback은 MotorDegState(num_joints=NUM_MOTORS=12)와
            # shape 불일치를 유발할 수 있으므로, NUM_MOTORS 기반 인덱스로 대체.
            import logging
            from .motor_deg.constants import NUM_MOTORS
            logging.warning(
                "[MotorDeg] motor_deg_joint_indices not set by init_motor_deg_interface. "
                f"Using range({NUM_MOTORS}) as fallback."
            )
            self.motor_deg_joint_indices = list(range(NUM_MOTORS))

        if isinstance(self.motor_deg_joint_indices, slice):
            self._motor_deg_joint_index_tensor = torch.arange(
                self.robot.data.joint_pos.shape[1], device=self.device, dtype=torch.long
            )
        else:
            self._motor_deg_joint_index_tensor = torch.as_tensor(self.motor_deg_joint_indices, device=self.device, dtype=torch.long)
        self._robot_to_motor_deg_local = {
            int(robot_idx): i for i, robot_idx in enumerate(self._motor_deg_joint_index_tensor.tolist())
        }

        # ---------------------------------------------------------------------
        # [Lifecycle Phase 3] Actuator Binding & Nominal Physics Backup
        # ---------------------------------------------------------------------
        self._nominal_effort_limits: torch.Tensor | None = None
        if hasattr(self.robot.data, "joint_effort_limits"):
             self._nominal_effort_limits = self.robot.data.joint_effort_limits.clone()
        
        self._nominal_stiffness = torch.zeros_like(self.robot.data.joint_pos)
        self._nominal_damping = torch.zeros_like(self.robot.data.joint_pos)
        found_motor_deg_actuator = False

        for _, actuator in self.robot.actuators.items():
            if hasattr(actuator, "bind_motor_deg_state"):
                actuator.bind_motor_deg_state(self.motor_deg_state)
                found_motor_deg_actuator = True

            if hasattr(actuator, "bind_asset"):
                actuator.bind_asset(self.robot)

            act_idx = getattr(actuator, "joint_indices", None)
            if act_idx is None:
                act_idx = getattr(actuator, "_joint_ids", None)
            if act_idx is None:
                act_idx = slice(None)

            k_p = getattr(actuator, "nominal_kp", actuator.stiffness)
            k_d = getattr(actuator, "nominal_kd", actuator.damping)

            if isinstance(k_p, torch.Tensor):
                k_p_t = k_p.to(self.device)
                if k_p_t.ndim == 0:
                    self._nominal_stiffness[:, act_idx] = float(k_p_t.item())
                elif k_p_t.ndim == 1:
                    self._nominal_stiffness[:, act_idx] = k_p_t.unsqueeze(0)
                elif k_p_t.ndim == 2 and k_p_t.shape[1] == self.robot.data.joint_pos.shape[1]:
                    self._nominal_stiffness[:, act_idx] = k_p_t[:, act_idx]
                else:
                    self._nominal_stiffness[:, act_idx] = k_p_t
            else:
                self._nominal_stiffness[:, act_idx] = float(k_p)

            if isinstance(k_d, torch.Tensor):
                k_d_t = k_d.to(self.device)
                if k_d_t.ndim == 0:
                    self._nominal_damping[:, act_idx] = float(k_d_t.item())
                elif k_d_t.ndim == 1:
                    self._nominal_damping[:, act_idx] = k_d_t.unsqueeze(0)
                elif k_d_t.ndim == 2 and k_d_t.shape[1] == self.robot.data.joint_pos.shape[1]:
                    self._nominal_damping[:, act_idx] = k_d_t[:, act_idx]
                else:
                    self._nominal_damping[:, act_idx] = k_d_t
            else:
                self._nominal_damping[:, act_idx] = float(k_d)

        if not found_motor_deg_actuator:
            raise RuntimeError("[Config Error] MotorDeg Actuator not found in env configuration.")

        # Optional external fault profile for replay/fault-injection experiments.
        num_motor_deg_joints = int(self._motor_deg_joint_index_tensor.numel())
        self._external_kp_scale = torch.ones((self.num_envs, num_motor_deg_joints), device=self.device)
        self._external_kd_scale = torch.ones((self.num_envs, num_motor_deg_joints), device=self.device)

        # ---------------------------------------------------------------------
        # [Lifecycle Phase 4] Double-PD Prevention -> REMOVED for Implicit PD
        # ---------------------------------------------------------------------
        # [Fix] Implicit 모드에서는 PhysX가 토크를 계산해야 하므로 게인을 0으로 만들지 않습니다.
        # self.robot.write_joint_stiffness_to_sim(0.0, joint_ids=self.motor_deg_joint_indices)
        # self.robot.write_joint_damping_to_sim(0.0, joint_ids=self.motor_deg_joint_indices)

        # [Perf] Cache scalar constants for brownout logic (avoid per-step tensor creation)
        # Keep thresholds aligned with replay/real safety policy:
        # - 24.5V: pack hard-stop neighborhood
        # - 25.0V: recovery hysteresis to prevent latch chattering
        self._const_true = torch.tensor(True, device=self.device)
        self._const_false = torch.tensor(False, device=self.device)
        self._const_brownout_scale_low = torch.tensor(0.5, device=self.device)
        self._const_brownout_scale_high = torch.tensor(1.0, device=self.device)
        self._brownout_enter_v = torch.tensor(float(getattr(cfg, "brownout_enter_v", 24.5)), device=self.device)
        self._brownout_recover_v = torch.tensor(float(getattr(cfg, "brownout_recover_v", 25.0)), device=self.device)
        self._brownout_voltage_source = str(getattr(cfg, "brownout_voltage_source", "bms_pred")).strip().lower()
        self._init_command_transport_dr(cfg)

        clear_motor_deg_step_metrics(self)
        self._reward_term_specs_cache = self._iter_active_reward_terms()
        self._init_velocity_command_curriculum(cfg)
        self._init_push_curriculum(cfg)
        self._init_dr_curriculum(cfg)
        self.enable_eval_gait_metrics = bool(getattr(cfg, "enable_eval_gait_metrics", False))
        self._eval_gait_contact_force_threshold = float(getattr(cfg, "eval_contact_force_threshold", 15.0))
        self._eval_nonzero_cmd_threshold = float(getattr(cfg, "eval_nonzero_cmd_threshold", 0.05))
        self._eval_stand_cmd_threshold = float(getattr(cfg, "eval_stand_cmd_threshold", 0.01))
        self._eval_gait_foot_body_names = getattr(cfg, "eval_foot_body_names", None)
        self._init_eval_gait_metric_buffers()
        self._init_critical_command_governor(cfg)
        self._validate_base_command_write_through_once()

    def clear_external_fault_profile(self):
        """Reset external actuator fault multipliers to nominal (1.0)."""
        self._external_kp_scale.fill_(1.0)
        self._external_kd_scale.fill_(1.0)

    def set_external_fault_profile(
        self, joint_names: Sequence[str], kp_scale: float = 1.0, kd_scale: float = 1.0
    ):
        """
        Apply constant external Kp/Kd scaling on selected joints.

        This is designed for reproducible fault-injection experiments.
        """
        self.clear_external_fault_profile()
        if len(joint_names) == 0:
            return

        robot_joint_ids, _ = self.robot.find_joints(list(joint_names))
        local_ids: list[int] = []
        for rid in robot_joint_ids:
            rid_int = int(rid.item()) if isinstance(rid, torch.Tensor) else int(rid)
            local = self._robot_to_motor_deg_local.get(rid_int, None)
            if local is not None:
                local_ids.append(local)

        if len(local_ids) == 0:
            raise ValueError(
                f"[FaultProfile] None of joints {list(joint_names)} are part of MotorDeg-controlled joints."
            )

        self._external_kp_scale[:, local_ids] = float(kp_scale)
        self._external_kd_scale[:, local_ids] = float(kd_scale)
        self._apply_physical_degradation()

    def reset(self, seed: int | None = None, options: dict | None = None):
        if getattr(self, "_dbg_first_reset", False):
            print(f"[DBG] Env.reset start (num_envs={self.num_envs})", flush=True)
            t0 = time.time()
        out = super().reset(seed=seed, options=options)
        if getattr(self, "_dbg_first_reset", False):
            print(f"[DBG] Env.reset done in {time.time() - t0:.2f}s", flush=True)
            self._dbg_first_reset = False
        return out

    def configure_eval_gait_metrics(
        self,
        enabled: bool | None = None,
        foot_body_names: Sequence[str] | None = None,
        contact_force_threshold: float | None = None,
        nonzero_cmd_threshold: float | None = None,
        stand_cmd_threshold: float | None = None,
    ) -> None:
        """Runtime configuration hook for evaluation-only gait diagnostics."""
        if enabled is not None:
            self.enable_eval_gait_metrics = bool(enabled)
        if foot_body_names is not None:
            self._eval_gait_foot_body_names = list(foot_body_names)
        if contact_force_threshold is not None:
            self._eval_gait_contact_force_threshold = float(contact_force_threshold)
        if nonzero_cmd_threshold is not None:
            self._eval_nonzero_cmd_threshold = float(nonzero_cmd_threshold)
        if stand_cmd_threshold is not None:
            self._eval_stand_cmd_threshold = float(stand_cmd_threshold)
        self._init_eval_gait_metric_buffers()

    def _get_eval_contact_sensor(self):
        if "contact_forces" in self.scene.sensors:
            return self.scene["contact_forces"]
        return None

    def _get_robot_body_names(self) -> list[str]:
        data = self.robot.data
        for key in ("body_names", "_body_names", "link_names", "_link_names"):
            names = getattr(data, key, None)
            if names is not None:
                try:
                    return list(names)
                except Exception:
                    pass
        return []

    def _resolve_eval_foot_body_names(self) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        sensor = self._get_eval_contact_sensor()
        if sensor is None or not hasattr(sensor, "body_names"):
            return [], torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        sensor_body_names = list(sensor.body_names)
        robot_body_names = self._get_robot_body_names()

        requested = self._eval_gait_foot_body_names
        if requested is None:
            resolved_names = [n for n in sensor_body_names if ("foot" in n.lower() or "toe" in n.lower())]
            resolved_names = resolved_names[:4]
        else:
            resolved_names = []
            for name in list(requested):
                if name in sensor_body_names:
                    resolved_names.append(name)
                else:
                    low = str(name).lower()
                    matches = [bn for bn in sensor_body_names if low in bn.lower()]
                    if len(matches) == 1:
                        resolved_names.append(matches[0])
            resolved_names = resolved_names[:4]

        if len(resolved_names) < 4:
            return [], torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        sensor_ids = [sensor_body_names.index(n) for n in resolved_names]
        robot_ids = []
        for n in resolved_names:
            if n in robot_body_names:
                robot_ids.append(robot_body_names.index(n))
            else:
                low = n.lower()
                matches = [i for i, bn in enumerate(robot_body_names) if low in bn.lower()]
                if len(matches) == 1:
                    robot_ids.append(matches[0])
                else:
                    return [], torch.empty(0, dtype=torch.long, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        return (
            resolved_names,
            torch.tensor(sensor_ids, dtype=torch.long, device=self.device),
            torch.tensor(robot_ids, dtype=torch.long, device=self.device),
        )

    def _init_eval_gait_metric_buffers(self) -> None:
        names, sensor_ids, robot_ids = self._resolve_eval_foot_body_names()
        self._eval_gait_resolved_foot_names = names
        self._eval_gait_foot_sensor_ids = sensor_ids
        self._eval_gait_foot_robot_ids = robot_ids
        self._eval_gait_ready = bool(sensor_ids.numel() >= 4 and robot_ids.numel() >= 4)

        n = self.num_envs
        f = int(sensor_ids.numel())
        self._eval_ep_steps = torch.zeros(n, dtype=torch.long, device=self.device)
        self._eval_cmd_speed_sum = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_actual_speed_sum = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_nonzero_cmd_steps = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_stand_cmd_steps = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_path_length = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_progress_distance = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_gait_valid_steps = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_gait_quad_support_steps = torch.zeros(n, dtype=torch.float32, device=self.device)
        self._eval_gait_touchdown_count = torch.zeros((n, f), dtype=torch.float32, device=self.device)
        self._eval_gait_slip_distance = torch.zeros((n, f), dtype=torch.float32, device=self.device)
        self._eval_prev_contact = torch.zeros((n, f), dtype=torch.bool, device=self.device)

        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_eval_gait_metric_buffers(all_env_ids)

    def _init_critical_command_governor(self, cfg: Any) -> None:
        """Initialize critical-scenario command governor state."""
        self._motor_deg_scenario_id = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._motor_deg_scenario_id_critical = int(getattr(cfg, "motor_deg_scenario_id_critical", 4))

        self._crit_governor_enable = bool(getattr(cfg, "critical_governor_enable", False))
        self._crit_v_cap_norm = float(max(getattr(cfg, "critical_governor_v_cap_norm", 0.15), 0.0))
        self._crit_wz_cap = float(max(getattr(cfg, "critical_governor_wz_cap", 0.0), 0.0))
        self._crit_ramp_tau_s = float(max(getattr(cfg, "critical_governor_ramp_tau_s", 2.0), 1e-6))
        self._crit_ramp_alpha = float(
            max(0.0, min(1.0, float(self.step_dt) / (float(self._crit_ramp_tau_s) + float(self.step_dt))))
        )
        self._crit_p_stand_high = float(max(0.0, min(1.0, getattr(cfg, "critical_governor_p_stand_high", 0.25))))
        self._crit_stand_trigger_norm = float(max(getattr(cfg, "critical_governor_stand_trigger_norm", 0.2), 0.0))

        self._crit_latch_hold_steps = max(int(getattr(cfg, "critical_governor_latch_hold_steps", 100)), 0)
        self._crit_unlatch_stable_steps_req = max(
            int(getattr(cfg, "critical_governor_unlatch_stable_steps", 50)), 1
        )
        self._crit_unlatch_cmd_norm = float(max(getattr(cfg, "critical_governor_unlatch_cmd_norm", 0.1), 0.0))
        self._crit_unlatch_require_low_cmd = bool(
            getattr(cfg, "critical_governor_unlatch_require_low_cmd", True)
        )
        self._crit_unlatch_require_sat_recovery = bool(
            getattr(cfg, "critical_governor_unlatch_require_sat_recovery", False)
        )
        self._crit_post_unlatch_action_ramp_s = float(
            max(getattr(cfg, "critical_governor_post_unlatch_action_ramp_s", 0.0), 0.0)
        )
        self._crit_post_unlatch_action_ramp_steps_total = max(
            int(round(self._crit_post_unlatch_action_ramp_s / max(float(self.step_dt), 1e-6))),
            0,
        )
        self._crit_post_unlatch_action_delta_max = float(
            max(getattr(cfg, "critical_governor_post_unlatch_action_delta_max", 0.0), 0.0)
        )
        self._crit_pose_roll_pitch_max_rad = float(
            max(getattr(cfg, "critical_governor_pose_roll_pitch_max_rad", 0.25), 0.0)
        )
        self._crit_pose_height_margin_m = float(max(getattr(cfg, "critical_governor_pose_height_margin_m", 0.05), 0.0))
        self._crit_safe_height_min = float(BASE_HEIGHT_MIN) + float(self._crit_pose_height_margin_m)
        self._crit_sat_thr = float(max(getattr(cfg, "critical_governor_sat_thr", 0.99), 0.0))
        self._crit_sat_window_steps = max(int(getattr(cfg, "critical_governor_sat_window_steps", 15)), 1)
        self._crit_sat_trigger = float(max(getattr(cfg, "critical_governor_sat_trigger", 0.95), 0.0))
        self._crit_sat_trigger_hi = float(
            max(getattr(cfg, "critical_governor_sat_trigger_hi", self._crit_sat_trigger), 0.0)
        )
        self._crit_sat_trigger_lo = float(
            max(
                0.0,
                min(
                    self._crit_sat_trigger_hi,
                    getattr(cfg, "critical_governor_sat_trigger_lo", self._crit_sat_trigger_hi),
                ),
            )
        )
        # Keep legacy field as alias of high trigger for downstream metadata compatibility.
        self._crit_sat_trigger = float(self._crit_sat_trigger_hi)
        self._crit_sat_latch = SatRatioLatch(
            num_envs=self.num_envs,
            device=self.device,
            cfg=SatLatchCfg(
                sat_thr=float(self._crit_sat_thr),
                window_steps=int(self._crit_sat_window_steps),
                trigger=float(self._crit_sat_trigger_hi),
            ),
        )
        self._crit_sat_any = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._crit_sat_ratio = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._crit_sat_over_trigger = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._crit_action_delta_norm_step = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._crit_cmd_delta_norm_step = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._crit_governor_mode_step = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._crit_cmd_delta_low_eps = float(
            max(getattr(cfg, "critical_governor_cmd_delta_low_eps", 1e-3), 0.0)
        )
        self._crit_warn_latched_cmd_delta_ratio = float(
            max(
                0.0,
                min(1.0, getattr(cfg, "critical_governor_warn_latched_cmd_delta_ratio", 0.90)),
            )
        )
        self._crit_warn_min_latched_frac = float(
            max(0.0, min(1.0, getattr(cfg, "critical_governor_warn_min_latched_frac", 0.30)))
        )
        self._crit_warn_every_steps = max(
            int(getattr(cfg, "critical_governor_warn_every_steps", 200)),
            1,
        )
        self._crit_warn_step_i = 0
        self._gov_sat_ratio_step: torch.Tensor | None = None
        self._gov_sat_any_over_1p00_step: torch.Tensor | None = None
        self._gov_sat_valid_step: torch.Tensor | None = None
        self._debug_gov_sat = os.getenv("DEBUG_GOV_SAT", "0") == "1"
        self._debug_gov_sat_every = max(int(os.getenv("DEBUG_GOV_SAT_EVERY", "25")), 1)
        self._debug_gov_sat_step_i = 0

        self._crit_cmd_raw = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._crit_cmd_eff = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._crit_last_cmd_eff = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self._crit_last_action_eff: torch.Tensor | None = None
        self._crit_episode_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._crit_is_stand_episode = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._crit_latch_steps_remaining = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._crit_need_unlatch = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self._crit_unlatch_stable_steps = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self._crit_post_unlatch_action_ramp_steps_remaining = torch.zeros(
            (self.num_envs,), dtype=torch.long, device=self.device
        )
        self._crit_action_transition_delta_norm_step = torch.zeros(
            (self.num_envs,), dtype=torch.float32, device=self.device
        )
        self._crit_latch_trigger_count_ep = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)

        self._base_velocity_cmd_write_through = False
        self._base_velocity_cmd_write_check_done = False
        self._crit_governor_cmd_write_warned = False
        if self._crit_governor_enable:
            logging.info(
                "[CriticalGovernor] enabled=True v_cap_norm=%.3f wz_cap=%.3f ramp_tau=%.2fs "
                "p_stand=%.2f stand_trigger=%.3f latch_hold=%d unlatch_steps=%d "
                "sat_trigger_hi=%.3f sat_trigger_lo=%.3f unlatch_low_cmd=%s unlatch_sat_recovery=%s "
                "post_unlatch_action_ramp_s=%.2f post_unlatch_action_delta_max=%.3f",
                self._crit_v_cap_norm,
                self._crit_wz_cap,
                self._crit_ramp_tau_s,
                self._crit_p_stand_high,
                self._crit_stand_trigger_norm,
                self._crit_latch_hold_steps,
                self._crit_unlatch_stable_steps_req,
                self._crit_sat_trigger_hi,
                self._crit_sat_trigger_lo,
                str(self._crit_unlatch_require_low_cmd),
                str(self._crit_unlatch_require_sat_recovery),
                self._crit_post_unlatch_action_ramp_s,
                self._crit_post_unlatch_action_delta_max,
            )

    def _reset_critical_command_governor(self, env_ids: torch.Tensor) -> None:
        """Reset governor episode states for selected environments."""
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self._crit_cmd_raw[ids] = 0.0
        self._crit_cmd_eff[ids] = 0.0
        self._crit_last_cmd_eff[ids] = 0.0
        if isinstance(self._crit_last_action_eff, torch.Tensor):
            self._crit_last_action_eff[ids] = 0.0
        self._crit_episode_steps[ids] = 0
        self._crit_is_stand_episode[ids] = False
        self._crit_latch_steps_remaining[ids] = 0
        self._crit_need_unlatch[ids] = False
        self._crit_unlatch_stable_steps[ids] = 0
        self._crit_post_unlatch_action_ramp_steps_remaining[ids] = 0
        self._crit_latch_trigger_count_ep[ids] = 0.0
        self._crit_sat_any[ids] = False
        self._crit_sat_ratio[ids] = 0.0
        self._crit_sat_over_trigger[ids] = False
        self._crit_action_delta_norm_step[ids] = 0.0
        self._crit_action_transition_delta_norm_step[ids] = 0.0
        self._crit_cmd_delta_norm_step[ids] = 0.0
        self._crit_governor_mode_step[ids] = 0
        self._crit_sat_latch.reset(ids)
        if isinstance(self._gov_sat_ratio_step, torch.Tensor):
            self._gov_sat_ratio_step[ids] = 0.0
        if isinstance(self._gov_sat_any_over_1p00_step, torch.Tensor):
            self._gov_sat_any_over_1p00_step[ids] = False
        if isinstance(self._gov_sat_valid_step, torch.Tensor):
            self._gov_sat_valid_step[ids] = False

    def _cache_governor_sat_inputs(self) -> None:
        """Cache per-step saturation tensors so eval diagnostics and governor use the same source."""
        sat = getattr(getattr(self, "motor_deg_state", None), "torque_saturation", None)
        if not isinstance(sat, torch.Tensor) or sat.ndim != 2 or sat.shape[0] != self.num_envs:
            self._gov_sat_ratio_step = None
            self._gov_sat_any_over_1p00_step = None
            self._gov_sat_valid_step = None
            return

        sat_ratio = sat.to(dtype=torch.float32, device=self.device)
        sat_valid = torch.isfinite(sat_ratio).all(dim=-1)
        sat_ratio = torch.nan_to_num(sat_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        sat_any_over_1p00 = (sat_ratio > 1.0).any(dim=-1) & sat_valid

        self._gov_sat_ratio_step = sat_ratio
        self._gov_sat_any_over_1p00_step = sat_any_over_1p00
        self._gov_sat_valid_step = sat_valid

    def _update_critical_sat_ratio_latch(self) -> None:
        """Update saturation ratio latch once per control step (not per physics substep)."""
        sat_ratio = getattr(self, "_gov_sat_ratio_step", None)
        sat_valid = getattr(self, "_gov_sat_valid_step", None)
        if not isinstance(sat_ratio, torch.Tensor):
            # Fallback path for safety when cache was not populated.
            self._cache_governor_sat_inputs()
            sat_ratio = getattr(self, "_gov_sat_ratio_step", None)
            sat_valid = getattr(self, "_gov_sat_valid_step", None)
        if not isinstance(sat_ratio, torch.Tensor):
            self._crit_sat_any[:] = False
            self._crit_sat_ratio[:] = 0.0
            self._crit_sat_over_trigger[:] = False
            return
        if sat_ratio.ndim != 2 or sat_ratio.shape[0] != self.num_envs:
            self._crit_sat_any[:] = False
            self._crit_sat_ratio[:] = 0.0
            self._crit_sat_over_trigger[:] = False
            return
        sat_any, sat_ratio_window, sat_over_trigger = self._crit_sat_latch.update(
            sat_ratio,
            valid_mask=sat_valid if isinstance(sat_valid, torch.Tensor) else None,
        )
        self._crit_sat_any[:] = sat_any
        # Keep mirrored cache for legacy readers; SSOT is SatRatioLatch.ratio.
        self._crit_sat_ratio[:] = self._crit_sat_latch.ratio
        self._crit_sat_over_trigger[:] = sat_over_trigger

        if self._debug_gov_sat and (self._debug_gov_sat_step_i % self._debug_gov_sat_every == 0):
            eval_any = getattr(self, "_gov_sat_any_over_1p00_step", None)
            eval_any_mean = float(eval_any.float().mean().item()) if isinstance(eval_any, torch.Tensor) else -1.0
            gov_any_mean = float(sat_any.float().mean().item())
            valid_mean = (
                float(sat_valid.float().mean().item())
                if isinstance(sat_valid, torch.Tensor)
                else 1.0
            )
            print(
                f"[GOV_SAT] step={self._debug_gov_sat_step_i} thr={self._crit_sat_thr:.3f} "
                f"sat_ratio(min/mean/max)=({float(sat_ratio.min().item()):.3f}/"
                f"{float(sat_ratio.mean().item()):.3f}/{float(sat_ratio.max().item()):.3f}) "
                f"eval_any>1.0={eval_any_mean:.3f} gov_any>thr={gov_any_mean:.3f} valid={valid_mean:.3f}",
                flush=True,
            )
            if isinstance(eval_any, torch.Tensor):
                mismatch = eval_any & (~sat_any)
                if isinstance(sat_valid, torch.Tensor):
                    mismatch = mismatch & sat_valid
                if bool(torch.any(mismatch).item()):
                    ids = mismatch.nonzero(as_tuple=False).flatten().tolist()
                    print(
                        f"[GOV_SAT_MISMATCH] env_ids={ids} "
                        "(eval_any>1.0 True but gov_any>thr False)",
                        flush=True,
                    )
        self._debug_gov_sat_step_i += 1

    def _validate_base_command_write_through_once(self) -> None:
        """
        Validate whether get_command('base_velocity') tensor supports in-place write-through.

        Always restores original command tensor before returning.
        """
        if self._base_velocity_cmd_write_check_done:
            return
        self._base_velocity_cmd_write_check_done = True
        self._base_velocity_cmd_write_through = False
        original_cmd = None
        try:
            cmd = self.command_manager.get_command("base_velocity")
            if not isinstance(cmd, torch.Tensor) or cmd.ndim < 2 or cmd.shape[0] == 0 or cmd.shape[1] < 3:
                return
            original_cmd = cmd.clone()
            probe_val = float(cmd[0, 0].item()) + 0.12345
            cmd[0, 0] = probe_val
            cmd_recheck = self.command_manager.get_command("base_velocity")
            if isinstance(cmd_recheck, torch.Tensor) and cmd_recheck.ndim >= 2:
                self._base_velocity_cmd_write_through = bool(
                    torch.isclose(cmd_recheck[0, 0], torch.tensor(probe_val, device=self.device), atol=1e-6).item()
                )
            cmd[:] = original_cmd
        except Exception as err:
            logging.warning("[CriticalGovernor] base_velocity write-through check failed: %s", err)
        finally:
            if original_cmd is not None:
                try:
                    cmd_restore = self.command_manager.get_command("base_velocity")
                    if isinstance(cmd_restore, torch.Tensor) and cmd_restore.shape == original_cmd.shape:
                        cmd_restore[:] = original_cmd
                except Exception:
                    pass

        logging.info(
            "[CriticalGovernor] base_velocity write_through=%s",
            bool(self._base_velocity_cmd_write_through),
        )

    def _get_base_velocity_command_tensor(self) -> torch.Tensor | None:
        """Resolve writable base-velocity command tensor."""
        try:
            cmd = self.command_manager.get_command("base_velocity")
        except Exception:
            return None

        if isinstance(cmd, torch.Tensor) and self._base_velocity_cmd_write_through:
            return cmd

        try:
            term = self.command_manager.get_term("base_velocity")
            for key in ("command", "_command", "commands", "_commands", "cmd", "_cmd"):
                buf = getattr(term, key, None)
                if isinstance(buf, torch.Tensor) and buf.ndim >= 2 and buf.shape[0] == self.num_envs and buf.shape[1] >= 3:
                    return buf
        except Exception:
            pass

        if (
            isinstance(cmd, torch.Tensor)
            and cmd.ndim >= 2
            and cmd.shape[0] == self.num_envs
            and cmd.shape[1] >= 3
            and self._base_velocity_cmd_write_through
        ):
            return cmd
        if not self._crit_governor_cmd_write_warned:
            logging.warning(
                "[CriticalGovernor] writable base_velocity command buffer not found; governor update skipped."
            )
            self._crit_governor_cmd_write_warned = True
        return None

    def _apply_critical_command_governor(self) -> None:
        """Apply critical-scenario command governor after command_manager.compute()."""
        cmd_buf = self._get_base_velocity_command_tensor()
        if cmd_buf is None or cmd_buf.ndim < 2 or cmd_buf.shape[1] < 3:
            return

        raw_cmd = cmd_buf[:, :3].clone()
        self._crit_cmd_raw[:] = raw_cmd
        self._crit_cmd_eff[:] = raw_cmd
        self._crit_governor_mode_step[:] = 0

        if not self._crit_governor_enable:
            self._crit_cmd_delta_norm_step[:] = 0.0
            if hasattr(self, "extras"):
                self.extras["cmd/base_velocity_raw"] = torch.mean(raw_cmd, dim=0)
                self.extras["cmd/base_velocity_eff"] = torch.mean(raw_cmd, dim=0)
                self.extras["crit/is_active"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_step"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_none"] = torch.tensor(1.0, device=self.device)
                self.extras["crit/governor_mode_v_cap"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_stand"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_stop_latch"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_other"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/sat_thr"] = torch.tensor(float(self._crit_sat_thr), device=self.device)
                self.extras["crit/sat_window_steps"] = torch.tensor(float(self._crit_sat_window_steps), device=self.device)
                self.extras["crit/sat_any_over_thr_ratio"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/sat_trigger"] = torch.tensor(float(self._crit_sat_trigger_hi), device=self.device)
                self.extras["crit/sat_trigger_hi"] = torch.tensor(float(self._crit_sat_trigger_hi), device=self.device)
                self.extras["crit/sat_trigger_lo"] = torch.tensor(float(self._crit_sat_trigger_lo), device=self.device)
                self.extras["crit/is_latched"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/cmd_delta_low_eps"] = torch.tensor(float(self._crit_cmd_delta_low_eps), device=self.device)
                self.extras["crit/latched_low_cmd_delta_ratio"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/action_delta_latched_norm_mean"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/action_delta_latched_norm_max"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/cmd_delta_latched_norm_mean"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/cmd_delta_latched_norm_max"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/post_unlatch_action_ramp_s"] = torch.tensor(
                    float(self._crit_post_unlatch_action_ramp_s), device=self.device
                )
                self.extras["crit/post_unlatch_action_ramp_active_ratio"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/post_unlatch_action_ramp_steps_remaining_mean"] = torch.tensor(
                    0.0, device=self.device
                )
                self.extras["crit/action_transition_delta_norm_mean"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/action_transition_delta_norm_max"] = torch.tensor(0.0, device=self.device)
            return

        critical_mask = self._motor_deg_scenario_id == int(self._motor_deg_scenario_id_critical)
        non_critical_mask = ~critical_mask
        if torch.any(non_critical_mask):
            self._crit_episode_steps[non_critical_mask] = 0
            self._crit_is_stand_episode[non_critical_mask] = False
            self._crit_latch_steps_remaining[non_critical_mask] = 0
            self._crit_need_unlatch[non_critical_mask] = False
            self._crit_unlatch_stable_steps[non_critical_mask] = 0
            self._crit_post_unlatch_action_ramp_steps_remaining[non_critical_mask] = 0
            self._crit_latch_trigger_count_ep[non_critical_mask] = 0.0

        if torch.any(critical_mask):
            raw_norm = torch.norm(raw_cmd[:, :2], dim=1)
            episode_start_mask = critical_mask & (self._crit_episode_steps <= 0)
            if torch.any(episode_start_mask):
                start_ids = episode_start_mask.nonzero(as_tuple=False).squeeze(-1)
                eligible = raw_norm[start_ids] >= float(self._crit_stand_trigger_norm)
                sampled = torch.rand((start_ids.numel(),), device=self.device) < float(self._crit_p_stand_high)
                self._crit_is_stand_episode[start_ids] = eligible & sampled
                self._crit_latch_steps_remaining[start_ids] = 0
                self._crit_need_unlatch[start_ids] = False
                self._crit_unlatch_stable_steps[start_ids] = 0

            target_cmd = raw_cmd.clone()
            raw_norm_safe = torch.clamp(raw_norm, min=1e-6)
            scale = torch.minimum(
                torch.ones_like(raw_norm_safe),
                torch.full_like(raw_norm_safe, float(self._crit_v_cap_norm)) / raw_norm_safe,
            )
            target_cmd[:, 0] = raw_cmd[:, 0] * scale
            target_cmd[:, 1] = raw_cmd[:, 1] * scale
            target_cmd[:, 2] = torch.clamp(raw_cmd[:, 2], -float(self._crit_wz_cap), float(self._crit_wz_cap))

            stand_mode_mask = critical_mask & self._crit_is_stand_episode
            if torch.any(stand_mode_mask):
                target_cmd[stand_mode_mask] = 0.0

            if self._crit_latch_hold_steps > 0:
                active_latch = self._crit_latch_steps_remaining > 0
                self._crit_latch_steps_remaining[active_latch] -= 1

            tilt = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
            if hasattr(self.robot.data, "projected_gravity_b"):
                gravity_b = self.robot.data.projected_gravity_b
                tilt = torch.acos(torch.clamp(-gravity_b[:, 2], min=-1.0, max=1.0))
            root_height = torch.full((self.num_envs,), float(self._crit_safe_height_min + 1.0), device=self.device)
            if hasattr(self.robot.data, "root_pos_w"):
                root_height = self.robot.data.root_pos_w[:, 2]

            brownout_latched = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            if hasattr(self.motor_deg_state, "brownout_latched"):
                brownout_latched = self.motor_deg_state.brownout_latched

            pose_unstable = (tilt > float(self._crit_pose_roll_pitch_max_rad)) | (
                root_height < float(self._crit_safe_height_min)
            )
            sat_unstable = self._crit_sat_over_trigger
            trigger_mask = critical_mask & (pose_unstable | sat_unstable | brownout_latched)
            if torch.any(trigger_mask):
                self._crit_latch_steps_remaining[trigger_mask] = int(self._crit_latch_hold_steps)
                self._crit_need_unlatch[trigger_mask] = True
                self._crit_unlatch_stable_steps[trigger_mask] = 0
                self._crit_latch_trigger_count_ep[trigger_mask] += 1.0

            hard_latch_mask = critical_mask & (self._crit_latch_steps_remaining > 0)
            post_hold_mask = critical_mask & self._crit_need_unlatch & (~hard_latch_mask)
            if torch.any(post_hold_mask):
                stable_now = (
                    (tilt <= float(self._crit_pose_roll_pitch_max_rad))
                    & (root_height >= float(self._crit_safe_height_min))
                )
                if self._crit_unlatch_require_low_cmd:
                    stable_now = stable_now & (raw_norm <= float(self._crit_unlatch_cmd_norm))
                if self._crit_unlatch_require_sat_recovery:
                    sat_ratio_now = self._crit_sat_latch.ratio
                    stable_now = stable_now & (sat_ratio_now <= float(self._crit_sat_trigger_lo))
                stable_ids = post_hold_mask.nonzero(as_tuple=False).squeeze(-1)
                prev = self._crit_unlatch_stable_steps[stable_ids]
                self._crit_unlatch_stable_steps[stable_ids] = torch.where(
                    stable_now[stable_ids],
                    prev + 1,
                    torch.zeros_like(prev),
                )
                unlatch_done = stable_ids[
                    self._crit_unlatch_stable_steps[stable_ids] >= int(self._crit_unlatch_stable_steps_req)
                ]
                if unlatch_done.numel() > 0:
                    self._crit_need_unlatch[unlatch_done] = False
                    self._crit_unlatch_stable_steps[unlatch_done] = 0
                    transition_steps = int(self._crit_post_unlatch_action_ramp_steps_total)
                    if transition_steps <= 0 and self._crit_post_unlatch_action_delta_max > 0.0:
                        # Clamp-only mode still needs one active transition step.
                        transition_steps = 1
                    if transition_steps > 0:
                        self._crit_post_unlatch_action_ramp_steps_remaining[unlatch_done] = int(
                            transition_steps
                        )

            soft_latch_mask = critical_mask & self._crit_need_unlatch & (self._crit_latch_steps_remaining <= 0)
            latched_mask = hard_latch_mask | soft_latch_mask

            eff_cmd = raw_cmd.clone()
            crit_ids = critical_mask.nonzero(as_tuple=False).squeeze(-1)
            eff_cmd[crit_ids] = self._crit_last_cmd_eff[crit_ids] + float(self._crit_ramp_alpha) * (
                target_cmd[crit_ids] - self._crit_last_cmd_eff[crit_ids]
            )
            if torch.any(latched_mask):
                eff_cmd[latched_mask] = 0.0
            eff_cmd[non_critical_mask] = raw_cmd[non_critical_mask]

            self._crit_last_cmd_eff[:] = eff_cmd
            self._crit_cmd_eff[:] = eff_cmd
            self._crit_cmd_delta_norm_step[:] = torch.norm(self._crit_cmd_eff - self._crit_cmd_raw, dim=1)
            cmd_buf[:, :3] = eff_cmd
            self._crit_episode_steps[critical_mask] += 1

            # Governor mode taxonomy per step:
            # 0=none, 1=v_cap, 2=stand, 3=stop_latch, 4=reserved_other.
            cap_xy_applied = scale < (1.0 - 1e-6)
            cap_wz_applied = torch.abs(raw_cmd[:, 2]) > (float(self._crit_wz_cap) + 1e-6)
            mode_step = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
            v_cap_mask = critical_mask & (cap_xy_applied | cap_wz_applied)
            mode_step[v_cap_mask] = 1
            mode_step[stand_mode_mask] = 2
            mode_step[latched_mask] = 3
            self._crit_governor_mode_step[:] = mode_step

        if hasattr(self, "extras"):
            self.extras["cmd/base_velocity_raw"] = torch.mean(self._crit_cmd_raw, dim=0)
            self.extras["cmd/base_velocity_eff"] = torch.mean(self._crit_cmd_eff, dim=0)
            self.extras["crit/is_active"] = torch.mean(critical_mask.float())
            self.extras["crit/is_stand_episode"] = torch.mean(self._crit_is_stand_episode.float())
            latched = (self._crit_latch_steps_remaining > 0) | self._crit_need_unlatch
            self.extras["crit/is_latched"] = torch.mean(latched.float())
            self.extras["crit/latch_steps_remaining"] = torch.mean(self._crit_latch_steps_remaining.float())
            self.extras["crit/latch_trigger_rate"] = torch.mean((self._crit_latch_trigger_count_ep > 0).float())
            self.extras["crit/cmd_raw_norm_mean"] = torch.mean(torch.norm(self._crit_cmd_raw[:, :2], dim=1))
            self.extras["crit/cmd_eff_norm_mean"] = torch.mean(torch.norm(self._crit_cmd_eff[:, :2], dim=1))
            self.extras["crit/post_unlatch_action_ramp_s"] = torch.tensor(
                float(self._crit_post_unlatch_action_ramp_s), device=self.device
            )
            self.extras["crit/cmd_delta_low_eps"] = torch.tensor(float(self._crit_cmd_delta_low_eps), device=self.device)
            critical_ramp_active = critical_mask & (self._crit_post_unlatch_action_ramp_steps_remaining > 0)
            if torch.any(latched):
                self.extras["crit/action_delta_latched_norm_mean"] = torch.mean(self._crit_action_delta_norm_step[latched])
                self.extras["crit/action_delta_latched_norm_max"] = torch.max(self._crit_action_delta_norm_step[latched])
                self.extras["crit/cmd_delta_latched_norm_mean"] = torch.mean(self._crit_cmd_delta_norm_step[latched])
                self.extras["crit/cmd_delta_latched_norm_max"] = torch.max(self._crit_cmd_delta_norm_step[latched])
                latched_low_cmd_delta_ratio = torch.mean(
                    (self._crit_cmd_delta_norm_step[latched] <= float(self._crit_cmd_delta_low_eps)).float()
                )
            else:
                self.extras["crit/action_delta_latched_norm_mean"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/action_delta_latched_norm_max"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/cmd_delta_latched_norm_mean"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/cmd_delta_latched_norm_max"] = torch.tensor(0.0, device=self.device)
                latched_low_cmd_delta_ratio = torch.tensor(0.0, device=self.device)
            self.extras["crit/latched_low_cmd_delta_ratio"] = latched_low_cmd_delta_ratio
            if torch.any(critical_mask):
                self.extras["crit/post_unlatch_action_ramp_active_ratio"] = torch.mean(
                    critical_ramp_active[critical_mask].float()
                )
                self.extras["crit/post_unlatch_action_ramp_steps_remaining_mean"] = torch.mean(
                    self._crit_post_unlatch_action_ramp_steps_remaining[critical_mask].float()
                )
                self.extras["crit/action_transition_delta_norm_mean"] = torch.mean(
                    self._crit_action_transition_delta_norm_step[critical_mask]
                )
                self.extras["crit/action_transition_delta_norm_max"] = torch.max(
                    self._crit_action_transition_delta_norm_step[critical_mask]
                )
            else:
                self.extras["crit/post_unlatch_action_ramp_active_ratio"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/post_unlatch_action_ramp_steps_remaining_mean"] = torch.tensor(
                    0.0, device=self.device
                )
                self.extras["crit/action_transition_delta_norm_mean"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/action_transition_delta_norm_max"] = torch.tensor(0.0, device=self.device)
            crit_count = torch.sum(critical_mask)
            if int(crit_count.item()) > 0:
                self.extras["crit/sat_any_over_thr_ratio"] = torch.mean(self._crit_sat_latch.ratio[critical_mask])
                self.extras["crit/is_latched"] = torch.mean(latched[critical_mask].float())
                self.extras["crit/sat_ratio_valid_steps"] = torch.mean(
                    self._crit_sat_latch.valid_steps[critical_mask].float()
                )
                mode_crit = self._crit_governor_mode_step[critical_mask]
                self.extras["crit/governor_mode_step"] = torch.mean(mode_crit.float())
                self.extras["crit/governor_mode_none"] = torch.mean((mode_crit == 0).float())
                self.extras["crit/governor_mode_v_cap"] = torch.mean((mode_crit == 1).float())
                self.extras["crit/governor_mode_stand"] = torch.mean((mode_crit == 2).float())
                self.extras["crit/governor_mode_stop_latch"] = torch.mean((mode_crit == 3).float())
                self.extras["crit/governor_mode_other"] = torch.mean((mode_crit == 4).float())
                latched_crit = latched[critical_mask]
                latched_frac = float(torch.mean(latched_crit.float()).item())
                low_delta_ratio = float(latched_low_cmd_delta_ratio.detach().cpu().item())
                if (
                    (self._crit_warn_step_i % int(self._crit_warn_every_steps)) == 0
                    and latched_frac >= float(self._crit_warn_min_latched_frac)
                    and low_delta_ratio >= float(self._crit_warn_latched_cmd_delta_ratio)
                ):
                    logging.warning(
                        "[CriticalGovernor] latched but low cmd-delta dominated: "
                        "latched_frac=%.3f low_delta_ratio=%.3f eps=%.4f",
                        latched_frac,
                        low_delta_ratio,
                        float(self._crit_cmd_delta_low_eps),
                    )
            else:
                self.extras["crit/sat_any_over_thr_ratio"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/sat_ratio_valid_steps"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_step"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_none"] = torch.tensor(1.0, device=self.device)
                self.extras["crit/governor_mode_v_cap"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_stand"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_stop_latch"] = torch.tensor(0.0, device=self.device)
                self.extras["crit/governor_mode_other"] = torch.tensor(0.0, device=self.device)
            self.extras["crit/sat_thr"] = torch.tensor(float(self._crit_sat_thr), device=self.device)
            self.extras["crit/sat_window_steps"] = torch.tensor(float(self._crit_sat_window_steps), device=self.device)
            self.extras["crit/sat_trigger"] = torch.tensor(float(self._crit_sat_trigger_hi), device=self.device)
            self.extras["crit/sat_trigger_hi"] = torch.tensor(float(self._crit_sat_trigger_hi), device=self.device)
            self.extras["crit/sat_trigger_lo"] = torch.tensor(float(self._crit_sat_trigger_lo), device=self.device)
        self._crit_warn_step_i += 1

    def _apply_critical_post_unlatch_action_transition(self, action: torch.Tensor) -> torch.Tensor:
        """Smooth action release right after unlatch to reduce transition spikes."""
        if not isinstance(action, torch.Tensor):
            return action
        if (
            not isinstance(self._crit_last_action_eff, torch.Tensor)
            or self._crit_last_action_eff.shape != action.shape
        ):
            self._crit_last_action_eff = torch.zeros_like(action)
        if (not self._crit_governor_enable) or (
            self._crit_post_unlatch_action_ramp_steps_total <= 0 and self._crit_post_unlatch_action_delta_max <= 0.0
        ):
            self._crit_action_transition_delta_norm_step[:] = 0.0
            self._crit_last_action_eff[:] = action.detach()
            return action

        out = action.clone()
        critical_mask = self._motor_deg_scenario_id == int(self._motor_deg_scenario_id_critical)
        ramp_active = critical_mask & (self._crit_post_unlatch_action_ramp_steps_remaining > 0)

        if torch.any(ramp_active):
            ids = ramp_active.nonzero(as_tuple=False).squeeze(-1)
            prev = self._crit_last_action_eff[ids]
            tgt = action[ids]

            total = max(int(self._crit_post_unlatch_action_ramp_steps_total), 1)
            rem = self._crit_post_unlatch_action_ramp_steps_remaining[ids].to(torch.float32)
            alpha = (float(total) - rem + 1.0) / float(total)
            alpha = torch.clamp(alpha, min=0.0, max=1.0).unsqueeze(-1)

            blended = prev + alpha * (tgt - prev)
            if self._crit_post_unlatch_action_delta_max > 0.0:
                delta = torch.clamp(
                    blended - prev,
                    min=-float(self._crit_post_unlatch_action_delta_max),
                    max=float(self._crit_post_unlatch_action_delta_max),
                )
                blended = prev + delta
            out[ids] = blended
            self._crit_post_unlatch_action_ramp_steps_remaining[ids] = torch.clamp(
                self._crit_post_unlatch_action_ramp_steps_remaining[ids] - 1,
                min=0,
            )

        self._crit_action_transition_delta_norm_step[:] = torch.norm(out - action, dim=1)
        self._crit_last_action_eff[:] = out.detach()
        return out

    def _reset_eval_gait_metric_buffers(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self._eval_ep_steps[ids] = 0
        self._eval_cmd_speed_sum[ids] = 0.0
        self._eval_actual_speed_sum[ids] = 0.0
        self._eval_nonzero_cmd_steps[ids] = 0.0
        self._eval_stand_cmd_steps[ids] = 0.0
        self._eval_path_length[ids] = 0.0
        self._eval_progress_distance[ids] = 0.0
        self._eval_gait_valid_steps[ids] = 0.0
        self._eval_gait_quad_support_steps[ids] = 0.0
        if self._eval_gait_touchdown_count.shape[1] > 0:
            self._eval_gait_touchdown_count[ids] = 0.0
            self._eval_gait_slip_distance[ids] = 0.0
            self._eval_prev_contact[ids] = False

            sensor = self._get_eval_contact_sensor()
            if sensor is not None and hasattr(sensor.data, "net_forces_w") and self._eval_gait_ready:
                forces = sensor.data.net_forces_w[ids][:, self._eval_gait_foot_sensor_ids, :]
                contact_force = torch.norm(forces, dim=-1)
                self._eval_prev_contact[ids] = contact_force > self._eval_gait_contact_force_threshold

    def _update_eval_gait_metric_buffers(self) -> None:
        if not self.enable_eval_gait_metrics or not self._enable_terminal_snapshot:
            return

        cmd = self.command_manager.get_command("base_velocity")
        cmd_xy = cmd[:, :2]
        cmd_speed = torch.norm(cmd_xy, dim=1)
        vel_xy = self.robot.data.root_lin_vel_b[:, :2]
        actual_speed = torch.norm(vel_xy, dim=1)
        cmd_dir = cmd_xy / (cmd_speed.unsqueeze(-1) + 1e-6)
        progress_speed = torch.sum(vel_xy * cmd_dir, dim=1)
        progress_speed = torch.where(cmd_speed > 1e-3, progress_speed, torch.zeros_like(progress_speed))

        self._eval_ep_steps += 1
        self._eval_cmd_speed_sum += cmd_speed
        self._eval_actual_speed_sum += actual_speed
        self._eval_nonzero_cmd_steps += (cmd_speed > self._eval_nonzero_cmd_threshold).to(torch.float32)
        self._eval_stand_cmd_steps += (cmd_speed < self._eval_stand_cmd_threshold).to(torch.float32)
        self._eval_path_length += actual_speed * float(self.step_dt)
        self._eval_progress_distance += torch.clamp(progress_speed, min=0.0) * float(self.step_dt)

        sensor = self._get_eval_contact_sensor()
        if sensor is None or not self._eval_gait_ready:
            return
        if not hasattr(sensor.data, "net_forces_w") or not hasattr(self.robot.data, "body_lin_vel_w"):
            return

        forces = sensor.data.net_forces_w[:, self._eval_gait_foot_sensor_ids, :]
        contact_force = torch.norm(forces, dim=-1)
        thr_on = float(self._eval_gait_contact_force_threshold)
        thr_off = 0.6 * thr_on
        prev_contact = self._eval_prev_contact
        contact = torch.where(prev_contact, contact_force > thr_off, contact_force > thr_on)
        touchdown = (~prev_contact) & contact
        self._eval_gait_touchdown_count += touchdown.to(torch.float32)
        self._eval_prev_contact = contact
        self._eval_gait_quad_support_steps += torch.all(contact, dim=1).to(torch.float32)
        self._eval_gait_valid_steps += 1.0

        foot_vel_xy = self.robot.data.body_lin_vel_w[:, self._eval_gait_foot_robot_ids, :2]
        foot_speed_xy = torch.norm(foot_vel_xy, dim=-1)
        self._eval_gait_slip_distance += contact.to(torch.float32) * foot_speed_xy * float(self.step_dt)

    def _init_velocity_command_curriculum(self, cfg: Any) -> None:
        """Initialize staged velocity-command range widening for locomotion training."""
        self._vel_cmd_curriculum_enable = bool(getattr(cfg, "velocity_cmd_curriculum_enable", True))
        self._vel_cmd_steps_per_iter = max(int(getattr(cfg, "velocity_cmd_curriculum_steps_per_iter", 24)), 1)
        self._vel_cmd_start_iter = max(int(getattr(cfg, "velocity_cmd_curriculum_start_iter", 160)), 0)
        self._vel_cmd_ramp_iters = max(int(getattr(cfg, "velocity_cmd_curriculum_ramp_iters", 340)), 0)
        self._vel_cmd_start_step = self._vel_cmd_start_iter * self._vel_cmd_steps_per_iter
        self._vel_cmd_ramp_steps = self._vel_cmd_ramp_iters * self._vel_cmd_steps_per_iter

        self._vel_cmd_initial_lin_x: tuple[float, float] = (-0.1, 0.1)
        self._vel_cmd_initial_lin_y: tuple[float, float] = (-0.1, 0.1)
        self._vel_cmd_initial_ang_z: tuple[float, float] = (-1.0, 1.0)

        self._vel_cmd_target_lin_x = tuple(getattr(cfg, "velocity_cmd_target_lin_vel_x", (-1.0, 1.0)))
        self._vel_cmd_target_lin_y = tuple(getattr(cfg, "velocity_cmd_target_lin_vel_y", (-0.4, 0.4)))
        self._vel_cmd_target_ang_z = tuple(getattr(cfg, "velocity_cmd_target_ang_vel_z", (-1.0, 1.0)))

        self._vel_cmd_last_alpha = -1.0
        self._vel_cmd_curriculum_started = False
        self._vel_cmd_curriculum_reached_target = False

        try:
            vel_term = self.command_manager.get_term("base_velocity")
            if hasattr(vel_term, "cfg") and hasattr(vel_term.cfg, "ranges"):
                rng = vel_term.cfg.ranges
                if hasattr(rng, "lin_vel_x"):
                    self._vel_cmd_initial_lin_x = tuple(rng.lin_vel_x)
                if hasattr(rng, "lin_vel_y"):
                    self._vel_cmd_initial_lin_y = tuple(rng.lin_vel_y)
                if hasattr(rng, "ang_vel_z"):
                    self._vel_cmd_initial_ang_z = tuple(rng.ang_vel_z)
        except Exception:
            # If command manager is not ready, keep safe defaults.
            pass

        if self._vel_cmd_curriculum_enable:
            logging.info(
                "[VelCmdCurriculum] enabled=True start_iter=%d ramp_iters=%d initial=(%s,%s,%s) target=(%s,%s,%s)",
                self._vel_cmd_start_iter,
                self._vel_cmd_ramp_iters,
                self._vel_cmd_initial_lin_x,
                self._vel_cmd_initial_lin_y,
                self._vel_cmd_initial_ang_z,
                self._vel_cmd_target_lin_x,
                self._vel_cmd_target_lin_y,
                self._vel_cmd_target_ang_z,
            )

    @staticmethod
    def _lerp_range(src: tuple[float, float], dst: tuple[float, float], alpha: float) -> tuple[float, float]:
        a = float(max(0.0, min(1.0, alpha)))
        return (
            float(src[0]) + (float(dst[0]) - float(src[0])) * a,
            float(src[1]) + (float(dst[1]) - float(src[1])) * a,
        )

    def _update_velocity_command_curriculum(self) -> None:
        """Expand base-velocity command ranges after a warmup period."""
        if not self._vel_cmd_curriculum_enable:
            return

        try:
            vel_term = self.command_manager.get_term("base_velocity")
        except Exception:
            return
        if not hasattr(vel_term, "cfg") or not hasattr(vel_term.cfg, "ranges"):
            return

        step = int(self.common_step_counter)
        if self._vel_cmd_ramp_steps <= 0:
            alpha = 1.0 if step >= self._vel_cmd_start_step else 0.0
        else:
            alpha = (step - self._vel_cmd_start_step) / float(self._vel_cmd_ramp_steps)
            alpha = float(max(0.0, min(1.0, alpha)))

        # Keep updates cheap when alpha does not change numerically.
        if abs(alpha - self._vel_cmd_last_alpha) <= 1e-9:
            return

        rng = vel_term.cfg.ranges
        rng.lin_vel_x = self._lerp_range(self._vel_cmd_initial_lin_x, self._vel_cmd_target_lin_x, alpha)
        rng.lin_vel_y = self._lerp_range(self._vel_cmd_initial_lin_y, self._vel_cmd_target_lin_y, alpha)
        rng.ang_vel_z = self._lerp_range(self._vel_cmd_initial_ang_z, self._vel_cmd_target_ang_z, alpha)

        # Force one-time immediate resample at curriculum transitions to avoid long stale commands.
        crossed_start = (self._vel_cmd_last_alpha <= 0.0) and (alpha > 0.0)
        reached_target = (self._vel_cmd_last_alpha < 1.0) and (alpha >= 1.0)
        if crossed_start or reached_target:
            vel_term.time_left[:] = 0.0

        if crossed_start and not self._vel_cmd_curriculum_started:
            self._vel_cmd_curriculum_started = True
            logging.info(
                "[VelCmdCurriculum] started at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._vel_cmd_steps_per_iter),
            )
        if reached_target and not self._vel_cmd_curriculum_reached_target:
            self._vel_cmd_curriculum_reached_target = True
            logging.info(
                "[VelCmdCurriculum] reached target at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._vel_cmd_steps_per_iter),
            )

        if hasattr(self, "extras"):
            self.extras["cmd/vel_curriculum_alpha"] = torch.tensor(alpha, device=self.device)
            self.extras["cmd/lin_vel_x_max"] = torch.tensor(float(rng.lin_vel_x[1]), device=self.device)
            self.extras["cmd/lin_vel_y_max"] = torch.tensor(float(rng.lin_vel_y[1]), device=self.device)
            self.extras["cmd/ang_vel_z_max"] = torch.tensor(float(rng.ang_vel_z[1]), device=self.device)

        self._vel_cmd_last_alpha = alpha

    def _init_push_curriculum(self, cfg: Any) -> None:
        """Initialize staged push-disturbance range widening for locomotion stability."""
        self._push_curriculum_enable = bool(getattr(cfg, "push_curriculum_enable", True))
        self._push_steps_per_iter = max(int(getattr(cfg, "push_curriculum_steps_per_iter", 24)), 1)
        self._push_start_iter = max(int(getattr(cfg, "push_curriculum_start_iter", 501)), 0)
        self._push_ramp_iters = max(int(getattr(cfg, "push_curriculum_ramp_iters", 499)), 0)
        self._push_start_step = self._push_start_iter * self._push_steps_per_iter
        self._push_ramp_steps = self._push_ramp_iters * self._push_steps_per_iter
        self._push_initial_xy = tuple(getattr(cfg, "push_curriculum_initial_xy", (0.0, 0.0)))
        self._push_target_xy = tuple(getattr(cfg, "push_curriculum_target_xy", (-0.5, 0.5)))
        self._push_last_alpha = -1.0
        self._push_curriculum_started = False
        self._push_curriculum_reached_target = False
        self._push_cfg_lookup_warned = False
        self._push_cfg_set_warned = False

        if self._push_curriculum_enable:
            logging.info(
                "[PushCurriculum] enabled=True start_iter=%d ramp_iters=%d initial_xy=%s target_xy=%s",
                self._push_start_iter,
                self._push_ramp_iters,
                self._push_initial_xy,
                self._push_target_xy,
            )

    def _init_dr_curriculum(self, cfg: Any) -> None:
        """Initialize staged DR widening (friction/mass/command-delay)."""
        self._dr_curriculum_enable = bool(getattr(cfg, "dr_curriculum_enable", True))
        self._dr_steps_per_iter = max(int(getattr(cfg, "dr_curriculum_steps_per_iter", 24)), 1)
        self._dr_start_iter = max(int(getattr(cfg, "dr_curriculum_start_iter", 501)), 0)
        self._dr_ramp_iters = max(int(getattr(cfg, "dr_curriculum_ramp_iters", 499)), 0)
        self._dr_start_step = self._dr_start_iter * self._dr_steps_per_iter
        self._dr_ramp_steps = self._dr_ramp_iters * self._dr_steps_per_iter
        self._dr_initial_friction_range = tuple(
            getattr(cfg, "dr_curriculum_initial_friction_range", (0.6, 1.25))
        )
        self._dr_target_friction_range = tuple(
            getattr(cfg, "dr_curriculum_target_friction_range", (0.5, 1.3))
        )
        self._dr_initial_mass_scale_range = tuple(
            getattr(cfg, "dr_curriculum_initial_mass_scale_range", (0.9, 1.1))
        )
        self._dr_target_mass_scale_range = tuple(
            getattr(cfg, "dr_curriculum_target_mass_scale_range", (0.8, 1.2))
        )
        self._dr_initial_cmd_delay_steps = max(
            int(getattr(cfg, "dr_curriculum_initial_cmd_delay_max_steps", self._cmd_delay_max_steps)),
            0,
        )
        self._dr_target_cmd_delay_steps = max(
            int(getattr(cfg, "dr_curriculum_target_cmd_delay_max_steps", self._cmd_delay_max_steps)),
            0,
        )
        # Ensure delay DR starts from the intended initial bound.
        self._cmd_delay_max_steps = self._dr_initial_cmd_delay_steps

        self._dr_last_alpha = -1.0
        self._dr_curriculum_started = False
        self._dr_curriculum_reached_target = False
        self._event_cfg_lookup_warned: dict[str, bool] = {}
        self._event_cfg_set_warned: dict[str, bool] = {}

        if self._dr_curriculum_enable:
            logging.info(
                "[DRCurriculum] enabled=True start_iter=%d ramp_iters=%d friction=%s->%s mass_scale=%s->%s "
                "delay_steps=%d->%d",
                self._dr_start_iter,
                self._dr_ramp_iters,
                self._dr_initial_friction_range,
                self._dr_target_friction_range,
                self._dr_initial_mass_scale_range,
                self._dr_target_mass_scale_range,
                self._dr_initial_cmd_delay_steps,
                self._dr_target_cmd_delay_steps,
            )

    def _init_command_transport_dr(self, cfg: Any) -> None:
        """Initialize command transport DR (delay/dropout as network/control jitter proxy)."""
        self._cmd_transport_dr_enable = bool(getattr(cfg, "cmd_transport_dr_enable", True))
        self._cmd_delay_max_steps = max(int(getattr(cfg, "cmd_delay_max_steps", 1)), 0)
        self._cmd_dropout_prob = float(getattr(cfg, "cmd_dropout_prob", 0.005))
        self._cmd_dropout_prob = float(max(0.0, min(1.0, self._cmd_dropout_prob)))
        self._cmd_delay_buffer: torch.Tensor | None = None
        self._cmd_last_applied: torch.Tensor | None = None

    def _apply_command_transport_dr(self, action: torch.Tensor) -> torch.Tensor:
        """Apply per-env random command delay and packet-drop hold."""
        if not self._cmd_transport_dr_enable:
            return action

        num_envs, num_actions = action.shape
        hist_len = max(self._cmd_delay_max_steps + 1, 1)

        if (
            self._cmd_delay_buffer is None
            or self._cmd_delay_buffer.shape[0] != hist_len
            or self._cmd_delay_buffer.shape[1] != num_envs
            or self._cmd_delay_buffer.shape[2] != num_actions
        ):
            self._cmd_delay_buffer = action.unsqueeze(0).repeat(hist_len, 1, 1).clone()
            self._cmd_last_applied = action.clone()

        self._cmd_delay_buffer = torch.roll(self._cmd_delay_buffer, shifts=1, dims=0)
        self._cmd_delay_buffer[0] = action

        if self._cmd_delay_max_steps > 0:
            delay_steps = torch.randint(
                low=0,
                high=self._cmd_delay_max_steps + 1,
                size=(num_envs,),
                device=self.device,
            )
        else:
            delay_steps = torch.zeros((num_envs,), dtype=torch.long, device=self.device)

        env_idx = torch.arange(num_envs, device=self.device)
        delayed_action = self._cmd_delay_buffer[delay_steps, env_idx, :]

        if self._cmd_dropout_prob > 0.0:
            drop_mask = torch.rand((num_envs,), device=self.device) < self._cmd_dropout_prob
            if torch.any(drop_mask):
                delayed_action = delayed_action.clone()
                if self._cmd_last_applied is None:
                    self._cmd_last_applied = delayed_action.clone()
                delayed_action[drop_mask] = self._cmd_last_applied[drop_mask]
        else:
            drop_mask = torch.zeros((num_envs,), dtype=torch.bool, device=self.device)

        self._cmd_last_applied = delayed_action.detach().clone()
        if hasattr(self, "extras"):
            self.extras["dr/cmd_delay_steps_mean"] = torch.mean(delay_steps.float())
            self.extras["dr/cmd_drop_rate"] = torch.mean(drop_mask.float())

        return delayed_action

    def _get_event_term_cfg(self, term_name: str) -> tuple[Any | None, str]:
        """Resolve event term cfg via EventManager API, with cfg fallback."""
        mgr = getattr(self, "event_manager", None)
        if mgr is not None and hasattr(mgr, "get_term_cfg"):
            try:
                return mgr.get_term_cfg(term_name), "manager"
            except Exception as err:
                if not self._event_cfg_lookup_warned.get(term_name, False):
                    logging.warning(
                        "[DRCurriculum] Failed to read event term '%s' from EventManager; falling back to cfg path. "
                        "error=%s",
                        term_name,
                        err,
                    )
                    self._event_cfg_lookup_warned[term_name] = True

        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, term_name):
            return getattr(events_cfg, term_name), "cfg"
        return None, "none"

    def _set_event_term_cfg(self, term_name: str, term_cfg: Any, source: str) -> None:
        """Write event term cfg back through public manager API when available."""
        if source != "manager":
            return
        mgr = getattr(self, "event_manager", None)
        if mgr is None or not hasattr(mgr, "set_term_cfg"):
            return
        try:
            mgr.set_term_cfg(term_name, term_cfg)
        except Exception as err:
            if not self._event_cfg_set_warned.get(term_name, False):
                logging.warning(
                    "[DRCurriculum] Failed to write event term '%s' through EventManager; keeping cfg-side update. "
                    "error=%s",
                    term_name,
                    err,
                )
                self._event_cfg_set_warned[term_name] = True

    def _update_dr_curriculum(self) -> None:
        """Widen DR range in the 501~1000-iter phase (friction/mass/latency)."""
        if not self._dr_curriculum_enable:
            return

        step = int(self.common_step_counter)
        if self._dr_ramp_steps <= 0:
            alpha = 1.0 if step >= self._dr_start_step else 0.0
        else:
            alpha = (step - self._dr_start_step) / float(self._dr_ramp_steps)
            alpha = float(max(0.0, min(1.0, alpha)))

        if abs(alpha - self._dr_last_alpha) <= 1e-9:
            return

        friction_range = self._lerp_range(self._dr_initial_friction_range, self._dr_target_friction_range, alpha)
        mass_scale_range = self._lerp_range(self._dr_initial_mass_scale_range, self._dr_target_mass_scale_range, alpha)
        delay_steps_f = (
            float(self._dr_initial_cmd_delay_steps)
            + (float(self._dr_target_cmd_delay_steps) - float(self._dr_initial_cmd_delay_steps)) * alpha
        )
        self._cmd_delay_max_steps = max(int(round(delay_steps_f)), 0)

        # Update physics-material randomization bounds.
        material_term, source = self._get_event_term_cfg("physics_material")
        if material_term is not None:
            params = getattr(material_term, "params", None)
            if isinstance(params, dict):
                params["static_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
                params["dynamic_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
            self._set_event_term_cfg("physics_material", material_term, source)

        # Update mass randomization scale bounds.
        mass_term, source = self._get_event_term_cfg("add_mass")
        if mass_term is not None:
            params = getattr(mass_term, "params", None)
            if isinstance(params, dict):
                params["mass_distribution_params"] = (float(mass_scale_range[0]), float(mass_scale_range[1]))
                params["operation"] = "scale"
            self._set_event_term_cfg("add_mass", mass_term, source)

        # Keep cfg-side mirror synchronized for readability.
        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None:
            if hasattr(events_cfg, "physics_material"):
                term = getattr(events_cfg, "physics_material")
                params = getattr(term, "params", None)
                if isinstance(params, dict):
                    params["static_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
                    params["dynamic_friction_range"] = (float(friction_range[0]), float(friction_range[1]))
            if hasattr(events_cfg, "add_mass"):
                term = getattr(events_cfg, "add_mass")
                params = getattr(term, "params", None)
                if isinstance(params, dict):
                    params["mass_distribution_params"] = (float(mass_scale_range[0]), float(mass_scale_range[1]))
                    params["operation"] = "scale"

        crossed_start = (self._dr_last_alpha <= 0.0) and (alpha > 0.0)
        reached_target = (self._dr_last_alpha < 1.0) and (alpha >= 1.0)
        if crossed_start and not self._dr_curriculum_started:
            self._dr_curriculum_started = True
            logging.info(
                "[DRCurriculum] started at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._dr_steps_per_iter),
            )
        if reached_target and not self._dr_curriculum_reached_target:
            self._dr_curriculum_reached_target = True
            logging.info(
                "[DRCurriculum] reached target at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._dr_steps_per_iter),
            )

        if hasattr(self, "extras"):
            self.extras["dr/curriculum_alpha"] = torch.tensor(alpha, device=self.device)
            self.extras["dr/friction_min"] = torch.tensor(float(friction_range[0]), device=self.device)
            self.extras["dr/friction_max"] = torch.tensor(float(friction_range[1]), device=self.device)
            self.extras["dr/mass_scale_min"] = torch.tensor(float(mass_scale_range[0]), device=self.device)
            self.extras["dr/mass_scale_max"] = torch.tensor(float(mass_scale_range[1]), device=self.device)
            self.extras["dr/cmd_delay_max_steps"] = torch.tensor(float(self._cmd_delay_max_steps), device=self.device)

        self._dr_last_alpha = alpha

    def _get_push_event_term_cfg(self) -> tuple[Any | None, str]:
        """Resolve push event term cfg via stable public API, with cfg fallback."""
        mgr = getattr(self, "event_manager", None)
        if mgr is not None and hasattr(mgr, "get_term_cfg"):
            try:
                term_cfg = mgr.get_term_cfg("push_robot")
                return term_cfg, "manager"
            except Exception as err:
                if not self._push_cfg_lookup_warned:
                    logging.warning(
                        "[PushCurriculum] Failed to read push_robot term from EventManager; "
                        "falling back to cfg path. error=%s",
                        err,
                    )
                    self._push_cfg_lookup_warned = True

        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, "push_robot"):
            return getattr(events_cfg, "push_robot"), "cfg"

        return None, "none"

    def _update_push_curriculum(self) -> None:
        """Ramp external push disturbance from easy to target range."""
        if not self._push_curriculum_enable:
            return

        step = int(self.common_step_counter)
        if self._push_ramp_steps <= 0:
            alpha = 1.0 if step >= self._push_start_step else 0.0
        else:
            alpha = (step - self._push_start_step) / float(self._push_ramp_steps)
            alpha = float(max(0.0, min(1.0, alpha)))

        if abs(alpha - self._push_last_alpha) <= 1e-9:
            return

        push_xy = self._lerp_range(self._push_initial_xy, self._push_target_xy, alpha)
        push_term_cfg, source = self._get_push_event_term_cfg()
        if push_term_cfg is not None:
            params = getattr(push_term_cfg, "params", None)
            if isinstance(params, dict):
                vel_range = params.get("velocity_range", None)
                if isinstance(vel_range, dict):
                    vel_range["x"] = (float(push_xy[0]), float(push_xy[1]))
                    vel_range["y"] = (float(push_xy[0]), float(push_xy[1]))

            # Persist back to manager through public API.
            if source == "manager":
                mgr = getattr(self, "event_manager", None)
                if mgr is not None and hasattr(mgr, "set_term_cfg"):
                    try:
                        mgr.set_term_cfg("push_robot", push_term_cfg)
                    except Exception as err:
                        if not self._push_cfg_set_warned:
                            logging.warning(
                                "[PushCurriculum] Failed to write updated push_robot term via EventManager. "
                                "Using cfg-side update only. error=%s",
                                err,
                            )
                            self._push_cfg_set_warned = True

        # Keep config copy synchronized for logging/debug readability.
        events_cfg = getattr(self.cfg, "events", None)
        if events_cfg is not None and hasattr(events_cfg, "push_robot"):
            cfg_term = getattr(events_cfg, "push_robot")
            cfg_params = getattr(cfg_term, "params", None)
            if isinstance(cfg_params, dict):
                cfg_vel = cfg_params.get("velocity_range", None)
                if isinstance(cfg_vel, dict):
                    cfg_vel["x"] = (float(push_xy[0]), float(push_xy[1]))
                    cfg_vel["y"] = (float(push_xy[0]), float(push_xy[1]))

        crossed_start = (self._push_last_alpha <= 0.0) and (alpha > 0.0)
        reached_target = (self._push_last_alpha < 1.0) and (alpha >= 1.0)
        if crossed_start and not self._push_curriculum_started:
            self._push_curriculum_started = True
            logging.info(
                "[PushCurriculum] started at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._push_steps_per_iter),
            )
        if reached_target and not self._push_curriculum_reached_target:
            self._push_curriculum_reached_target = True
            logging.info(
                "[PushCurriculum] reached target at step=%d (iter≈%.1f)",
                step,
                float(step) / float(self._push_steps_per_iter),
            )

        if hasattr(self, "extras"):
            self.extras["cmd/push_curriculum_alpha"] = torch.tensor(alpha, device=self.device)
            self.extras["cmd/push_vel_xy_max"] = torch.tensor(float(push_xy[1]), device=self.device)

        self._push_last_alpha = alpha

    def _apply_physical_degradation(self, env_ids=None):
        """
        [MotorDeg Dynamics Core - Physics Loop (200Hz)]
        Updates the GROUND TRUTH hardware state based on accumulated physics.
        This enables Implicit PD to handle degraded physics seamlessly.
        
        [Fix] 이중 적분 제거: update_motor_deg_dynamics()가 이미 fatigue_index와 coil_temp를
        적분 완료했으므로, 여기서는 현재 상태를 그대로 사용하여 게인을 계산합니다.
        이전에는 한 번 더 Euler step을 적용하여 관측-물리 비대칭이 발생했습니다.
        """
        ids = slice(None) if env_ids is None else env_ids

        # 1. 현재 MotorDeg 상태 직접 사용 (이미 update_motor_deg_dynamics에서 적분 완료)
        fatigue = torch.clamp(self.motor_deg_state.fatigue_index[ids], min=0.0)
        temp = torch.clamp(self.motor_deg_state.coil_temp[ids], min=25.0)

        # 2. Actuator Gain Calculation (Physical Degradation)
        # [Fix #1] 연속적 열 감쇠 모델: voltage predictor의 저항 모델(ALPHA_CU)과 동일한 물리 사용.
        #   - T_AMB(25°C)~75°C: 저항 증가에 의한 점진적 게인 감소 (1/(1+α*ΔT))
        #   - 75°C~90°C: 추가 derating (절연 열화 / 감자 모사)
        # 이전: 75°C 이하에서 gain=1.0 (계단식) → 전압 예측과 물리적 비대칭 발생.
        gain_factor_fatigue = torch.clamp(1.0 - (fatigue * 0.6), min=0.2)
        # (a) 연속적 저항 기반 감쇠: gain ∝ 1/R ∝ 1/(1 + α*(T-T_AMB))
        resistance_derating = 1.0 / (1.0 + ALPHA_CU * (temp - T_AMB))
        # (a2) 자석 감자 효과: Kt ∝ (1 + α_mag*(T-T_AMB)), α_mag < 0이므로 고온 시 Kt 감소
        magnet_derating = torch.clamp(1.0 + ALPHA_MAG * (temp - T_AMB), min=0.5, max=1.0)
        # (b) 고온 구간 추가 derating (TEMP_WARN ~ TEMP_CRITICAL → 1.0 ~ 0.0)
        # [Fix] min=0.05: 90°C에서도 최소 5% 제어 권한을 유지하여
        # termination check 전에 Kp=0으로 인한 제어 불능 붕괴를 방지.
        severe_derating = torch.clamp(
            1.0 - (temp - TEMP_WARN_THRESHOLD) / max(TEMP_CRITICAL_THRESHOLD - TEMP_WARN_THRESHOLD, 1e-6),
            min=0.05, max=1.0,
        )
        gain_factor_thermal = resistance_derating * magnet_derating * severe_derating
        
        # [Fix #6] motor_deg_joint_indices로 슬라이싱하여 (N, all_joints) vs (N, NUM_MOTORS) shape mismatch 방지
        joint_idx = self.motor_deg_joint_indices
        nominal_kp = self._nominal_stiffness[ids][:, joint_idx] if not isinstance(joint_idx, slice) else self._nominal_stiffness[ids]
        current_kp = nominal_kp * gain_factor_fatigue * gain_factor_thermal
        # [Fix] 베어링 마모 시 Kd 하한을 Kp(min=0.2)보다 높게 유지 (min=0.4)
        # 물리: 마모 → 유격 증가로 감쇠가 줄어들지만, Kd가 지나치게 낮으면
        # 제어 불안정(진동/발산)이 발생하므로 안정성 확보를 위해 하한을 보존함.
        kd_factor_wear = torch.clamp(gain_factor_fatigue, min=0.4)
        nominal_kd = self._nominal_damping[ids][:, joint_idx] if not isinstance(joint_idx, slice) else self._nominal_damping[ids]
        current_kd = nominal_kd * kd_factor_wear * gain_factor_thermal

        # Apply optional external fault profile.
        current_kp = current_kp * self._external_kp_scale[ids]
        current_kd = current_kd * self._external_kd_scale[ids]

        # 3. Apply to Shared State (State for Observation/Reward)
        self.motor_deg_state.degraded_stiffness[ids] = current_kp
        self.motor_deg_state.degraded_damping[ids] = current_kd

        # 4. [CRITICAL] Apply to PhysX Simulator (Implicit Control Injection)
        # Actuator가 'joint_efforts=None'을 반환하므로, PhysX는 이 게인을 사용하여
        # 내부적으로 토크를 계산합니다. (Skating 제거 + 열화 반영)
        if hasattr(self.robot, "write_joint_stiffness_to_sim"):
            self.robot.write_joint_stiffness_to_sim(current_kp, joint_ids=self.motor_deg_joint_indices, env_ids=env_ids)
            self.robot.write_joint_damping_to_sim(current_kd, joint_ids=self.motor_deg_joint_indices, env_ids=env_ids)

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        [MotorDeg-Aware Dual-Rate Step Function]
        """
        
        if getattr(self, "_dbg_first_step", False):
            print(f"[DBG] Env.step start action_shape={tuple(action.shape)} decimation={self.cfg.decimation}", flush=True)
            t_step0 = time.time()

        if self._enable_terminal_snapshot:
            self._last_terminal_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self._last_terminal_metrics = {}

        # ---------------------------------------------------------------------
        # 1. Reset Step Accumulators (NOT Persistent State)
        # ---------------------------------------------------------------------
        if hasattr(self, "motor_deg_state") and self.motor_deg_state is not None:
            clear_motor_deg_step_metrics(self) 
            
            # Reset Log buffer for this step (Must be >= max possible voltage for min-tracking)
            if hasattr(self.motor_deg_state, "min_voltage_log"):
                self.motor_deg_state.min_voltage_log[:] = 33.6

        # ---------------------------------------------------------------------
        # 2. Action Processing & Sensor Sample-and-Hold
        # ---------------------------------------------------------------------
        action_device = action.to(self.device)
        action_in = self._apply_command_transport_dr(action_device)
        action_in = self._apply_critical_post_unlatch_action_transition(action_in)
        self._crit_action_delta_norm_step[:] = torch.norm(action_in - action_device, dim=1)
        self.action_manager.process_action(action_in)
        self.recorder_manager.record_pre_step()
        
        # [Fix 1] Sensor Noise Synchronization
        if hasattr(self, "motor_deg_state"):
            refresh_motor_deg_sensors(self)

        # ---------------------------------------------------------------------
        # 3. Physics Decimation Loop (e.g., 4 steps x 5ms)
        # ---------------------------------------------------------------------
        for substep in range(self.cfg.decimation):
            # (A) Apply Action FIRST (substep 0 only)
            # [Fix #9] apply_action()을 voltage prediction보다 먼저 호출.
            # 이전: BMS가 joint_pos_target을 읽기 전에 apply_action이 호출되지 않아
            # 이전 step의 stale target으로 전압을 예측하여 brownout 판정이 1-step 지연됨.
            if substep == 0:
                self.action_manager.apply_action()

            # (B) Voltage Prediction & Brownout Logic (Control-step rate, substep 0 only)
            if substep == 0:
                # BMS Perception (Model Mismatch for Logic)
                # joint_pos_target이 이미 현재 action으로 설정된 상태
                v_bms_pred = self._predict_instant_voltage_ivp(use_noisy_state=True, use_nominal_model=True)

                # [Fix #8] Cache BMS predicted voltage for strategic observations.
                # Brownout source is configurable via cfg.brownout_voltage_source.
                if hasattr(self.motor_deg_state, "bms_voltage_pred"):
                    self.motor_deg_state.bms_voltage_pred[:] = v_bms_pred

                if hasattr(self.motor_deg_state, "brownout_scale"):
                    current_scale = self.motor_deg_state.brownout_scale
                    latched = self.motor_deg_state.brownout_latched

                    if self._brownout_voltage_source == "true_voltage" and hasattr(self.motor_deg_state, "battery_voltage_true"):
                        v_for_brownout = self.motor_deg_state.battery_voltage_true
                    elif self._brownout_voltage_source == "sensor_voltage" and hasattr(self.motor_deg_state, "battery_voltage"):
                        v_for_brownout = self.motor_deg_state.battery_voltage
                    else:
                        v_for_brownout = v_bms_pred

                    # Latch Logic
                    is_low_voltage = v_for_brownout < self._brownout_enter_v
                    is_recovered = v_for_brownout > self._brownout_recover_v

                    new_latch = torch.where(is_low_voltage, self._const_true, latched)
                    new_latch = torch.where(is_recovered, self._const_false, new_latch)
                    self.motor_deg_state.brownout_latched[:] = new_latch

                    # Scale Target
                    target_scale = torch.where(
                        new_latch, self._const_brownout_scale_low, self._const_brownout_scale_high
                    )

                    # LPF Smoothing
                    alpha = 0.1
                    new_scale = alpha * target_scale + (1.0 - alpha) * current_scale
                    self.motor_deg_state.brownout_scale[:] = new_scale

            # [Fix #9] thermal_limits와 degradation을 substep 0에서만 계산.
            # 이전: 매 substep마다 재계산하여 4x 성능 낭비 (substep 간 fatigue/temp 변화 미미).
            # 수정: substep 0에서 1회 계산 후 나머지 substep에서 재사용.
            if substep == 0:
                if hasattr(self.motor_deg_state, "brownout_scale"):
                    thermal_limits = self._compute_thermal_limits()
                    final_limits = thermal_limits * self.motor_deg_state.brownout_scale.unsqueeze(-1)

                    if hasattr(self.robot, "write_joint_effort_limit_to_sim"):
                        self.robot.write_joint_effort_limit_to_sim(final_limits, env_ids=None)

                    self.robot.data.joint_effort_limits[:] = final_limits

                # (C) Apply Degradation BEFORE sim step (once per control step)
                self._apply_physical_degradation()

            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            
            # (D) Dynamics Integration (sim.step 이후 fresh torques/velocities로 상태 갱신)
            update_motor_deg_dynamics(self, self.physics_dt)

            # (E) Min-Voltage Tracking (Fix #5)
            # 매 substep의 실제 battery_voltage로 최저값 추적.
            # 이전: substep 0의 PD 추정 기반 예측값만 캡처하여 실제 최저 전압과 괴리.
            if hasattr(self.motor_deg_state, "min_voltage_log"):
                # [Fix #9] biased voltage 대신 true voltage 추적 (센서 바이어스 오염 방지)
                voltage_for_log = getattr(self.motor_deg_state, "battery_voltage_true", self.motor_deg_state.battery_voltage)
                self.motor_deg_state.min_voltage_log[:] = torch.min(
                    self.motor_deg_state.min_voltage_log, voltage_for_log
                )
            
            self._sim_step_counter += 1

        # ---------------------------------------------------------------------
        # 4. Post-Step Processing (RL Loop)
        # ---------------------------------------------------------------------
        
        # Rendering
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()

        # Update Timers
        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Evaluation-only gait/motion evidence buffers.
        self._update_eval_gait_metric_buffers()
        # Cache per-step saturation source once so governor and eval diagnostics share it.
        self._cache_governor_sat_inputs()
        # Update saturation latch before terminations/snapshot so terminal metrics
        # can see the current control-step saturation evidence.
        self._update_critical_sat_ratio_latch()

        # Terminations & Rewards
        self.reset_buf = self.termination_manager.compute()
        self.rew_buf = self.reward_manager.compute(dt=self.step_dt)

        # Resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._cache_terminal_snapshot(reset_env_ids)
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            self.recorder_manager.record_post_reset(reset_env_ids)

        # Command & Event Updates
        self._update_velocity_command_curriculum()
        self._update_dr_curriculum()
        self._update_push_curriculum()
        self.command_manager.compute(dt=self.step_dt)
        self._apply_critical_command_governor()
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # Observations
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # ---------------------------------------------------------------------
        # [DEBUG] Real-time Contact Force Monitor (Method 2)
        # ---------------------------------------------------------------------
        if self._debug_contact_force_monitor and self.common_step_counter % 60 == 0:
            # 1. Access Sensor Data
            if "contact_forces" in self.scene.sensors:
                sensor = self.scene["contact_forces"]
                
                # 2. Extract Z-force for Env 0 (Assuming [Env, Body, Axis])
                # Note: net_forces_w gives raw physics data, safer for debug
                if hasattr(sensor.data, "net_forces_w"):
                    net_forces = sensor.data.net_forces_w[0, :, 2] # Env 0, All Bodies, Z-axis
                    
                    # 3. Filter Active Contacts (> 1.0 N)
                    # We use >1.0N to filter out numerical noise/floating limbs
                    active_indices = torch.nonzero(net_forces > 1.0, as_tuple=False).squeeze(-1)
                    
                    if len(active_indices) > 0:
                        forces_str = ", ".join([f"{net_forces[i]:.1f}" for i in active_indices])
                        print(f"[Step {self.common_step_counter}] Active feet forces (N): [{forces_str}]")
                    else:
                        print(f"[Step {self.common_step_counter}] Warning: no ground contact detected")

        self.recorder_manager.record_post_step()

        # Metrics Logging
        self._log_motor_deg_metrics()

        if getattr(self, "_dbg_first_step", False):
            print(f"[DBG] Env.step done in {time.time() - t_step0:.2f}s", flush=True)
            self._dbg_first_step = False

        return self.obs_buf, self.rew_buf, self.termination_manager.terminated, self.termination_manager.time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """
        Reset the selected environments while keeping task/MotorDeg reset scopes separated.
        """
        if isinstance(env_ids, slice):
            env_ids_t = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        # Curriculum gate input: count only non-timeout resets as instability signal.
        # Timeout resets are normal episode rollovers and should not freeze difficulty.
        non_timeout_count = float(env_ids_t.numel())
        try:
            if hasattr(self, "termination_manager") and hasattr(self.termination_manager, "time_outs"):
                time_out_mask = self.termination_manager.time_outs[env_ids_t]
                non_timeout_count = float(torch.sum(~time_out_mask).item())
        except Exception:
            pass
        self._curriculum_non_timeout_resets = non_timeout_count
        self._curriculum_reset_count = int(env_ids_t.numel())

        super()._reset_idx(env_ids_t)

        # Persistent State Reset (Fatigue, Latch, etc.)
        reset_motor_deg_interface(self, env_ids_t)

        # [Fix #9] 리셋 환경의 센서 노이즈 재샘플링.
        # state.reset()이 encoder_noise/encoder_vel_noise를 0으로 초기화하므로,
        # 여기서 재샘플링하지 않으면 리셋 직후 첫 관측이 노이즈 없는 ground truth가 됨.
        # 이는 에이전트가 "리셋 직후에는 완벽한 센서 데이터" 패턴을 학습하는 원인.
        refresh_motor_deg_sensors(self, env_ids=env_ids_t)

        # Command transport DR buffer cleanup for reset environments.
        if self._cmd_delay_buffer is not None:
            self._cmd_delay_buffer[:, env_ids_t, :] = 0.0
        if self._cmd_last_applied is not None:
            self._cmd_last_applied[env_ids_t] = 0.0

        # Note: step_energy_log, avg_power_log, friction_power, stall_timer,
        # brownout_scale, min_voltage_log are already reset inside reset_motor_deg_interface().
        # Only reset fields NOT covered by reset_motor_deg_interface:
        if hasattr(self, "motor_deg_state"):
            if hasattr(self.motor_deg_state, "instant_power"):
                self.motor_deg_state.instant_power[env_ids_t] = 0.0

        # [Fix] 커리큘럼이 non-zero fatigue/temp를 설정했을 수 있으므로,
        # 리셋 직후 degraded gains를 즉시 재계산하여 observation-physics 일관성 확보
        if hasattr(self, "motor_deg_state") and self.motor_deg_state is not None:
            # [Fix #6] 리셋 환경만 대상으로 degradation 재계산
            self._apply_physical_degradation(env_ids=env_ids_t)

            # [Fix #4] 리셋 환경만 대상으로 effort limits 재계산.
            # 이전: 전체 환경의 limits를 덮어써서 nominal 복원이 무효화되고
            # 비리셋 환경에 불필요한 불연속이 발생했음.
            if hasattr(self.motor_deg_state, "brownout_scale"):
                # [Fix #9] 리셋 대상 환경만 thermal limits 계산 (전체 4096 환경 연산 방지)
                thermal_limits = self._compute_thermal_limits(env_ids=env_ids_t)
                final_limits = thermal_limits * self.motor_deg_state.brownout_scale[env_ids_t].unsqueeze(-1)
                self.robot.data.joint_effort_limits[env_ids_t] = final_limits
                if hasattr(self.robot, "write_joint_effort_limit_to_sim"):
                    self.robot.write_joint_effort_limit_to_sim(final_limits, env_ids=env_ids_t)

        # Evaluation-only gait/motion evidence buffers.
        self._reset_eval_gait_metric_buffers(env_ids_t)
        self._reset_critical_command_governor(env_ids_t)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _predict_instant_voltage_ivp(self, use_noisy_state: bool = False, use_nominal_model: bool = False) -> torch.Tensor:
        """
        [Fix 3] Instant Voltage Prediction with Model Mismatch Capability
        
        Electrical/mechanical losses are computed via shared compute_component_losses() (SSOT).
        Only friction computation is mode-specific (nominal vs actual) for model mismatch.
        """
        
        # 1. State Source Selection (Sensor Noise)
        joint_dim_robot = self.robot.data.joint_pos.shape[1]
        gt_pos = self.robot.data.joint_pos[:, self.motor_deg_joint_indices]
        gt_vel = self.robot.data.joint_vel[:, self.motor_deg_joint_indices]
        if use_noisy_state:
            # [Fix] BMS는 전압 센서에서 SOC를 역추정하므로, 전압 센서 바이어스가 SOC 오차 유발
            # OCV 기울기 ≈ 9.6V/100% → 전압 바이어스 1V ≈ SOC 오차 ~0.104
            soc_bias = getattr(self.motor_deg_state, "voltage_sensor_bias", None)
            if soc_bias is not None and isinstance(soc_bias, torch.Tensor):
                soc_estimation_error = soc_bias / 9.6  # V → SOC fraction
                soc = torch.clamp(self.motor_deg_state.soc + soc_estimation_error, 0.0, 1.0)
            else:
                soc = self.motor_deg_state.soc

            if hasattr(self.motor_deg_state, "encoder_meas_pos") and hasattr(self.motor_deg_state, "encoder_meas_vel"):
                current_pos = self.motor_deg_state.encoder_meas_pos
                joint_vel = self.motor_deg_state.encoder_meas_vel
            else:
                pos_noise = getattr(self.motor_deg_state, "encoder_noise", 0.0)
                pos_offset = getattr(self.motor_deg_state, "encoder_offset", 0.0)
                if isinstance(pos_noise, torch.Tensor):
                    if pos_noise.ndim == 2 and pos_noise.shape[1] == joint_dim_robot:
                        pos_noise = pos_noise[:, self.motor_deg_joint_indices]
                if isinstance(pos_offset, torch.Tensor):
                    if pos_offset.ndim == 2 and pos_offset.shape[1] == joint_dim_robot:
                        pos_offset = pos_offset[:, self.motor_deg_joint_indices]
                current_pos = gt_pos + pos_offset + pos_noise

                vel_noise = getattr(self.motor_deg_state, "encoder_vel_noise", 0.0)
                if isinstance(vel_noise, torch.Tensor):
                    if vel_noise.ndim == 2 and vel_noise.shape[1] == joint_dim_robot:
                        vel_noise = vel_noise[:, self.motor_deg_joint_indices]
                joint_vel = gt_vel + vel_noise
        else:
            soc = self.motor_deg_state.soc
            current_pos = gt_pos
            joint_vel = gt_vel

        # 2. Parameter Source Selection (Model Mismatch Logic)
        if use_nominal_model:
            # BMS View: Unknown degradation, assumes Nominal specs
            # _nominal_stiffness is (N, all_joints) → slice to MotorDeg joints
            kp = self._nominal_stiffness 
            kd = self._nominal_damping
            if isinstance(kp, torch.Tensor) and kp.ndim == 2 and kp.shape[1] == joint_dim_robot:
                kp = kp[:, self.motor_deg_joint_indices]
            if isinstance(kd, torch.Tensor) and kd.ndim == 2 and kd.shape[1] == joint_dim_robot:
                kd = kd[:, self.motor_deg_joint_indices]
        else:
            # Physics View: Actual degraded hardware
            # degraded_stiffness is already (N, NUM_MOTORS) → no slicing needed
            kp = self.motor_deg_state.degraded_stiffness
            kd = self.motor_deg_state.degraded_damping

        # 3. Torque Source Selection
        if use_nominal_model:
            # BMS path: PD 공식으로 토크 추정 (PhysX 내부 데이터 접근 불가)
            target_pos = self.robot.data.joint_pos_target[:, self.motor_deg_joint_indices]
            est_torque_demand = kp * (target_pos - current_pos) - kd * joint_vel
            torque_clamp = self._nominal_effort_limits[:, self.motor_deg_joint_indices] if self._nominal_effort_limits is not None else 23.7
            est_torque_demand = torch.clamp(est_torque_demand, -torque_clamp, torque_clamp)
        else:
            # [Fix #3] Ground truth: PhysX가 실제 적용한 토크 사용.
            # 이전: PD 공식 추정 토크를 사용하여 interface.py의 전력 계산과 체계적 괴리 발생.
            # PhysX applied_torque는 접촉력, 관절 한계, 수치 솔버를 모두 반영한 실제 값.
            est_torque_demand = self.robot.data.applied_torque[:, self.motor_deg_joint_indices]

        # 4. Power Calculation (SSOT: compute_component_losses from utils.py)
        vel_abs = torch.abs(joint_vel)
        
        # (a) Mode-specific friction (model mismatch by design)
        base_stiction = getattr(self.motor_deg_state, "base_friction_torque", None)
        if base_stiction is not None and isinstance(base_stiction, torch.Tensor):
            if base_stiction.ndim == 2:
                base_stiction = base_stiction[:, self.motor_deg_joint_indices]
        else:
            base_stiction = STICTION_NOMINAL

        if use_nominal_model:
            # BMS는 로봇별 실제 마찰을 모르므로 공칭 스펙 사용
            p_friction = B_VISCOUS * vel_abs * vel_abs + STICTION_NOMINAL * vel_abs
            temp_for_loss = T_AMB
        else:
            fatigue = torch.clamp(self.motor_deg_state.fatigue_index[:, self.motor_deg_joint_indices] if self.motor_deg_state.fatigue_index.ndim == 2 else self.motor_deg_state.fatigue_index, min=0.0)
            friction_bias = getattr(self.motor_deg_state, "friction_bias", None)
            if friction_bias is not None:
                if friction_bias.ndim == 2:
                    friction_bias = friction_bias[:, self.motor_deg_joint_indices]
            else:
                friction_bias = 1.0
            viscous_coeff = B_VISCOUS * (1.0 + fatigue * WEAR_FRICTION_GAIN) * friction_bias
            stiction_val = base_stiction * (1.0 + fatigue * STICTION_WEAR_FACTOR) * friction_bias
            p_friction = viscous_coeff * vel_abs * vel_abs + stiction_val * vel_abs
            coil_temp = self.motor_deg_state.coil_temp[:, self.motor_deg_joint_indices] if self.motor_deg_state.coil_temp.ndim == 2 else self.motor_deg_state.coil_temp
            temp_for_loss = coil_temp
        
        # (b) Electrical + mechanical losses via shared utility (SSOT with interface.py)
        p_copper, p_inverter, p_mech_total = compute_component_losses(
            torque=est_torque_demand,
            velocity=joint_vel,
            temp=temp_for_loss,
            external_friction_power=p_friction
        )
        
        # (c) Mechanical work with regen efficiency
        gross_mechanical_power = est_torque_demand * joint_vel
        net_mechanical_power = gross_mechanical_power - p_friction
        regen_eff = compute_regenerative_efficiency(joint_vel)
        mechanical_load = torch.where(
            net_mechanical_power < 0,
            net_mechanical_power * regen_eff,
            net_mechanical_power
        )
        
        # (d) Total Power = all losses + net mechanical work
        est_power_load = torch.sum(
            p_copper + p_inverter + p_mech_total + mechanical_load, dim=1
        )
        
        return compute_battery_voltage(soc, est_power_load)

    def _compute_thermal_limits(self, env_ids=None) -> torch.Tensor:
        """
        Computes physical torque limits based on Temperature, Fatigue, and Wear.
        [Fix #2] Fatigue에 의한 게인 감소도 effort limit에 반영.
        [Fix #6] 모든 MotorDeg 텐서를 joint_idx로 슬라이싱하여 shape 안전성 확보.
        [Fix #9] env_ids 지원: _reset_idx에서 리셋 대상만 계산하여 불필요한 전체 연산 방지.
        
        Args:
            env_ids: 특정 환경만 계산할 경우 지정. None이면 전체 환경 계산.
        """
        if self._nominal_effort_limits is None: 
            if env_ids is None:
                return self.robot.data.joint_effort_limits.clone()
            return self.robot.data.joint_effort_limits[env_ids].clone()

        ids = slice(None) if env_ids is None else env_ids
        limits = self._nominal_effort_limits[ids].clone()
        joint_idx = self.motor_deg_joint_indices

        # [Fix #6] 모든 MotorDeg 텐서를 joint_idx로 슬라이싱 (shape mismatch 방지)
        coil_temp = self.motor_deg_state.coil_temp[ids]
        coil_temp = coil_temp[:, joint_idx] if coil_temp.ndim == 2 else coil_temp
        fatigue = torch.nan_to_num(self.motor_deg_state.fatigue_index[ids], nan=0.0).clamp(min=0.0)
        fatigue = fatigue[:, joint_idx] if fatigue.ndim == 2 else fatigue
        friction_bias = self.motor_deg_state.friction_bias[ids]
        friction_bias = friction_bias[:, joint_idx] if friction_bias.ndim == 2 else friction_bias
        # [Fix #1 consistency] 연속적 열 감쇠 + 고온 derating (voltage predictor와 동일 물리)
        resistance_derating = 1.0 / (1.0 + ALPHA_CU * (coil_temp - T_AMB))
        # [Fix] min=0.05: _apply_physical_degradation과 동일 — 제어 불능 방지.
        severe_derating = torch.clamp(
            1.0 - (coil_temp - TEMP_WARN_THRESHOLD) / max(TEMP_CRITICAL_THRESHOLD - TEMP_WARN_THRESHOLD, 1e-6),
            min=0.05, max=1.0,
        )
        magnet_derating = torch.clamp(1.0 + ALPHA_MAG * (coil_temp - T_AMB), min=0.5, max=1.0)
        thermal_factor = resistance_derating * magnet_derating * severe_derating

        # [Fix #2] Fatigue에 의한 게인 감소 반영 (_apply_physical_degradation과 동일 공식)
        # 이전: effort limit이 fatigue를 무시하여, Kp가 줄었는데 effort limit은 그대로인 모순 발생.
        fatigue_factor = torch.clamp(1.0 - (fatigue * 0.6), min=0.2)

        # 복합 derating: thermal × fatigue
        combined_factor = thermal_factor * fatigue_factor
        if isinstance(joint_idx, slice):
            limits *= combined_factor
        else:
            limits[:, joint_idx] *= combined_factor
        
        # Friction Loss Compensation (Effective Torque Reduction)
        # [Fix #6] base_friction_torque도 joint_idx 슬라이싱
        base_stiction = getattr(self.motor_deg_state, "base_friction_torque", None)
        if base_stiction is not None and isinstance(base_stiction, torch.Tensor):
            base_stiction_s = base_stiction[ids]
            base_stiction_s = base_stiction_s[:, joint_idx] if base_stiction_s.ndim == 2 else base_stiction_s
        else:
            base_stiction_s = STICTION_NOMINAL
        
        # [Fix #7] 정적 마찰(stiction)만 effort limit에서 차감.
        # 이전: 속도 의존 점성 마찰(viscous)도 차감하여 effort limit이 매 substep 변동,
        # 모터 peak torque가 본래 속도 독립적 특성인 물리 원칙에 위배됨.
        # [Fix] fatigue-dependent 마찰 증가 반영 (interface.py 마찰 계산과 동일 공식).
        # fatigue_factor는 모터의 전기적 토크 생성 능력 감소 (곱셈적),
        # stiction의 fatigue 스케일링은 베어링 마모에 의한 기계적 마찰 증가 (가산적).
        stiction_loss = base_stiction_s * (1.0 + fatigue * STICTION_WEAR_FACTOR) * friction_bias
        
        # [Fix] 마찰 보상 계수 0.5 적용: stiction은 이미 interface.py에서
        # (1) 마찰열 → 온도 상승 → thermal derating, (2) 마찰 전력 → SOC 감소 → brownout
        # 두 경로로 성능 저하에 기여하므로, effort limit 차감은 절반만 적용하여
        # 3중 경로에 의한 과도한 열화를 완화합니다.
        friction_total = stiction_loss * 0.5
        if isinstance(joint_idx, slice):
            limits -= friction_total
        else:
            limits[:, joint_idx] -= friction_total
        
        return torch.clamp(limits, min=0.0)

    def _iter_active_reward_terms(self) -> list[tuple[str, Any, float, dict[str, Any]]]:
        """Collect active reward terms from cfg for optional diagnostics."""
        reward_cfg = getattr(getattr(self, "cfg", None), "rewards", None)
        if reward_cfg is None:
            return []

        if hasattr(reward_cfg, "__dataclass_fields__"):
            candidate_names = list(reward_cfg.__dataclass_fields__.keys())
        else:
            candidate_names = [k for k in vars(reward_cfg).keys() if not k.startswith("_")]

        terms: list[tuple[str, Any, float, dict[str, Any]]] = []
        for name in candidate_names:
            term = getattr(reward_cfg, name, None)
            if term is None or not hasattr(term, "func") or not hasattr(term, "weight"):
                continue
            try:
                weight = float(term.weight)
            except Exception:
                continue
            if abs(weight) <= 1e-12:
                continue
            params = getattr(term, "params", None)
            if params is None:
                params = {}
            if not isinstance(params, dict):
                continue
            terms.append((name, term.func, weight, params))
        return terms

    def _as_reward_vector(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if y.ndim == 0:
            y = y.repeat(self.num_envs)
        if y.ndim > 1:
            if y.shape[-1] == 1:
                y = y.squeeze(-1)
            else:
                y = torch.mean(y, dim=tuple(range(1, y.ndim)))
        return y

    def _compute_reward_term_contributions(
        self, env_ids: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], bool | None, float | None]:
        """
        Compute per-term weighted contributions for selected env_ids.

        Used only for terminal snapshot diagnostics when enabled.
        """
        if env_ids.numel() == 0:
            return {}, None, None
        if not hasattr(self, "rew_buf"):
            return {}, None, None

        term_specs = getattr(self, "_reward_term_specs_cache", None)
        if term_specs is None:
            term_specs = self._iter_active_reward_terms()
            self._reward_term_specs_cache = term_specs
        if len(term_specs) == 0:
            return {}, None, None

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        target_reward = self._as_reward_vector(self.rew_buf)[env_ids]
        sum_no_dt = torch.zeros_like(target_reward)
        sum_with_dt = torch.zeros_like(target_reward)
        contrib_no_dt: dict[str, torch.Tensor] = {}

        for term_name, term_func, term_weight, term_params in term_specs:
            raw = term_func(self, **term_params)
            raw = torch.nan_to_num(self._as_reward_vector(raw), nan=0.0, posinf=0.0, neginf=0.0)
            weighted = raw * float(term_weight)
            weighted_sel = weighted[env_ids]
            contrib_no_dt[term_name] = weighted_sel
            sum_no_dt = sum_no_dt + weighted_sel
            sum_with_dt = sum_with_dt + (weighted_sel * float(self.step_dt))

        mae_no_dt = float(torch.mean(torch.abs(sum_no_dt - target_reward)).item())
        mae_with_dt = float(torch.mean(torch.abs(sum_with_dt - target_reward)).item())
        use_dt = bool(mae_with_dt <= mae_no_dt)

        if use_dt:
            contrib = {k: v * float(self.step_dt) for k, v in contrib_no_dt.items()}
            recon = sum_with_dt
        else:
            contrib = contrib_no_dt
            recon = sum_no_dt
        recon_mae = float(torch.mean(torch.abs(recon - target_reward)).item())
        return contrib, use_dt, recon_mae

    def _thermal_termination_params(self) -> tuple[bool, float]:
        """Read thermal termination config for backward-compatible defaults."""
        use_case_proxy = False
        coil_to_case_delta_c = 5.0
        term_cfg = getattr(getattr(self, "cfg", None), "terminations", None)
        thermal_failure = getattr(term_cfg, "thermal_failure", None) if term_cfg is not None else None
        params = getattr(thermal_failure, "params", None)
        if isinstance(params, dict):
            use_case_proxy = bool(params.get("use_case_proxy", use_case_proxy))
            coil_to_case_delta_c = float(params.get("coil_to_case_delta_c", coil_to_case_delta_c))
        return use_case_proxy, coil_to_case_delta_c

    def _temperature_metric_semantics(self) -> str:
        """Resolve metric temperature channel from explicit cfg, with safe fallback."""
        cfg = getattr(self, "cfg", None)
        explicit = str(getattr(cfg, "temperature_metric_semantics", "")).strip().lower()
        if explicit in {"case_proxy", "case", "case_like"}:
            return "case_proxy"
        if explicit in {"coil_hotspot", "coil"}:
            return "coil_hotspot"
        use_case_proxy, _ = self._thermal_termination_params()
        return "case_proxy" if use_case_proxy else "coil_hotspot"

    def _case_temperature_tensor(self, env_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """Find case/housing temperature tensor from MotorDeg state if available."""
        deg_state = self.motor_deg_state
        for name in ("motor_case_temp", "case_temp", "motor_temp_case", "housing_temp", "motor_housing_temp"):
            if hasattr(deg_state, name):
                val = getattr(deg_state, name)
                if isinstance(val, torch.Tensor):
                    return val if env_ids is None else val[env_ids]
        return None

    def _temperature_tensor_for_metrics(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Return task-consistent temperature tensor for logging/evaluation metrics.

        Main task can use case-like proxy semantics even if hard thermal termination is disabled.
        """
        semantics = self._temperature_metric_semantics()
        _, coil_to_case_delta_c = self._thermal_termination_params()

        coil_temp = self.motor_deg_state.coil_temp if env_ids is None else self.motor_deg_state.coil_temp[env_ids]
        if semantics == "case_proxy":
            case_like = self._case_temperature_tensor(env_ids=env_ids)
            if case_like is not None:
                return case_like
            return coil_temp - float(coil_to_case_delta_c)
        return coil_temp

    def _cache_terminal_snapshot(self, env_ids: torch.Tensor):
        """Cache terminal metrics before reset so evaluators can avoid post-reset contamination."""
        if not self._enable_terminal_snapshot or env_ids.numel() == 0:
            return
        if self.motor_deg_state is None:
            return

        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        robot = self.scene["robot"]
        deg_state = self.motor_deg_state

        cmd = self.command_manager.get_command("base_velocity")[env_ids]
        actual_vel = robot.data.root_lin_vel_b[env_ids][:, :2]
        actual_ang = robot.data.root_ang_vel_b[env_ids][:, 2]
        tracking_error_xy = torch.norm(cmd[:, :2] - actual_vel, dim=1)
        tracking_error_ang = torch.abs(cmd[:, 2] - actual_ang)

        total_power = torch.sum(deg_state.avg_power_log[env_ids], dim=1)

        # Use explicit task semantics (cfg.temperature_metric_semantics) when provided.
        temps = self._temperature_tensor_for_metrics(env_ids=env_ids)

        avg_temp = torch.mean(temps, dim=1)
        max_temp = torch.max(temps, dim=1)[0]
        max_fatigue = torch.max(deg_state.fatigue_index[env_ids], dim=1)[0]
        soh = deg_state.motor_health_capacity[env_ids] - deg_state.fatigue_index[env_ids]
        min_soh = torch.min(soh, dim=1)[0]
        max_saturation = torch.max(deg_state.torque_saturation[env_ids], dim=1)[0]
        crit_latched = ((self._crit_latch_steps_remaining[env_ids] > 0) | self._crit_need_unlatch[env_ids]).to(
            torch.float32
        )
        crit_sat_any_step = self._crit_sat_any[env_ids].to(torch.float32)
        crit_sat_ratio = self._crit_sat_latch.ratio[env_ids]
        crit_sat_ratio_valid_steps = self._crit_sat_latch.valid_steps[env_ids].to(torch.float32)
        crit_action_delta = self._crit_action_delta_norm_step[env_ids]
        crit_action_transition_delta = self._crit_action_transition_delta_norm_step[env_ids]
        crit_cmd_delta = self._crit_cmd_delta_norm_step[env_ids]
        crit_governor_mode = self._crit_governor_mode_step[env_ids].to(torch.float32)
        crit_action_delta_latched = crit_action_delta * crit_latched
        crit_cmd_delta_latched = crit_cmd_delta * crit_latched
        crit_post_unlatch_ramp_active = (
            self._crit_post_unlatch_action_ramp_steps_remaining[env_ids] > 0
        ).to(torch.float32)
        crit_post_unlatch_ramp_steps_remaining = self._crit_post_unlatch_action_ramp_steps_remaining[
            env_ids
        ].to(torch.float32)
        crit_governor_enabled = torch.full(
            (env_ids.numel(),),
            1.0 if self._crit_governor_enable else 0.0,
            device=self.device,
            dtype=torch.float32,
        )
        reward_terms, reward_dt_scaled, reward_recon_mae = self._compute_reward_term_contributions(env_ids)

        self._last_terminal_env_ids = env_ids.clone()
        self._last_terminal_metrics = {
            "tracking_error_xy": tracking_error_xy.detach().clone(),
            "tracking_error_ang": tracking_error_ang.detach().clone(),
            "total_power": total_power.detach().clone(),
            "soc": deg_state.soc[env_ids].detach().clone(),
            "avg_temp": avg_temp.detach().clone(),
            "max_temp": max_temp.detach().clone(),
            "max_fatigue": max_fatigue.detach().clone(),
            "min_soh": min_soh.detach().clone(),
            "max_saturation": max_saturation.detach().clone(),
            "crit/governor_enabled": crit_governor_enabled.detach().clone(),
            "crit/is_latched": crit_latched.detach().clone(),
            "crit/sat_any_over_thr_step": crit_sat_any_step.detach().clone(),
            "crit/sat_any_over_thr_ratio": crit_sat_ratio.detach().clone(),
            "crit/sat_ratio_valid_steps": crit_sat_ratio_valid_steps.detach().clone(),
            "crit/action_delta_norm_step": crit_action_delta.detach().clone(),
            "crit/action_transition_delta_norm_step": crit_action_transition_delta.detach().clone(),
            "crit/cmd_delta_norm_step": crit_cmd_delta.detach().clone(),
            "crit/governor_mode_step": crit_governor_mode.detach().clone(),
            "crit/action_delta_latched_norm_step": crit_action_delta_latched.detach().clone(),
            "crit/cmd_delta_latched_norm_step": crit_cmd_delta_latched.detach().clone(),
            "crit/post_unlatch_action_ramp_active_step": crit_post_unlatch_ramp_active.detach().clone(),
            "crit/post_unlatch_action_ramp_steps_remaining_step": (
                crit_post_unlatch_ramp_steps_remaining.detach().clone()
            ),
        }

        if self.enable_eval_gait_metrics:
            steps = torch.clamp(self._eval_ep_steps[env_ids], min=1).to(torch.float32)
            mean_cmd_speed_xy = self._eval_cmd_speed_sum[env_ids] / steps
            mean_actual_speed_xy = self._eval_actual_speed_sum[env_ids] / steps
            nonzero_cmd_ratio = self._eval_nonzero_cmd_steps[env_ids] / steps
            stand_cmd_ratio = self._eval_stand_cmd_steps[env_ids] / steps
            path_length_xy = self._eval_path_length[env_ids]
            progress_distance_along_cmd = self._eval_progress_distance[env_ids]
            self._last_terminal_metrics["motion/mean_cmd_speed_xy"] = mean_cmd_speed_xy.detach().clone()
            self._last_terminal_metrics["motion/mean_actual_speed_xy"] = mean_actual_speed_xy.detach().clone()
            self._last_terminal_metrics["motion/nonzero_cmd_ratio"] = nonzero_cmd_ratio.detach().clone()
            self._last_terminal_metrics["motion/stand_cmd_ratio"] = stand_cmd_ratio.detach().clone()
            self._last_terminal_metrics["motion/path_length_xy"] = path_length_xy.detach().clone()
            self._last_terminal_metrics["motion/progress_distance_along_cmd"] = progress_distance_along_cmd.detach().clone()

            if self._eval_gait_ready and self._eval_gait_touchdown_count.shape[1] > 0:
                valid_steps = torch.clamp(self._eval_gait_valid_steps[env_ids], min=1.0)
                step_count_total = torch.sum(self._eval_gait_touchdown_count[env_ids], dim=1)
                quad_support_ratio = self._eval_gait_quad_support_steps[env_ids] / valid_steps
                slip_distance_total = torch.sum(self._eval_gait_slip_distance[env_ids], dim=1)
                slip_per_progress = slip_distance_total / torch.clamp(progress_distance_along_cmd, min=0.05)
                self._last_terminal_metrics["gait/step_count_total"] = step_count_total.detach().clone()
                self._last_terminal_metrics["gait/quad_support_ratio"] = quad_support_ratio.detach().clone()
                self._last_terminal_metrics["gait/slip_per_progress"] = slip_per_progress.detach().clone()

        for term_name, term_val in reward_terms.items():
            self._last_terminal_metrics[f"reward_term/{term_name}"] = term_val.detach().clone()
        if reward_dt_scaled is not None:
            dt_scaled_flag = 1.0 if reward_dt_scaled else 0.0
            self._last_terminal_metrics["reward_term/dt_scaled"] = torch.full(
                (env_ids.numel(),), dt_scaled_flag, device=self.device, dtype=torch.float32
            )
        if reward_recon_mae is not None:
            self._last_terminal_metrics["reward_term/recon_mae"] = torch.full(
                (env_ids.numel(),), float(reward_recon_mae), device=self.device, dtype=torch.float32
            )

    def _log_motor_deg_metrics(self):
        if self.motor_deg_state is None: return
        temps = self._temperature_tensor_for_metrics()
        log_dict = self.extras.setdefault("log", {})

        self.extras["motor_deg/avg_temp"] = torch.mean(temps)
        if hasattr(self.motor_deg_state, "motor_case_temp"):
            self.extras["motor_deg/avg_case_temp"] = torch.mean(self.motor_deg_state.motor_case_temp)
        self.extras["motor_deg/max_fatigue"] = torch.max(self.motor_deg_state.fatigue_index)
        self.extras["motor_deg/saturation_rate"] = torch.mean(self.motor_deg_state.torque_saturation)
        if hasattr(self.motor_deg_state, "fault_mask"):
            fault_mask = self.motor_deg_state.fault_mask
            self.extras["motor_deg/fault_joint_ratio"] = torch.mean(fault_mask)
            if fault_mask.ndim == 2 and fault_mask.shape[1] >= 12:
                fr = torch.max(fault_mask[:, 0:3], dim=1)[0]
                fl = torch.max(fault_mask[:, 3:6], dim=1)[0]
                rr = torch.max(fault_mask[:, 6:9], dim=1)[0]
                rl = torch.max(fault_mask[:, 9:12], dim=1)[0]
                self.extras["motor_deg/fault_leg_fr_ratio"] = torch.mean(fr)
                self.extras["motor_deg/fault_leg_fl_ratio"] = torch.mean(fl)
                self.extras["motor_deg/fault_leg_rr_ratio"] = torch.mean(rr)
                self.extras["motor_deg/fault_leg_rl_ratio"] = torch.mean(rl)
        if hasattr(self.motor_deg_state, "fault_motor_id"):
            fault_motor_id = self.motor_deg_state.fault_motor_id
            num_motors = int(self.motor_deg_state.fault_mask.shape[1]) if hasattr(self.motor_deg_state, "fault_mask") else 12
            if (
                not hasattr(self, "_motor_deg_fault_step_exposure_counts")
                or int(getattr(self._motor_deg_fault_step_exposure_counts, "numel", lambda: 0)()) != int(num_motors)
            ):
                self._motor_deg_fault_step_exposure_counts = torch.zeros((num_motors,), device=self.device, dtype=torch.float32)
            if not hasattr(self, "_motor_deg_fault_step_exposure_total"):
                self._motor_deg_fault_step_exposure_total = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            valid_fault = fault_motor_id >= 0
            if torch.any(valid_fault):
                self.extras["motor_deg/fault_motor_id_mean"] = torch.mean(fault_motor_id[valid_fault].float())
                # Cumulative step-exposure counter: how many env-steps each motor fault was active.
                binc = torch.bincount(fault_motor_id[valid_fault], minlength=num_motors).to(torch.float32)
                self._motor_deg_fault_step_exposure_counts += binc
                self._motor_deg_fault_step_exposure_total += float(torch.sum(valid_fault).item())
            else:
                self.extras["motor_deg/fault_motor_id_mean"] = torch.tensor(-1.0, device=self.device)

        # Fault sampling diagnostics (reset-time episode exposure + step exposure).
        if hasattr(self, "_motor_deg_fault_episode_counts"):
            ep_counts = self._motor_deg_fault_episode_counts.to(torch.float32)
            ep_total = torch.clamp(torch.sum(ep_counts), min=1.0)
            self.extras["motor_deg/fault_episode_total"] = torch.sum(ep_counts)
            log_dict["motor_deg/fault_episode_total"] = torch.sum(ep_counts)
            for m in range(min(int(ep_counts.numel()), 12)):
                key = f"motor_deg/fault_episode_ratio_m{m:02d}"
                val = ep_counts[m] / ep_total
                self.extras[key] = val
                log_dict[key] = val
            if int(ep_counts.numel()) >= 12:
                mirror_pairs = ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))
                for k, (a, b) in enumerate(mirror_pairs):
                    key = f"motor_deg/fault_episode_ratio_pair_{k}"
                    val = (ep_counts[a] + ep_counts[b]) / ep_total
                    self.extras[key] = val
                    log_dict[key] = val
        if hasattr(self, "_motor_deg_fault_step_exposure_counts"):
            step_counts = self._motor_deg_fault_step_exposure_counts.to(torch.float32)
            step_total = torch.clamp(torch.sum(step_counts), min=1.0)
            self.extras["motor_deg/fault_step_exposure_total"] = torch.sum(step_counts)
            log_dict["motor_deg/fault_step_exposure_total"] = torch.sum(step_counts)
            for m in range(min(int(step_counts.numel()), 12)):
                key = f"motor_deg/fault_step_exposure_ratio_m{m:02d}"
                val = step_counts[m] / step_total
                self.extras[key] = val
                log_dict[key] = val
            if int(step_counts.numel()) >= 12:
                mirror_pairs = ((0, 3), (1, 4), (2, 5), (6, 9), (7, 10), (8, 11))
                for k, (a, b) in enumerate(mirror_pairs):
                    key = f"motor_deg/fault_step_exposure_ratio_pair_{k}"
                    val = (step_counts[a] + step_counts[b]) / step_total
                    self.extras[key] = val
                    log_dict[key] = val
        if hasattr(self, "_motor_deg_fault_focus_draw_count") and hasattr(self, "_motor_deg_fault_focus_draw_total"):
            draw_total = torch.clamp(self._motor_deg_fault_focus_draw_total.to(torch.float32), min=1.0)
            self.extras["motor_deg/fault_focus_draw_count"] = self._motor_deg_fault_focus_draw_count.to(torch.float32)
            self.extras["motor_deg/fault_focus_draw_total"] = self._motor_deg_fault_focus_draw_total.to(torch.float32)
            self.extras["motor_deg/fault_focus_draw_ratio"] = self._motor_deg_fault_focus_draw_count.to(torch.float32) / draw_total
            log_dict["motor_deg/fault_focus_draw_count"] = self._motor_deg_fault_focus_draw_count.to(torch.float32)
            log_dict["motor_deg/fault_focus_draw_total"] = self._motor_deg_fault_focus_draw_total.to(torch.float32)
            log_dict["motor_deg/fault_focus_draw_ratio"] = self._motor_deg_fault_focus_draw_count.to(torch.float32) / draw_total
        if hasattr(self, "_motor_deg_fault_pair_sampling_probs"):
            probs = self._motor_deg_fault_pair_sampling_probs.to(torch.float32)
            for k in range(min(int(probs.numel()), 6)):
                key = f"motor_deg/fault_pair_sampling_prob_pair_{k}"
                self.extras[key] = probs[k]
                log_dict[key] = probs[k]
        if hasattr(self, "_motor_deg_fault_pair_sampling_alpha"):
            self.extras["motor_deg/fault_pair_sampling_alpha"] = self._motor_deg_fault_pair_sampling_alpha.to(torch.float32)
            log_dict["motor_deg/fault_pair_sampling_alpha"] = self._motor_deg_fault_pair_sampling_alpha.to(torch.float32)
        if hasattr(self, "_motor_deg_fault_pair_adaptive_enabled"):
            self.extras["motor_deg/fault_pair_adaptive_enabled"] = self._motor_deg_fault_pair_adaptive_enabled.to(torch.float32)
            log_dict["motor_deg/fault_pair_adaptive_enabled"] = self._motor_deg_fault_pair_adaptive_enabled.to(torch.float32)
        if hasattr(self, "_motor_deg_fault_pair_adaptive_mix"):
            self.extras["motor_deg/fault_pair_adaptive_mix"] = self._motor_deg_fault_pair_adaptive_mix.to(torch.float32)
            log_dict["motor_deg/fault_pair_adaptive_mix"] = self._motor_deg_fault_pair_adaptive_mix.to(torch.float32)
        if hasattr(self, "_motor_deg_fault_pair_adaptive_target_probs"):
            probs = self._motor_deg_fault_pair_adaptive_target_probs.to(torch.float32)
            for k in range(min(int(probs.numel()), 6)):
                key = f"motor_deg/fault_pair_adaptive_target_prob_pair_{k}"
                self.extras[key] = probs[k]
                log_dict[key] = probs[k]
        if hasattr(self, "_motor_deg_fault_pair_adaptive_scores"):
            scores = self._motor_deg_fault_pair_adaptive_scores.to(torch.float32)
            for k in range(min(int(scores.numel()), 6)):
                key = f"motor_deg/fault_pair_adaptive_score_pair_{k}"
                self.extras[key] = scores[k]
                log_dict[key] = scores[k]
        if hasattr(self, "_motor_deg_fault_pair_adaptive_confidence"):
            conf = self._motor_deg_fault_pair_adaptive_confidence.to(torch.float32)
            for k in range(min(int(conf.numel()), 6)):
                key = f"motor_deg/fault_pair_adaptive_conf_pair_{k}"
                self.extras[key] = conf[k]
                log_dict[key] = conf[k]
        if hasattr(self, "_motor_deg_fault_motor_adaptive_enabled"):
            self.extras["motor_deg/fault_motor_adaptive_enabled"] = self._motor_deg_fault_motor_adaptive_enabled.to(torch.float32)
            log_dict["motor_deg/fault_motor_adaptive_enabled"] = self._motor_deg_fault_motor_adaptive_enabled.to(torch.float32)
        if hasattr(self, "_motor_deg_fault_motor_adaptive_topk"):
            self.extras["motor_deg/fault_motor_adaptive_topk"] = self._motor_deg_fault_motor_adaptive_topk.to(torch.float32)
            log_dict["motor_deg/fault_motor_adaptive_topk"] = self._motor_deg_fault_motor_adaptive_topk.to(torch.float32)
        if hasattr(self, "_motor_deg_fault_motor_adaptive_target_probs"):
            probs = self._motor_deg_fault_motor_adaptive_target_probs.to(torch.float32)
            for m in range(min(int(probs.numel()), 12)):
                key = f"motor_deg/fault_motor_adaptive_target_prob_m{m:02d}"
                self.extras[key] = probs[m]
                log_dict[key] = probs[m]
        motor_scores = None
        if hasattr(self, "_motor_deg_fault_adaptive_motor_scores"):
            motor_scores = self._motor_deg_fault_adaptive_motor_scores.to(torch.float32)
        elif hasattr(self, "_motor_deg_fault_pair_adaptive_motor_scores"):
            motor_scores = self._motor_deg_fault_pair_adaptive_motor_scores.to(torch.float32)
        if motor_scores is not None:
            ms = motor_scores
            for m in range(min(int(ms.numel()), 12)):
                key = f"motor_deg/fault_adaptive_motor_score_m{m:02d}"
                self.extras[key] = ms[m]
                log_dict[key] = ms[m]
        if hasattr(self, "_motor_deg_fault_adaptive_motor_confidence"):
            conf = self._motor_deg_fault_adaptive_motor_confidence.to(torch.float32)
            for m in range(min(int(conf.numel()), 12)):
                key = f"motor_deg/fault_adaptive_motor_conf_m{m:02d}"
                self.extras[key] = conf[m]
                log_dict[key] = conf[m]

        if hasattr(self.motor_deg_state, "min_voltage_log"):
             self.extras["motor_deg/min_voltage"] = torch.min(self.motor_deg_state.min_voltage_log)
        if hasattr(self.motor_deg_state, "cell_voltage"):
            self.extras["motor_deg/min_cell_voltage"] = torch.min(self.motor_deg_state.cell_voltage)
        if hasattr(self.motor_deg_state, "encoder_hold_flag"):
            self.extras["dr/encoder_hold_rate"] = torch.mean(self.motor_deg_state.encoder_hold_flag)
        if hasattr(self, "_curriculum_effective_step"):
            self.extras["motor_deg/curriculum_effective_step"] = torch.tensor(
                float(self._curriculum_effective_step), device=self.device
            )
        if hasattr(self, "_curriculum_term_ema"):
            self.extras["motor_deg/curriculum_term_ema"] = torch.tensor(
                float(self._curriculum_term_ema), device=self.device
            )
        if hasattr(self, "_curriculum_track_ema"):
            self.extras["motor_deg/curriculum_track_ema"] = torch.tensor(
                float(self._curriculum_track_ema), device=self.device
            )
        if hasattr(self, "_curriculum_p_fresh"):
            self.extras["motor_deg/curriculum_p_fresh"] = torch.tensor(
                float(self._curriculum_p_fresh), device=self.device
            )
        if hasattr(self, "_curriculum_p_used"):
            self.extras["motor_deg/curriculum_p_used"] = torch.tensor(
                float(self._curriculum_p_used), device=self.device
            )
        if hasattr(self, "_curriculum_p_aged"):
            self.extras["motor_deg/curriculum_p_aged"] = torch.tensor(
                float(self._curriculum_p_aged), device=self.device
            )
        if hasattr(self, "_curriculum_p_crit"):
            self.extras["motor_deg/curriculum_p_crit"] = torch.tensor(
                float(self._curriculum_p_crit), device=self.device
            )
