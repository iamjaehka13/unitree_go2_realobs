# =============================================================================
# unitree_go2_realobs/unitree_go2_baseline_env_cfg.py
# Baseline configuration for motor-degradation safety locomotion.
#
# Purpose: same degraded physics, no MotorDeg observations, no MotorDeg rewards.
# The agent sees only locomotion observations while the motor degradation stack
# still runs internally.
# =============================================================================

from __future__ import annotations
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils.noise import UniformNoiseCfg

from . import mdp as deg_mdp

# Reuse scene/actions/events/commands from the strategic comparator config.
from .unitree_go2_strategic_env_cfg import (
    UnitreeGo2StrategicSceneCfg,
    ActionsCfg,
    EventCfg,
    CommandsCfg,
    TerminationsCfg as StrategicTerminationsCfg,
)
from .unitree_go2_motor_deg_env import UnitreeGo2MotorDegEnv
from .paper_b_task_contract import validate_paper_b_task_cfg


# -------------------------------------------------------------------------
# Baseline observations (no MotorDeg channels)
# -------------------------------------------------------------------------
@configclass
class BaselineObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        enable_corruption: bool = True
        concatenate_terms: bool = True

        # Reference locomotion policy core
        base_ang_vel = ObsTerm(
            func=deg_mdp.base_ang_vel_motor_deg,
            scale=0.2,
            clip=(-100, 100),
            noise=UniformNoiseCfg(n_min=-0.12, n_max=0.12),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        # deg_mdp.*_motor_deg already injects sim-side sensor corruption.
        # Keep ObsTerm noise off to avoid double-noise injection.
        joint_pos = ObsTerm(
            func=deg_mdp.joint_pos_rel_motor_deg,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=None,
        )
        joint_vel = ObsTerm(
            func=deg_mdp.joint_vel_motor_deg,
            scale=0.05,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=None,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        # Same degraded physics, but no MotorDeg observation channels.
        # energy_budget      -- REMOVED
        # thermal_stress     -- REMOVED
        # mech_health        -- REMOVED
        # vibration_level    -- REMOVED
        # degradation_trend  -- REMOVED
        # thermal_overload   -- REMOVED

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Reference critic core. No corruption.
        enable_corruption: bool = False
        concatenate_terms: bool = True

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=deg_mdp.base_ang_vel_motor_deg, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=deg_mdp.joint_pos_rel_motor_deg, clip=(-100, 100), params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(
            func=deg_mdp.joint_vel_motor_deg,
            scale=0.05,
            clip=(-100, 100),
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100), params={"asset_cfg": SceneEntityCfg("robot")})
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    critic: CriticCfg = CriticCfg()


# -------------------------------------------------------------------------
# Baseline rewards (no MotorDeg objectives)
# -------------------------------------------------------------------------
@configclass
class BaselineRewardsCfg:
    # Task rewards shared with the strategic comparator
    track_lin_vel_xy = RewTerm(
        func=deg_mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": SceneEntityCfg("robot")}
    )

    track_ang_vel_z = RewTerm(
        func=deg_mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.25, "asset_cfg": SceneEntityCfg("robot")}
    )

    foot_clearance = RewTerm(
        func=deg_mdp.foot_clearance,
        weight=0.03,
        params={
            "command_name": "base_velocity",
            "target_height": 0.04,
            "std": 0.02,
            "tanh_mult": 2.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
        }
    )

    feet_slide = RewTerm(
        func=deg_mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

    # Stability & Regularization (identical)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5, params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # *** MotorDeg rewards REMOVED ***
    # energy_efficiency       -- REMOVED
    # bearing_health          -- REMOVED
    # thermal_safety          -- REMOVED
    # saturation_prevention   -- REMOVED


# -------------------------------------------------------------------------
# Baseline Terminations
# -------------------------------------------------------------------------
@configclass
class BaselineTerminationsCfg(StrategicTerminationsCfg):
    """Main-ladder baseline: match the soft-safety termination contract of ObsOnly/RealObs."""

    thermal_failure = None


# -------------------------------------------------------------------------
# Main Baseline Config
# -------------------------------------------------------------------------
@configclass
class UnitreeGo2BaselineEnvCfg(ManagerBasedRLEnvCfg):
    class_type = UnitreeGo2MotorDegEnv
    paper_b_family: str = "main_ladder"
    paper_b_variant: str = "baseline"
    paper_b_observation_scope: str = "locomotion_only"
    paper_b_reward_scope: str = "locomotion_only"
    paper_b_deployable: bool = True
    scene: UnitreeGo2StrategicSceneCfg = UnitreeGo2StrategicSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: BaselineObservationsCfg = BaselineObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: BaselineRewardsCfg = BaselineRewardsCfg()
    terminations: BaselineTerminationsCfg = BaselineTerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    curriculum = None
    curriculum_total_steps: int = 120_000
    motor_deg_curriculum_use_performance_gate: bool = False
    motor_deg_curriculum_steps_per_iter: int = 24
    motor_deg_curriculum_used_start_iter: int = 1601
    motor_deg_curriculum_used_end_iter: int = 1900
    motor_deg_curriculum_aged_end_iter: int = 2400
    motor_deg_curriculum_critical_end_iter: int = 2800
    motor_deg_curriculum_final_end_iter: int = 3000
    temperature_metric_semantics: str = "case_proxy"
    motor_deg_fault_injection_mode: str = "single_motor_random"
    motor_deg_fault_fixed_motor_id: int = -1
    # For single_motor_random: sample mirror pairs uniformly then side 50:50.
    motor_deg_fault_pair_uniform_enable: bool = True
    # Hold sampled fault id for a fixed env-step window to stabilize step exposure.
    motor_deg_fault_hold_steps: int = 1000
    # Optional hard-case focus controls.
    # In plain focus mode, `motor_deg_fault_focus_prob` is the probability of replacing
    # a fresh draw with focus motors/pairs. In weighted-pair mode, the same value
    # becomes the uniform-vs-target mixing alpha.
    motor_deg_fault_focus_prob: float = 0.0
    motor_deg_fault_focus_motor_ids: tuple[int, ...] = ()
    motor_deg_fault_focus_pairs: tuple[tuple[int, int], ...] = ()
    # Optional weighted pair sampler:
    # p_pair = normalize(clamp((1-alpha)*uniform + alpha*target, floor, cap)).
    motor_deg_fault_pair_weighted_enable: bool = False
    motor_deg_fault_pair_prob_floor: float = 0.0
    motor_deg_fault_pair_prob_cap: float = 1.0
    # Target weights follow mirror-pair order:
    # [(0,3), (1,4), (2,5), (6,9), (7,10), (8,11)].
    motor_deg_fault_pair_target_weights: tuple[float, ...] = ()
    # Optional adaptive pair targeting from previous-episode difficulty signals.
    motor_deg_fault_pair_adaptive_enable: bool = False
    motor_deg_fault_pair_adaptive_mix: float = 1.0
    motor_deg_fault_pair_adaptive_beta: float = 4.0
    motor_deg_fault_pair_adaptive_ema: float = 0.9
    motor_deg_fault_pair_adaptive_min_episode_per_pair: float = 20.0
    motor_deg_fault_pair_adaptive_w_fail: float = 0.55
    motor_deg_fault_pair_adaptive_w_sat: float = 0.30
    motor_deg_fault_pair_adaptive_w_latch: float = 0.15
    motor_deg_fault_pair_adaptive_sat_scale: float = 1.0
    # Optional recent worst-motor focus on top of the base sampler.
    motor_deg_fault_motor_adaptive_enable: bool = False
    motor_deg_fault_motor_adaptive_topk: int = 3
    motor_deg_fault_motor_adaptive_min_episode_per_motor: float = 20.0
    # Critical command governor.
    motor_deg_scenario_id_critical: int = 4
    critical_governor_enable: bool = False
    critical_governor_v_cap_norm: float = 0.15
    critical_governor_wz_cap: float = 0.0
    critical_governor_ramp_tau_s: float = 2.0
    critical_governor_p_stand_high: float = 0.15
    critical_governor_stand_trigger_norm: float = 0.20
    critical_governor_latch_hold_steps: int = 100
    critical_governor_unlatch_stable_steps: int = 50
    critical_governor_unlatch_cmd_norm: float = 0.10
    critical_governor_unlatch_require_low_cmd: bool = True
    critical_governor_unlatch_require_sat_recovery: bool = False
    critical_governor_pose_roll_pitch_max_rad: float = 0.25
    critical_governor_pose_height_margin_m: float = 0.05
    critical_governor_sat_thr: float = 0.99
    critical_governor_sat_window_steps: int = 15
    critical_governor_sat_trigger: float = 0.95
    critical_governor_sat_trigger_hi: float = 0.95
    critical_governor_sat_trigger_lo: float = 0.95
    # Smooth the first action steps after unlatch to reduce release spikes.
    critical_governor_post_unlatch_action_ramp_s: float = 0.0
    # Optional per-step action slew clamp during the post-unlatch ramp.
    critical_governor_post_unlatch_action_delta_max: float = 0.0
    voltage_sensor_bias_range_v: tuple[float, float] = (-0.12, 0.12)
    encoder_pos_noise_std_rad: float = 0.005
    encoder_vel_noise_std_rads: float = 0.03
    friction_bias_range: tuple[float, float] = (0.95, 1.05)
    imu_gyro_drift_sensitivity: float = 0.0008
    imu_accel_drift_sensitivity: float = 0.0012
    cmd_transport_dr_enable: bool = True
    cmd_delay_max_steps: int = 1
    cmd_dropout_prob: float = 0.01
    encoder_sample_hold_prob: float = 0.02
    case_temp_quant_step_c: float = 1.0
    battery_voltage_quant_step_v: float = 0.01
    cell_voltage_quant_step_v: float = 0.005
    cell_ocv_bias_range_v: tuple[float, float] = (-0.015, 0.015)
    cell_ir_range_ohm: tuple[float, float] = (0.0060, 0.0090)
    cell_sensor_bias_range_v: tuple[float, float] = (-0.010, 0.010)
    velocity_cmd_curriculum_enable: bool = True
    velocity_cmd_curriculum_start_iter: int = 160
    velocity_cmd_curriculum_ramp_iters: int = 340
    velocity_cmd_curriculum_steps_per_iter: int = 24
    velocity_cmd_target_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    velocity_cmd_target_lin_vel_y: tuple[float, float] = (-0.4, 0.4)
    velocity_cmd_target_ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    push_curriculum_enable: bool = True
    push_curriculum_start_iter: int = 1001
    push_curriculum_ramp_iters: int = 599
    push_curriculum_steps_per_iter: int = 24
    push_curriculum_initial_xy: tuple[float, float] = (0.0, 0.0)
    push_curriculum_target_xy: tuple[float, float] = (-0.5, 0.5)
    dr_curriculum_enable: bool = True
    dr_curriculum_start_iter: int = 501
    dr_curriculum_ramp_iters: int = 499
    dr_curriculum_steps_per_iter: int = 24
    dr_curriculum_initial_friction_range: tuple[float, float] = (0.6, 1.25)
    dr_curriculum_target_friction_range: tuple[float, float] = (0.5, 1.3)
    dr_curriculum_initial_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    dr_curriculum_target_mass_scale_range: tuple[float, float] = (0.85, 1.15)
    dr_curriculum_initial_cmd_delay_max_steps: int = 1
    dr_curriculum_target_cmd_delay_max_steps: int = 2

    viewer: ViewerCfg = ViewerCfg(
        eye=(3.0, 3.0, 3.0), lookat=(0.0, 0.0, 0.0), origin_type="env", env_index=0, asset_name="robot"
    )

    def __post_init__(self):
        super().__post_init__()
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Paper-B baseline shares deployment semantics with RealObs.
        self.brownout_voltage_source = "sensor_voltage"
        self.brownout_enter_v = 24.5
        self.brownout_recover_v = 25.0
        self.commands.risk_factor.minimum = 1.0
        self.commands.risk_factor.maximum = 1.0
        validate_paper_b_task_cfg(self)
