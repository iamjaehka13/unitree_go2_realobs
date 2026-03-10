# =============================================================================
# unitree_go2_realobs/tasks/manager_based/unitree_go2_realobs/unitree_go2_strategic_env_cfg.py
# Strategic comparator configuration for Unitree Go2 motor-degradation control.
# =============================================================================

from __future__ import annotations
from isaaclab.utils import configclass

# [Isaac Lab Core]
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as TermTerm
import isaaclab.sim as sim_utils

# [Actuators]
from isaaclab.actuators import ImplicitActuatorCfg
from .mdp.actuators import MotorDegRealismActuator

# [Asset Helpers]
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.terrains as terrain_gen
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.noise import GaussianNoiseCfg, UniformNoiseCfg

# [User Custom]
from .unitree_go2_motor_deg_env import UnitreeGo2MotorDegEnv
from .mdp.commands import UniformScalarCommandCfg, UniformScalarCommand
from . import mdp as deg_mdp
from .paper_b_task_contract import validate_paper_b_task_cfg

# -------------------------------------------------------------------------
# A. Scene Configuration
# -------------------------------------------------------------------------
@configclass
class UnitreeGo2StrategicSceneCfg(InteractiveSceneCfg):
    """Configuration for the privileged strategic comparator scene."""

    # 1. 지형 (Terrain)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # Shared Paper B default: stay mostly flat, but inject enough floor irregularity
                # to remove low-clearance reward hacking without turning the task into blind
                # foothold planning.
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.9),
                "mild_random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.1,
                    noise_range=(0.0, 0.01),
                    noise_step=0.005,
                    downsampled_scale=0.2,
                    border_width=0.25,
                ),
            },
        ),
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/"
                "TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # 2. 로봇 (Robot)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd", 
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, 
                solver_position_iteration_count=4, 
                solver_velocity_iteration_count=0
            ),
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.40), 
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
        ),

        actuators={
            "hips": ImplicitActuatorCfg(
                class_type=MotorDegRealismActuator,
                joint_names_expr=[".*_hip_joint"],
                stiffness=60.0,
                damping=5.0,
                armature=0.01,
                effort_limit_sim=23.7,
                velocity_limit_sim=30.0,
            ),
            "thighs": ImplicitActuatorCfg(
                class_type=MotorDegRealismActuator,
                joint_names_expr=[".*_thigh_joint"],
                stiffness=80.0,
                damping=4.0,
                armature=0.01,
                effort_limit_sim=23.7,
                velocity_limit_sim=30.0,
            ),
            "calves": ImplicitActuatorCfg(
                class_type=MotorDegRealismActuator,
                joint_names_expr=[".*_calf_joint"],
                stiffness=80.0,
                damping=4.0,
                armature=0.01,
                effort_limit_sim=23.7,
                velocity_limit_sim=30.0,
            ),
        },
    )

    # 3. 센서

    # [3-1] Global Sensor (기존 유지)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        update_period=0.0, history_length=3, track_air_time=True, track_contact_points=False,
        debug_vis=False, 
        force_threshold=5.0,
    )

    base_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.0,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg( resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/"
                "kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )

# -------------------------------------------------------------------------
# B. Actions & Events
# -------------------------------------------------------------------------
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True,
    )

@configclass
class EventCfg:
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14), 
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0) 
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), 
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)
            }
        }
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)}
    )
    
    physics_material = EventTerm(
        # Per-episode domain randomization.
        # Initial range is intentionally moderate and later widened by DR curriculum.
        func=mdp.randomize_rigid_body_material, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"), 
            "static_friction_range": (0.6, 1.25),
            "dynamic_friction_range": (0.6, 1.25),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64
        }
    )
    
    add_mass = EventTerm(
        # Per-episode mass DR.
        # IsaacLab randomizes from default_mass each call, so this does not accumulate.
        # Start with narrow scale and widen later via DR curriculum.
        func=mdp.randomize_rigid_body_mass, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        }
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )

# -------------------------------------------------------------------------
# C. Commands
# -------------------------------------------------------------------------
@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
    
    risk_factor = UniformScalarCommandCfg(
        class_type=UniformScalarCommand,
        resampling_time_range=(5.0, 5.0),
        minimum=0.0, maximum=1.0, 
    )

# -------------------------------------------------------------------------
# D. Observations
# -------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        enable_corruption: bool = True
        concatenate_terms: bool = True

        # Reference locomotion observation core + MotorDeg channels
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
        # joint_pos/joint_vel already include MotorDeg sensor corruption in deg_mdp.*_motor_deg.
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

        energy_budget = ObsTerm(func=deg_mdp.available_voltage_budget, params={"cutoff_voltage": 24.5}, noise=None)
        thermal_stress = ObsTerm(func=deg_mdp.get_thermal_stress_index, params={"asset_cfg": SceneEntityCfg("robot")}, noise=GaussianNoiseCfg(std=0.015))
        mech_health = ObsTerm(func=deg_mdp.get_mechanical_health_index, params={"asset_cfg": SceneEntityCfg("robot")})
        
        vibration_level = ObsTerm(
            func=deg_mdp.weighted_contact_acceleration,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "imu_cfg": SceneEntityCfg("base_imu")
            }
        )

        # LongTermHealthBuffer 기반 피로도 기울기 관측.
        degradation_trend = ObsTerm(
            func=deg_mdp.degradation_slope,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # LongTermHealthBuffer 기반 누적 과열 시간 관측.
        thermal_overload = ObsTerm(
            func=deg_mdp.thermal_overload_duration_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.01}
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        # Reference critic core + MotorDeg channels. No corruption for value stability.
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
        energy_budget = ObsTerm(func=deg_mdp.available_voltage_budget, params={"cutoff_voltage": 24.5}, noise=None)
        thermal_stress = ObsTerm(func=deg_mdp.get_thermal_stress_index, params={"asset_cfg": SceneEntityCfg("robot")}, noise=None)
        mech_health = ObsTerm(func=deg_mdp.get_mechanical_health_index, params={"asset_cfg": SceneEntityCfg("robot")})
        vibration_level = ObsTerm(
            func=deg_mdp.weighted_contact_acceleration,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "imu_cfg": SceneEntityCfg("base_imu")},
        )
        degradation_trend = ObsTerm(func=deg_mdp.degradation_slope, params={"asset_cfg": SceneEntityCfg("robot")})
        thermal_overload = ObsTerm(
            func=deg_mdp.thermal_overload_duration_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "scale": 0.01},
        )

    critic: CriticCfg = CriticCfg()

# -------------------------------------------------------------------------
# E. Rewards (Best Walking Config Restored)
# -------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """Reference locomotion rewards + MotorDeg rewards."""
    
    # -------------------------------------------------------------------------
    # 1. Task Rewards (Gait & Tracking)
    # -------------------------------------------------------------------------
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

    # Shared locomotion scaffold: enforce terrain-relative swing clearance to remove
    # flat-ground low-clearance reward hacking across all Paper B variants.
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

    # -------------------------------------------------------------------------
    # 2. Stability & Regularization
    # -------------------------------------------------------------------------
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2, 
        weight=-2.0
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2, 
        weight=-0.05
    )

    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )

    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)

    action_rate = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-0.1
    )

    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # -------------------------------------------------------------------------
    # 3. MotorDeg Objectives (Health & Efficiency)
    # -------------------------------------------------------------------------
    energy_efficiency = RewTerm(
        func=deg_mdp.electrical_energy_reward, 
        weight=0.01, 
        # Net-zero exploit 방지: 관절 간 (+/-) 전력 상쇄 보상 경로 차단.
        params={"std": 25.0, "allow_regen_bonus": False}
    )

    bearing_health = RewTerm(
        func=deg_mdp.bearing_life_reward, 
        weight=0.05, 
        params={"threshold": 0.5, "scale_factor": 1.0} 
    )

    # limit_temp=90.0 (TEMP_CRITICAL_THRESHOLD 기준):
    # - threshold = 90.0 * 0.95 = 85.5°C에서 페널티 시작
    # - derating 시작(75°C)과 termination(90°C) 사이에 경고 구간 확보
    thermal_safety = RewTerm(
        func=deg_mdp.thermal_predictive_reward, 
        weight=0.03,
        params={"std": 3.0, "limit_temp": 90.0, "horizon_dt": 0.2}
    )

    saturation_prevention = RewTerm(
        func=deg_mdp.actuator_saturation_reward,
        weight=0.01,
        params={"std": 0.1}
    )

# -------------------------------------------------------------------------
# F. Terminations
# -------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out = TermTerm(func=mdp.time_out, time_out=True)
    
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )

    bad_orientation = TermTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})
    
    # 종료 임계값을 TEMP_CRITICAL_THRESHOLD(90°C)에 맞춘다.
    # derating 구간(75~90°C)을 충분히 경험한 뒤 안전하게 종료되도록 유지한다.
    thermal_failure = TermTerm(func=deg_mdp.thermal_runaway, params={"threshold_temp": 90.0})
    motor_stall = TermTerm(func=deg_mdp.motor_stall, params={"stall_time_threshold_s": 0.5})

# -------------------------------------------------------------------------
# H. Main Environment Config
# -------------------------------------------------------------------------
@configclass
class UnitreeGo2StrategicEnvCfg(ManagerBasedRLEnvCfg):
    class_type = UnitreeGo2MotorDegEnv 
    paper_b_family: str = "upper_bound"
    paper_b_variant: str = "strategic"
    paper_b_observation_scope: str = "privileged"
    paper_b_reward_scope: str = "privileged"
    paper_b_deployable: bool = False
    scene: UnitreeGo2StrategicSceneCfg = UnitreeGo2StrategicSceneCfg(num_envs=4096, env_spacing=2.5)
    # Keep heavy contact debug monitor off by default for scalable training.
    debug_contact_force_monitor: bool = False
    # Optional one-shot startup tracing for env reset/step.
    debug_first_reset: bool = False
    debug_first_step: bool = False
    # Brownout policy source:
    # - "bms_pred": nominal BMS model prediction (default strategic setting)
    # - "true_voltage": measured/ground-truth pack voltage channel
    # - "sensor_voltage": potentially biased sensed voltage
    brownout_voltage_source: str = "bms_pred"
    brownout_enter_v: float = 24.5
    brownout_recover_v: float = 25.0
    temperature_metric_semantics: str = "coil_hotspot"
    # Sensor-bias randomization for battery voltage channel (reset-time DR).
    # Default range is conservative; tune with real logs if available.
    voltage_sensor_bias_range_v: tuple[float, float] = (-0.12, 0.12)
    # Encoder sensor noise priors (realistic defaults; tune from real logs).
    encoder_pos_noise_std_rad: float = 0.005
    encoder_vel_noise_std_rads: float = 0.03
    # Joint-side friction bias DR.
    friction_bias_range: tuple[float, float] = (0.95, 1.05)
    # IMU temperature drift coefficients.
    imu_gyro_drift_sensitivity: float = 0.0008
    imu_accel_drift_sensitivity: float = 0.0012
    # Command transport DR: control-latency / packet-drop proxy.
    cmd_transport_dr_enable: bool = True
    cmd_delay_max_steps: int = 1
    cmd_dropout_prob: float = 0.01
    # Sensor staleness DR: hold previous encoder measurement with this probability.
    encoder_sample_hold_prob: float = 0.02
    # Real-observable discretization DR.
    case_temp_quant_step_c: float = 1.0
    battery_voltage_quant_step_v: float = 0.01
    cell_voltage_quant_step_v: float = 0.005
    # 8-cell DR priors (used when synthesizing per-cell voltage channel).
    cell_ocv_bias_range_v: tuple[float, float] = (-0.015, 0.015)
    cell_ir_range_ohm: tuple[float, float] = (0.0060, 0.0090)
    cell_sensor_bias_range_v: tuple[float, float] = (-0.010, 0.010)
    # Curriculum schedule length in environment steps (used by MotorDeg reset curriculum).
    curriculum_total_steps: int = 120_000
    # MotorDeg degradation curriculum milestones (iteration-based).
    # Requested 3000-iter plan:
    # - 0~1600 : locomotion/DR focus (MotorDeg almost fresh)
    # - 1601~2400: MotorDeg medium-strength ramp
    # - 2401~3000: hardest MotorDeg mix + stabilization
    motor_deg_curriculum_use_performance_gate: bool = False
    motor_deg_curriculum_steps_per_iter: int = 24
    motor_deg_curriculum_used_start_iter: int = 1601
    motor_deg_curriculum_used_end_iter: int = 1900
    motor_deg_curriculum_aged_end_iter: int = 2400
    motor_deg_curriculum_critical_end_iter: int = 2800
    motor_deg_curriculum_final_end_iter: int = 3000
    # Fault injection mode at reset:
    # - "single_motor_random" (recommended default for main experiments)
    # - "single_motor_fixed"  (deterministic eval; use `motor_deg_fault_fixed_motor_id`)
    # - "all_motors_random"   (legacy hardest mode)
    motor_deg_fault_injection_mode: str = "single_motor_random"
    motor_deg_fault_fixed_motor_id: int = -1
    # For single_motor_random: sample mirror pairs uniformly then side 50:50.
    motor_deg_fault_pair_uniform_enable: bool = True
    # For single_motor_random: keep sampled fault motor for this many env steps.
    # This equalizes per-motor training step exposure across resets.
    motor_deg_fault_hold_steps: int = 1000
    # Optional hard-case focus sampling on top of pair-uniform/hold logic.
    # In plain focus mode, `motor_deg_fault_focus_prob` is the probability of replacing
    # a fresh draw with focus motors/pairs. In weighted-pair mode, the same value
    # is reused as the uniform-vs-target mixing alpha.
    motor_deg_fault_focus_prob: float = 0.0
    motor_deg_fault_focus_motor_ids: tuple[int, ...] = ()
    motor_deg_fault_focus_pairs: tuple[tuple[int, int], ...] = ()
    # Optional weighted pair sampler:
    # p_pair = normalize(clamp((1-alpha)*uniform + alpha*target, floor, cap)),
    # where alpha = motor_deg_fault_focus_prob.
    motor_deg_fault_pair_weighted_enable: bool = False
    motor_deg_fault_pair_prob_floor: float = 0.0
    motor_deg_fault_pair_prob_cap: float = 1.0
    # Target pair weights in mirror-pair order:
    # [(0,3), (1,4), (2,5), (6,9), (7,10), (8,11)].
    # Empty tuple => fallback target from motor_deg_fault_focus_pairs.
    motor_deg_fault_pair_target_weights: tuple[float, ...] = ()
    # Optional adaptive pair targeting (difficulty-driven).
    # Uses previous-episode per-motor signals accumulated at reset time:
    # non-timeout termination ratio, sat-ratio, and latch ratio.
    # Final pair target = (1-mix)*manual_target + mix*adaptive_target.
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
    # A `motor_deg_fault_focus_prob` fraction is replaced with recent worst top-k motors.
    motor_deg_fault_motor_adaptive_enable: bool = False
    motor_deg_fault_motor_adaptive_topk: int = 3
    motor_deg_fault_motor_adaptive_min_episode_per_motor: float = 20.0
    # Critical command governor:
    # - base-velocity command write-through is applied after command_manager.compute()
    # - optional post-unlatch action smoothing runs before action_manager.process_action()
    motor_deg_scenario_id_critical: int = 4
    critical_governor_enable: bool = True
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
    # Smooth transition right after unlatch to reduce post-unlatch saturation spikes.
    critical_governor_post_unlatch_action_ramp_s: float = 0.0
    # Optional per-step action slew clamp during post-unlatch ramp (0.0 = disabled).
    critical_governor_post_unlatch_action_delta_max: float = 0.0
    # Velocity-command curriculum:
    # - keep easy standing/near-zero velocity phase in early training
    # - widen command ranges from `start_iter` onward.
    velocity_cmd_curriculum_enable: bool = True
    velocity_cmd_curriculum_start_iter: int = 160
    velocity_cmd_curriculum_ramp_iters: int = 340
    # RSL-RL default in this repo is 24 steps/env per iteration.
    velocity_cmd_curriculum_steps_per_iter: int = 24
    velocity_cmd_target_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    velocity_cmd_target_lin_vel_y: tuple[float, float] = (-0.4, 0.4)
    velocity_cmd_target_ang_vel_z: tuple[float, float] = (-1.0, 1.0)
    # Push-disturbance curriculum:
    # - 0~1000: off/very weak
    # - 1001~1600: ramp to full push range
    push_curriculum_enable: bool = True
    push_curriculum_start_iter: int = 1001
    push_curriculum_ramp_iters: int = 599
    push_curriculum_steps_per_iter: int = 24
    push_curriculum_initial_xy: tuple[float, float] = (0.0, 0.0)
    push_curriculum_target_xy: tuple[float, float] = (-0.5, 0.5)
    # DR curriculum (friction/mass/latency):
    # - 501~1000: widen from initial -> target.
    dr_curriculum_enable: bool = True
    dr_curriculum_start_iter: int = 501
    dr_curriculum_ramp_iters: int = 499
    dr_curriculum_steps_per_iter: int = 24
    dr_curriculum_initial_friction_range: tuple[float, float] = (0.6, 1.25)
    dr_curriculum_target_friction_range: tuple[float, float] = (0.5, 1.3)
    dr_curriculum_initial_mass_scale_range: tuple[float, float] = (0.9, 1.1)
    dr_curriculum_target_mass_scale_range: tuple[float, float] = (0.85, 1.15)
    # 50Hz control: step=20ms. delay 1->2 means 20ms -> 40ms max.
    dr_curriculum_initial_cmd_delay_max_steps: int = 1
    dr_curriculum_target_cmd_delay_max_steps: int = 2
    
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg() 
    
    curriculum = None 

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

        # Sensor updates follow reference cadence.
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        validate_paper_b_task_cfg(self)
