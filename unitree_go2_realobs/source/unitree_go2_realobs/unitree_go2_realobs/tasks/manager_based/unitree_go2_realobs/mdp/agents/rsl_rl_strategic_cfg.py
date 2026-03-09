# =============================================================================
# unitree_go2_realobs/mdp/agents/rsl_rl_strategic_cfg.py
# PPO runner defaults for the privileged strategic comparator.
# =============================================================================
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2StrategicPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner defaults for the privileged strategic comparator."""
    # Literature-aligned PPO defaults (2109.11978 + legged_gym convention).
    num_steps_per_env = 24
    # 5000 iters: 3000까지 ramp + 3001~5000 hold 안정화.
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go2_realobs_strategic"
    # Staged exploration schedule (iteration index is absolute learning iteration):
    # entropy: 0.01 -> 0.005 -> 0.003
    entropy_schedule_enable = True
    entropy_stage1_end_iter = 1000
    entropy_stage2_end_iter = 2000
    entropy_stage0_value = 0.01
    entropy_stage1_value = 0.005
    entropy_stage2_value = 0.003
    # action std: high exploration early, then stabilize.
    action_std_schedule_enable = True
    action_std_stage1_end_iter = 1000
    action_std_stage2_end_iter = 2000
    action_std_stage0_value = 1.0
    action_std_stage1_value = 0.7
    action_std_stage2_value = 0.5
    # Late-phase smooth adaptation (iter>=2200): cap-only + log-space rate limiter.
    action_std_late_rate_limit_enable = True
    action_std_late_rate_limit_start_iter = 2200
    action_std_late_rate_limit_segment_iters = 20
    action_std_late_max_up_factor = 1.02
    action_std_late_max_down_factor = 1.00
    # IsaacLab 2.3.1+ explicit observation-group mapping.
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    
    # ---------------------------------------------------------------------
    # Observation history for implicit system identification
    # ---------------------------------------------------------------------
    # 에이전트가 과거 5스텝의 관측값을 함께 봅니다.
    # 이를 통해 Friction Bias나 Voltage Sag의 추세를 진단할 수 있습니다.
    num_observation_history = 5
    
    # Policy Network Architecture
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        # 복잡한 MotorDeg 제어/진단 신호를 다루기 위해 기본 네트워크 용량을 확장.
        # 기존 [32, 32]는 Cartpole용이므로 Quadruped에 맞게 확장
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    
    # PPO Algorithm Hyperparameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # Widely used entropy baseline for legged PPO in Isaac Gym-style training.
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
