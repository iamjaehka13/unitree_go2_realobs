# =============================================================================
# Baseline PPO Runner Config (Smaller network for fewer observations)
# =============================================================================
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2BaselinePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """
    Baseline PPO config: same hyperparameters but smaller observation space
    (no MotorDeg channels) so we keep the same network to ensure fair comparison.
    """
    # Literature-aligned PPO defaults (2109.11978 + legged_gym convention).
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 50
    experiment_name = "unitree_go2_realobs_baseline"
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

    # Same history length for fair comparison
    num_observation_history = 5

    # Same architecture (fair comparison — only obs dimension differs)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

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
