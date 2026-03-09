from isaaclab.utils import configclass

from .rsl_rl_realobs_cfg import UnitreeGo2RealObsPPORunnerCfg


@configclass
class UnitreeGo2ObsOnlyPPORunnerCfg(UnitreeGo2RealObsPPORunnerCfg):
    """PPO config for measurable-only observations with baseline rewards."""

    experiment_name = "unitree_go2_realobs_obsonly"
