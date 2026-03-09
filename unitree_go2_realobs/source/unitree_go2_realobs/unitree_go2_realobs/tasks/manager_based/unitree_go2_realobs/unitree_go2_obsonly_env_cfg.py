from __future__ import annotations

from isaaclab.utils import configclass

from .unitree_go2_baseline_env_cfg import BaselineRewardsCfg
from .unitree_go2_realobs_env_cfg import (
    RealObsObservationsCfg,
    RealObsTerminationsCfg,
    UnitreeGo2RealObsEnvCfg,
)


@configclass
class UnitreeGo2ObsOnlyEnvCfg(UnitreeGo2RealObsEnvCfg):
    """Measurable-only observations with baseline locomotion rewards."""

    paper_b_variant: str = "obsonly"
    paper_b_observation_scope: str = "measurable_only"
    paper_b_reward_scope: str = "locomotion_only"
    observations: RealObsObservationsCfg = RealObsObservationsCfg()
    rewards: BaselineRewardsCfg = BaselineRewardsCfg()
    terminations: RealObsTerminationsCfg = RealObsTerminationsCfg()
