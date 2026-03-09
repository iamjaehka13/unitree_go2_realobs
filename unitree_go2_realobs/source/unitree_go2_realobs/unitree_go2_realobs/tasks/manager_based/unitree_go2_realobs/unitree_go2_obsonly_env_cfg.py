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

    observations: RealObsObservationsCfg = RealObsObservationsCfg()
    rewards: BaselineRewardsCfg = BaselineRewardsCfg()
    terminations: RealObsTerminationsCfg = RealObsTerminationsCfg()
