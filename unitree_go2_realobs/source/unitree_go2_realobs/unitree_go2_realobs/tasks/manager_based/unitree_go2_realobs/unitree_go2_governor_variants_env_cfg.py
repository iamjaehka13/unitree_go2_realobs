from __future__ import annotations

from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.utils import configclass

from . import mdp as deg_mdp
from .unitree_go2_realobs_env_cfg import RealObsTerminationsCfg, UnitreeGo2RealObsEnvCfg
from .unitree_go2_strategic_env_cfg import TerminationsCfg as StrategicTerminationsCfg, UnitreeGo2StrategicEnvCfg


@configclass
class UnitreeGo2StrategicNoGovEnvCfg(UnitreeGo2StrategicEnvCfg):
    """Privileged upper-bound comparator without critical governor intervention."""

    paper_b_variant: str = "strategic_nogov"
    critical_governor_enable: bool = False


@configclass
class UnitreeGo2StrategicSoftThermEnvCfg(UnitreeGo2StrategicEnvCfg):
    """Privileged upper-bound comparator with hard thermal stop disabled."""

    paper_b_variant: str = "strategic_softtherm"

    @configclass
    class TerminationsCfg(StrategicTerminationsCfg):
        thermal_failure = None

    terminations: TerminationsCfg = TerminationsCfg()


@configclass
class UnitreeGo2RealObsHardThermEnvCfg(UnitreeGo2RealObsEnvCfg):
    """RealObs comparator with hard thermal stop enabled for termination-confound analysis."""

    paper_b_family: str = "side_ablation"
    paper_b_variant: str = "realobs_hardtherm"
    # Align reported temperature metrics with the coil-based hard-stop semantics.
    temperature_metric_semantics: str = "coil_hotspot"

    @configclass
    class TerminationsCfg(RealObsTerminationsCfg):
        thermal_failure = TermTerm(
            func=deg_mdp.thermal_runaway,
            params={"threshold_temp": 90.0, "use_case_proxy": False},
        )

    terminations: TerminationsCfg = TerminationsCfg()
