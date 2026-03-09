# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


ENTRY_POINT = "unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_motor_deg_env:UnitreeGo2MotorDegEnv"


def _register_env(env_id: str, env_cfg_entry_point: str, runner_cfg_entry_point: str) -> None:
    gym.register(
        id=env_id,
        entry_point=ENTRY_POINT,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": runner_cfg_entry_point,
        },
    )

_register_env(
    "Unitree-Go2-Baseline-v1",
    f"{__name__}.unitree_go2_baseline_env_cfg:UnitreeGo2BaselineEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_baseline_cfg:UnitreeGo2BaselinePPORunnerCfg",
)
_register_env(
    "Unitree-Go2-ObsOnly-v1",
    f"{__name__}.unitree_go2_obsonly_env_cfg:UnitreeGo2ObsOnlyEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_obsonly_cfg:UnitreeGo2ObsOnlyPPORunnerCfg",
)
_register_env(
    "Unitree-Go2-RealObs-v1",
    f"{__name__}.unitree_go2_realobs_env_cfg:UnitreeGo2RealObsEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_realobs_cfg:UnitreeGo2RealObsPPORunnerCfg",
)
_register_env(
    "Unitree-Go2-Strategic-v1",
    f"{__name__}.unitree_go2_strategic_env_cfg:UnitreeGo2StrategicEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_strategic_cfg:UnitreeGo2StrategicPPORunnerCfg",
)
_register_env(
    "Unitree-Go2-Strategic-noGov-v1",
    f"{__name__}.unitree_go2_governor_variants_env_cfg:UnitreeGo2StrategicNoGovEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_strategic_cfg:UnitreeGo2StrategicPPORunnerCfg",
)
_register_env(
    "Unitree-Go2-Strategic-SoftTherm-v1",
    f"{__name__}.unitree_go2_governor_variants_env_cfg:UnitreeGo2StrategicSoftThermEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_strategic_cfg:UnitreeGo2StrategicPPORunnerCfg",
)
_register_env(
    "Unitree-Go2-RealObs-HardTherm-v1",
    f"{__name__}.unitree_go2_governor_variants_env_cfg:UnitreeGo2RealObsHardThermEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_realobs_cfg:UnitreeGo2RealObsPPORunnerCfg",
)
_register_env(
    "Unitree-Go2-TempDose-v1",
    f"{__name__}.unitree_go2_tempdose_env_cfg:UnitreeGo2TempDoseEnvCfg",
    f"{__name__}.mdp.agents.rsl_rl_realobs_cfg:UnitreeGo2RealObsPPORunnerCfg",
)
