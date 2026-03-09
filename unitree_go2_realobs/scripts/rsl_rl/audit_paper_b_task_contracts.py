from __future__ import annotations

import argparse
import json
import os
import sys


def _prepend_path(path: str) -> None:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTENSION_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SOURCE_ROOT = os.path.join(EXTENSION_ROOT, "source", "unitree_go2_realobs")
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", os.path.expanduser("~/IsaacLab"))
ISAACLAB_TASKS_ROOT = os.path.join(ISAACLAB_ROOT, "source", "isaaclab_tasks")

_prepend_path(SOURCE_ROOT)
_prepend_path(ISAACLAB_TASKS_ROOT)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Audit Paper B task contracts under Isaac runtime.")
parser.add_argument("--output", type=str, default="", help="Optional JSON output path.")
parser.add_argument("--indent", type=int, default=2, help="JSON indentation width.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import unitree_go2_realobs.tasks  # noqa: F401

from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.paper_b_task_contract import (
    summarize_paper_b_task_cfg,
    validate_paper_b_task_cfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_baseline_env_cfg import (
    UnitreeGo2BaselineEnvCfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_governor_variants_env_cfg import (
    UnitreeGo2RealObsHardThermEnvCfg,
    UnitreeGo2StrategicNoGovEnvCfg,
    UnitreeGo2StrategicSoftThermEnvCfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_obsonly_env_cfg import (
    UnitreeGo2ObsOnlyEnvCfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_realobs_env_cfg import (
    UnitreeGo2RealObsEnvCfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_strategic_env_cfg import (
    UnitreeGo2StrategicEnvCfg,
)
from unitree_go2_realobs.tasks.manager_based.unitree_go2_realobs.unitree_go2_tempdose_env_cfg import (
    UnitreeGo2TempDoseEnvCfg,
)


TASK_CFGS = {
    "Unitree-Go2-Baseline-v1": UnitreeGo2BaselineEnvCfg,
    "Unitree-Go2-ObsOnly-v1": UnitreeGo2ObsOnlyEnvCfg,
    "Unitree-Go2-RealObs-v1": UnitreeGo2RealObsEnvCfg,
    "Unitree-Go2-Strategic-v1": UnitreeGo2StrategicEnvCfg,
    "Unitree-Go2-Strategic-noGov-v1": UnitreeGo2StrategicNoGovEnvCfg,
    "Unitree-Go2-Strategic-SoftTherm-v1": UnitreeGo2StrategicSoftThermEnvCfg,
    "Unitree-Go2-RealObs-HardTherm-v1": UnitreeGo2RealObsHardThermEnvCfg,
    "Unitree-Go2-TempDose-v1": UnitreeGo2TempDoseEnvCfg,
}


def main() -> None:
    report: dict[str, dict[str, object]] = {}
    for task_id, cfg_cls in TASK_CFGS.items():
        cfg = cfg_cls()
        report[task_id] = validate_paper_b_task_cfg(cfg)

    text = json.dumps(report, indent=args_cli.indent, sort_keys=True)
    if args_cli.output:
        output_path = os.path.abspath(args_cli.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="ascii") as f:
            f.write(text)
            f.write("\n")
    print(text)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
