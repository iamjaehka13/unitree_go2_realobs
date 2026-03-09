# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""List registered `unitree_go2_realobs` Isaac Lab environments and entry points."""

"""Listing entrypoint.

AppLauncher must be initialized before importing Isaac task modules.
"""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Isaac task imports follow AppLauncher initialization."""

import gymnasium as gym
from prettytable import PrettyTable

import unitree_go2_realobs.tasks  # noqa: F401


def _is_project_env(task_spec: gym.envs.registration.EnvSpec) -> bool:
    """Return True if a Gym spec belongs to this project package."""
    entry_point = str(task_spec.entry_point)
    env_cfg_entry_point = str(task_spec.kwargs.get("env_cfg_entry_point", ""))
    return "unitree_go2_realobs" in entry_point or "unitree_go2_realobs" in env_cfg_entry_point


def main():
    """Print all environments registered in the `unitree_go2_realobs` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in sorted(gym.registry.values(), key=lambda spec: spec.id):
        if args_cli.keyword is not None and args_cli.keyword not in task_spec.id:
            continue
        if not _is_project_env(task_spec):
            continue

        env_cfg_entry_point = task_spec.kwargs.get("env_cfg_entry_point", "-")
        if env_cfg_entry_point is None:
            env_cfg_entry_point = "-"

        # add details to table
        table.add_row([index + 1, task_spec.id, task_spec.entry_point, env_cfg_entry_point])
        # increment count
        index += 1

    if index == 0:
        print("No matching unitree_go2_realobs environments were found.")
    else:
        print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    finally:
        # close the app
        simulation_app.close()
