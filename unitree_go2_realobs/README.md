# Unitree Go2 RealObs for Isaac Lab

## Overview

This extension is currently organized around the Paper B real-observable workflow:

- `Unitree-Go2-Baseline-v1`
- `Unitree-Go2-ObsOnly-v1`
- `Unitree-Go2-RealObs-v1`
- `Unitree-Go2-TempDose-v1`
- `Unitree-Go2-Strategic-v1`

The main paper ladder is:

`Baseline -> RealObs-ObsOnly -> RealObs-Full -> Strategic upper bound`

`RealObs-TempDose` is kept only as a side ablation.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory).

- From the repository root, install the extension in editable mode:

```bash
# use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
python -m pip install -e unitree_go2_realobs/source/unitree_go2_realobs
```

- Verify that the extension is correctly installed:

  - List the available tasks.
    Note: this prints the Paper B task surface registered from this package.

    ```bash
    python unitree_go2_realobs/scripts/list_envs.py
    ```

  - Run training.

    ```bash
    bash unitree_go2_realobs/scripts/rsl_rl/run_paper_b_core_train.sh realobs
    ```

  - Run evaluation.

    ```bash
    bash unitree_go2_realobs/scripts/rsl_rl/run_paper_b_core_eval.sh realobs
    ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Load as Isaac Lab Extension (Optional)

This package does not ship the default Isaac Lab UI scaffold.
If you want Isaac Lab to discover it through the extension manager,
add the repository's `unitree_go2_realobs/source` directory to the extension search paths and refresh.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Legacy Workflow

Legacy tuning/distillation, replay/debug tooling, and old comparison scripts are removed from the primary Paper B workflow.

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/unitree_go2_realobs"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
