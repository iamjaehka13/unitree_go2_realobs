#!/usr/bin/env bash
set -euo pipefail

CONDA_SH="${CONDA_SH:-/home/iamjaehka13/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"
PYTHON_BIN="${PYTHON_BIN:-/home/iamjaehka13/miniforge3/envs/isaaclab/bin/python}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found: $CONDA_SH" >&2
  exit 1
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"  # baseline | obsonly | realobs | strategic | all
SEED="${SEED:-45}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-8000}"
HOLD_STEPS="${HOLD_STEPS:-1000}"
HEADLESS="${HEADLESS:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OBS_ABLATION="${OBS_ABLATION:-none}"
SENSOR_PRESET="${SENSOR_PRESET:-full}"
CRITICAL_GOVERNOR="${CRITICAL_GOVERNOR:-off}"  # off | on

if [[ "$MODE" != "baseline" && "$MODE" != "obsonly" && "$MODE" != "realobs" && "$MODE" != "strategic" && "$MODE" != "all" ]]; then
  echo "[ERROR] MODE must be one of: baseline | obsonly | realobs | strategic | all" >&2
  exit 1
fi
if [[ "$CRITICAL_GOVERNOR" != "off" && "$CRITICAL_GOVERNOR" != "on" ]]; then
  echo "[ERROR] CRITICAL_GOVERNOR must be one of: off | on" >&2
  exit 1
fi

governor_flag=(--no-critical_governor_enable)
if [[ "$CRITICAL_GOVERNOR" == "on" ]]; then
  governor_flag=(--critical_governor_enable)
fi

run_one() {
  local task="$1"
  local run_prefix="$2"
  local run_name="${run_prefix}_paperb_${OBS_ABLATION}_${SENSOR_PRESET}_${MAX_ITERS}iter_s${SEED}_${STAMP}"

  local cmd=(
    "$PYTHON_BIN" train.py
    --task "$task"
    --agent rsl_rl_cfg_entry_point
    --num_envs "$NUM_ENVS"
    --max_iterations "$MAX_ITERS"
    --seed "$SEED"
    --run_name "$run_name"
    --train_fault_mode single_motor_random
    --train_fault_pair_uniform
    --train_fault_hold_steps "$HOLD_STEPS"
    --train_fault_focus_prob 0.0
    --no-train_fault_pair_weighted_enable
    --no-train_fault_pair_adaptive_enable
    --no-train_fault_motor_adaptive_enable
    --realobs_obs_ablation "$OBS_ABLATION"
    --realobs_sensor_preset "$SENSOR_PRESET"
    "${governor_flag[@]}"
  )

  if [[ "$HEADLESS" == "1" ]]; then
    cmd+=(--headless)
  fi

  echo "[START] $(date '+%F %T') task=${task} run_name=${run_name}"
  "${cmd[@]}"
  echo "[DONE ] $(date '+%F %T') task=${task} run_name=${run_name}"
}

if [[ "$MODE" == "baseline" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-Baseline-v1" "baseline"
fi
if [[ "$MODE" == "obsonly" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-ObsOnly-v1" "obsonly"
fi
if [[ "$MODE" == "realobs" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-RealObs-v1" "realobs"
fi
if [[ "$MODE" == "strategic" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-Strategic-v1" "strategic"
fi

echo "[ALL DONE] $(date '+%F %T') mode=${MODE} seed=${SEED} num_envs=${NUM_ENVS} max_iters=${MAX_ITERS}"
