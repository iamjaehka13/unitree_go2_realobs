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

MODE="${1:-realobs}"  # realobs | baseline | both
SEED="${SEED:-45}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-5000}"
HOLD_STEPS="${HOLD_STEPS:-1000}"
HEADLESS="${HEADLESS:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ "$MODE" != "realobs" && "$MODE" != "baseline" && "$MODE" != "both" ]]; then
  echo "[ERROR] MODE must be one of: realobs | baseline | both" >&2
  exit 1
fi

run_one() {
  local task="$1"
  local run_prefix="$2"
  local run_name="${run_prefix}_pairuniform_fromscratch_${MAX_ITERS}iter_s${SEED}_${STAMP}"

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
  )

  if [[ "$HEADLESS" == "1" ]]; then
    cmd+=(--headless)
  fi

  echo "[START] $(date '+%F %T') task=${task} run_name=${run_name}"
  "${cmd[@]}"
  echo "[DONE ] $(date '+%F %T') task=${task} run_name=${run_name}"
}

if [[ "$MODE" == "realobs" || "$MODE" == "both" ]]; then
  run_one "Unitree-Go2-RealObs-v1" "realobs"
fi

if [[ "$MODE" == "baseline" || "$MODE" == "both" ]]; then
  run_one "Unitree-Go2-Baseline-v1" "baseline"
fi

echo "[ALL DONE] $(date '+%F %T') mode=${MODE} seed=${SEED} num_envs=${NUM_ENVS} max_iters=${MAX_ITERS}"
