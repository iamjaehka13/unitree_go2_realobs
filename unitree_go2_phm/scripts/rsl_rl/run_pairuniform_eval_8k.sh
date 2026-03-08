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
ROOT="${SCRIPT_DIR}"
LOG_ROOT="${ROOT}/logs/rsl_rl"
EVAL_PY="${ROOT}/evaluate.py"

MODE="${1:-realobs}"  # realobs | baseline | both
NUM_ENVS="${NUM_ENVS:-64}"
NUM_EPISODES="${NUM_EPISODES:-50}"
HEADLESS="${HEADLESS:-1}"
EVAL_PROTOCOL_MODE="${EVAL_PROTOCOL_MODE:-combined}"
EVAL_CMD_PROFILE="${EVAL_CMD_PROFILE:-from_env}"
EVAL_SAFETY_CMD_PROFILE="${EVAL_SAFETY_CMD_PROFILE:-stand}"
CHECKPOINT_BASENAME="${CHECKPOINT_BASENAME:-model_7999.pt}"
EVAL_STAMP="${EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EVAL_FORCED_WALK_LIN_X_MIN="${EVAL_FORCED_WALK_LIN_X_MIN:-}"
EVAL_FORCED_WALK_LIN_X_MAX="${EVAL_FORCED_WALK_LIN_X_MAX:-}"
EVAL_FORCED_WALK_ANG_Z_MIN="${EVAL_FORCED_WALK_ANG_Z_MIN:-}"
EVAL_FORCED_WALK_ANG_Z_MAX="${EVAL_FORCED_WALK_ANG_Z_MAX:-}"

RUN_NAME_REALOBS="${RUN_NAME_REALOBS:-realobs_pairuniform_fromscratch_8000iter_s45_20260307_123617}"
RUN_NAME_BASELINE="${RUN_NAME_BASELINE:-baseline_pairuniform_fromscratch_8000iter_s45_20260307_123617}"

if [[ "$MODE" != "realobs" && "$MODE" != "baseline" && "$MODE" != "both" ]]; then
  echo "[ERROR] MODE must be one of: realobs | baseline | both" >&2
  exit 1
fi

find_run_dir() {
  local log_subdir="$1"
  local run_name="$2"
  local d
  d="$(ls -dt "${LOG_ROOT}/${log_subdir}"/*_"${run_name}" 2>/dev/null | head -n1 || true)"
  echo "${d}"
}

find_ckpt() {
  local run_dir="$1"
  if [[ -f "${run_dir}/${CHECKPOINT_BASENAME}" ]]; then
    echo "${run_dir}/${CHECKPOINT_BASENAME}"
  else
    ls "${run_dir}"/model_*.pt | sort -V | tail -n1
  fi
}

run_one() {
  local task="$1"
  local task_slug="$2"
  local log_subdir="$3"
  local run_name="$4"

  local run_dir
  run_dir="$(find_run_dir "${log_subdir}" "${run_name}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[ERROR] Could not find run dir for ${task_slug} run_name=${run_name}" >&2
    exit 1
  fi

  local ckpt
  ckpt="$(find_ckpt "${run_dir}")"
  local out_root="${ROOT}/eval_results/pairuniform_fromscratch_8k_${EVAL_STAMP}/${task_slug}"
  mkdir -p "${out_root}"

  echo "[INFO] $(date '+%F %T') task=${task} run=$(basename "${run_dir}") ckpt=$(basename "${ckpt}")"
  echo "[INFO] output_root=${out_root}"

  for mid in $(seq 0 11); do
    local outdir="${out_root}/m${mid}"
    local logfile="${outdir}.log"
    mkdir -p "${outdir}"

    local cmd=(
      "${PYTHON_BIN}" "${EVAL_PY}"
      --task "${task}"
      --checkpoint "${ckpt}"
      --num_envs "${NUM_ENVS}"
      --num_episodes "${NUM_EPISODES}"
      --output_dir "${outdir}"
      --eval_protocol_mode "${EVAL_PROTOCOL_MODE}"
      --eval_cmd_profile "${EVAL_CMD_PROFILE}"
      --eval_safety_cmd_profile "${EVAL_SAFETY_CMD_PROFILE}"
      --eval_fault_mode single_motor_fixed
      --eval_fault_motor_id "${mid}"
    )

    if [[ -n "${EVAL_FORCED_WALK_LIN_X_MIN}" ]]; then
      cmd+=(--eval_forced_walk_lin_x_min "${EVAL_FORCED_WALK_LIN_X_MIN}")
    fi
    if [[ -n "${EVAL_FORCED_WALK_LIN_X_MAX}" ]]; then
      cmd+=(--eval_forced_walk_lin_x_max "${EVAL_FORCED_WALK_LIN_X_MAX}")
    fi
    if [[ -n "${EVAL_FORCED_WALK_ANG_Z_MIN}" ]]; then
      cmd+=(--eval_forced_walk_ang_z_min "${EVAL_FORCED_WALK_ANG_Z_MIN}")
    fi
    if [[ -n "${EVAL_FORCED_WALK_ANG_Z_MAX}" ]]; then
      cmd+=(--eval_forced_walk_ang_z_max "${EVAL_FORCED_WALK_ANG_Z_MAX}")
    fi

    if [[ "${HEADLESS}" == "1" ]]; then
      cmd+=(--headless)
    fi

    echo "[START] $(date '+%F %T') ${task_slug} motor=${mid} -> ${logfile}"
    "${cmd[@]}" >"${logfile}" 2>&1
    echo "[DONE ] $(date '+%F %T') ${task_slug} motor=${mid}"
  done

  echo "[TASK DONE] $(date '+%F %T') ${task_slug}"
}

if [[ "$MODE" == "realobs" || "$MODE" == "both" ]]; then
  run_one "Unitree-Go2-RealObs-v1" "realobs" "unitree_go2_realobs" "${RUN_NAME_REALOBS}"
fi

if [[ "$MODE" == "baseline" || "$MODE" == "both" ]]; then
  run_one "Unitree-Go2-Baseline-v1" "baseline" "unitree_go2_baseline" "${RUN_NAME_BASELINE}"
fi

echo "[ALL DONE] $(date '+%F %T') mode=${MODE} eval_stamp=${EVAL_STAMP}"
