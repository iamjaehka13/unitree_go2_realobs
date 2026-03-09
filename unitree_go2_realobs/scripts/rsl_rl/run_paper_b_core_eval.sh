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

MODE="${1:-all}"  # baseline | obsonly | realobs | realobs_hardtherm | strategic | strategic_nogov | strategic_softtherm | tempdose | all
NUM_ENVS="${NUM_ENVS:-64}"
NUM_EPISODES="${NUM_EPISODES:-50}"
HEADLESS="${HEADLESS:-1}"
EVAL_PROTOCOL_MODE="${EVAL_PROTOCOL_MODE:-combined}"
EVAL_CMD_PROFILE="${EVAL_CMD_PROFILE:-from_env}"
EVAL_SAFETY_CMD_PROFILE="${EVAL_SAFETY_CMD_PROFILE:-stand}"
CHECKPOINT_BASENAME="${CHECKPOINT_BASENAME:-model_7999.pt}"
EVAL_STAMP="${EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OBS_ABLATION="${OBS_ABLATION:-none}"
SENSOR_PRESET="${SENSOR_PRESET:-full}"
GOVERNOR_MODE="${GOVERNOR_MODE:-from_env}"  # from_env | off | on | both
EVAL_FORCED_WALK_LIN_X_MIN="${EVAL_FORCED_WALK_LIN_X_MIN:-}"
EVAL_FORCED_WALK_LIN_X_MAX="${EVAL_FORCED_WALK_LIN_X_MAX:-}"
EVAL_FORCED_WALK_ANG_Z_MIN="${EVAL_FORCED_WALK_ANG_Z_MIN:-}"
EVAL_FORCED_WALK_ANG_Z_MAX="${EVAL_FORCED_WALK_ANG_Z_MAX:-}"

if [[ "$MODE" != "baseline" && "$MODE" != "obsonly" && "$MODE" != "realobs" && "$MODE" != "realobs_hardtherm" && "$MODE" != "strategic" && "$MODE" != "strategic_nogov" && "$MODE" != "strategic_softtherm" && "$MODE" != "tempdose" && "$MODE" != "all" ]]; then
  echo "[ERROR] MODE must be one of: baseline | obsonly | realobs | realobs_hardtherm | strategic | strategic_nogov | strategic_softtherm | tempdose | all" >&2
  exit 1
fi
if [[ "$GOVERNOR_MODE" != "from_env" && "$GOVERNOR_MODE" != "off" && "$GOVERNOR_MODE" != "on" && "$GOVERNOR_MODE" != "both" ]]; then
  echo "[ERROR] GOVERNOR_MODE must be one of: from_env | off | on | both" >&2
  exit 1
fi

supports_paper_b_obs_ablation() {
  local task_slug="$1"
  [[ "$task_slug" == "obsonly" || "$task_slug" == "realobs" || "$task_slug" == "realobs_hardtherm" || "$task_slug" == "tempdose" ]]
}

find_run_dir() {
  local log_subdir="$1"
  local run_name="$2"
  local run_prefix="$3"
  local d=""
  if [[ -n "$run_name" ]]; then
    d="$(ls -dt "${LOG_ROOT}/${log_subdir}"/*_"${run_name}" 2>/dev/null | head -n1 || true)"
  else
    d="$(ls -dt "${LOG_ROOT}/${log_subdir}"/*_"${run_prefix}"_paperb_* 2>/dev/null | head -n1 || true)"
  fi
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

run_one_variant() {
  local task="$1"
  local task_slug="$2"
  local log_subdir="$3"
  local run_prefix="$4"
  local run_name_env="$5"
  local governor_state="$6"
  local task_obs_ablation="none"
  local governor_tag="$governor_state"

  if supports_paper_b_obs_ablation "$task_slug"; then
    task_obs_ablation="$OBS_ABLATION"
  fi

  local run_dir
  run_dir="$(find_run_dir "${log_subdir}" "${run_name_env}" "${run_prefix}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[ERROR] Could not find run dir for ${task_slug} (log_subdir=${log_subdir}, prefix=${run_prefix}, run_name=${run_name_env})" >&2
    exit 1
  fi

  local ckpt
  ckpt="$(find_ckpt "${run_dir}")"
  local out_root="${ROOT}/eval_results/paper_b_core_${EVAL_STAMP}/${task_slug}_gov${governor_tag}"
  mkdir -p "${out_root}"

  local governor_flag=()
  if [[ "$governor_state" == "off" ]]; then
    governor_flag=(--no-critical_governor_enable)
  elif [[ "$governor_state" == "on" ]]; then
    governor_flag=(--critical_governor_enable)
  else
    governor_tag="cfg"
    out_root="${ROOT}/eval_results/paper_b_core_${EVAL_STAMP}/${task_slug}_gov${governor_tag}"
    mkdir -p "${out_root}"
  fi

  echo "[INFO] $(date '+%F %T') task=${task} run=$(basename "${run_dir}") ckpt=$(basename "${ckpt}") governor=${governor_state}"
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
      --paper_b_obs_ablation "${task_obs_ablation}"
      --paper_b_sensor_preset "${SENSOR_PRESET}"
      "${governor_flag[@]}"
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

    echo "[START] $(date '+%F %T') ${task_slug} governor=${governor_state} motor=${mid} -> ${logfile}"
    "${cmd[@]}" >"${logfile}" 2>&1
    echo "[DONE ] $(date '+%F %T') ${task_slug} governor=${governor_state} motor=${mid}"
  done
}

run_one() {
  local task="$1"
  local task_slug="$2"
  local log_subdir="$3"
  local run_prefix="$4"
  local run_name_env="$5"

  if [[ "${GOVERNOR_MODE}" == "both" ]]; then
    run_one_variant "${task}" "${task_slug}" "${log_subdir}" "${run_prefix}" "${run_name_env}" "off"
    run_one_variant "${task}" "${task_slug}" "${log_subdir}" "${run_prefix}" "${run_name_env}" "on"
  else
    run_one_variant "${task}" "${task_slug}" "${log_subdir}" "${run_prefix}" "${run_name_env}" "${GOVERNOR_MODE}"
  fi
}

RUN_NAME_BASELINE="${RUN_NAME_BASELINE:-}"
RUN_NAME_OBSONLY="${RUN_NAME_OBSONLY:-}"
RUN_NAME_REALOBS="${RUN_NAME_REALOBS:-}"
RUN_NAME_REALOBS_HARDTHERM="${RUN_NAME_REALOBS_HARDTHERM:-}"
RUN_NAME_STRATEGIC="${RUN_NAME_STRATEGIC:-}"
RUN_NAME_STRATEGIC_NOGOV="${RUN_NAME_STRATEGIC_NOGOV:-}"
RUN_NAME_STRATEGIC_SOFTTHERM="${RUN_NAME_STRATEGIC_SOFTTHERM:-}"
RUN_NAME_TEMPDOSE="${RUN_NAME_TEMPDOSE:-}"
LOG_SUBDIR_TEMPDOSE="${LOG_SUBDIR_TEMPDOSE:-unitree_go2_realobs_tempdose}"

if [[ "$MODE" == "baseline" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-Baseline-v1" "baseline" "unitree_go2_realobs_baseline" "baseline" "${RUN_NAME_BASELINE}"
fi
if [[ "$MODE" == "obsonly" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-ObsOnly-v1" "obsonly" "unitree_go2_realobs_obsonly" "obsonly" "${RUN_NAME_OBSONLY}"
fi
if [[ "$MODE" == "realobs" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-RealObs-v1" "realobs" "unitree_go2_realobs" "realobs" "${RUN_NAME_REALOBS}"
fi
if [[ "$MODE" == "realobs_hardtherm" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-RealObs-HardTherm-v1" "realobs_hardtherm" "unitree_go2_realobs" "realobs_hardtherm" "${RUN_NAME_REALOBS_HARDTHERM}"
fi
if [[ "$MODE" == "strategic" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-Strategic-v1" "strategic" "unitree_go2_realobs_strategic" "strategic" "${RUN_NAME_STRATEGIC}"
fi
if [[ "$MODE" == "strategic_nogov" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-Strategic-noGov-v1" "strategic_nogov" "unitree_go2_realobs_strategic" "strategic_nogov" "${RUN_NAME_STRATEGIC_NOGOV}"
fi
if [[ "$MODE" == "strategic_softtherm" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-Strategic-SoftTherm-v1" "strategic_softtherm" "unitree_go2_realobs_strategic" "strategic_softtherm" "${RUN_NAME_STRATEGIC_SOFTTHERM}"
fi
if [[ "$MODE" == "tempdose" || "$MODE" == "all" ]]; then
  run_one "Unitree-Go2-TempDose-v1" "tempdose" "${LOG_SUBDIR_TEMPDOSE}" "tempdose" "${RUN_NAME_TEMPDOSE}"
fi

echo "[ALL DONE] $(date '+%F %T') mode=${MODE} eval_stamp=${EVAL_STAMP}"
