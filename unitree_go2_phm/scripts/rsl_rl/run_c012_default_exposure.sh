#!/usr/bin/env bash
set -euo pipefail

# C0/C1/C2 default exposure experiment runner
# ------------------------------------------------------------
# C0: uniform only
# C1: bounded adaptive (recommended default candidate)
# C2: aggressive adaptive (stress candidate)
#
# Usage:
#   bash unitree_go2_phm/scripts/rsl_rl/run_c012_default_exposure.sh train
#   bash unitree_go2_phm/scripts/rsl_rl/run_c012_default_exposure.sh eval <STAMP>
#   bash unitree_go2_phm/scripts/rsl_rl/run_c012_default_exposure.sh all
#
# Notes:
# - For eval stage with explicit <STAMP>, script uses eval root:
#   unitree_go2_phm/scripts/rsl_rl/eval_results/c012_defaultexp_<STAMP>
# - For eval stage without stamp, script auto-generates a fresh timestamp.

MODE="${1:-train}"  # train | eval | all
STAMP_IN="${2:-}"

ROOT="/home/iamjaehka13/unitree_go2_phm/unitree_go2_phm/scripts/rsl_rl"
PY="/home/iamjaehka13/miniforge3/envs/isaaclab/bin/python"
TRAIN_PY="${ROOT}/train.py"
EVAL_PY="${ROOT}/evaluate.py"
COMPARE_PY="${ROOT}/compare_default_exposure_c012.py"

LOG_ROOT="${ROOT}/logs/rsl_rl/unitree_go2_realobs"
EVAL_ROOT_BASE="${ROOT}/eval_results"

TASK="Unitree-Go2-RealObs-v1"
SEED=45
TRAIN_NUM_ENVS=32
# IMPORTANT:
# train.py interprets --max_iterations as additional learning iterations.
# We resume from model_5000.pt and target ~model_5999/6000, so this must be 1000.
TRAIN_MAX_ITERS=1000

# Common start point
LOAD_RUN="2026-02-25_00-44-11_realobs_fromscratch_8k_s45"
LOAD_CKPT="model_5000.pt"

# Evaluation defaults
EVAL_NUM_ENVS=256
EVAL_NUM_EPISODES=50
EVAL_PROTOCOL_MODE="combined"
EVAL_CMD_PROFILE="forced_walk"

# Curriculum compression (fixed for this experiment family)
CURR_ARGS=(
  "env.phm_curriculum_used_start_iter=0"
  "env.phm_curriculum_used_end_iter=80"
  "env.phm_curriculum_aged_end_iter=160"
  "env.phm_curriculum_critical_end_iter=240"
  "env.phm_curriculum_final_end_iter=300"
)

if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "all" ]]; then
  echo "[ERR] MODE must be one of: train | eval | all"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${STAMP_IN}" ]]; then
  STAMP="${STAMP_IN}"
fi

RUN_C0="c0_uniform_defaultexp_from5k_to6k_s${SEED}_${STAMP}"
RUN_C1="c1_bounded_adaptive_defaultexp_from5k_to6k_s${SEED}_${STAMP}"
RUN_C2="c2_aggressive_adaptive_defaultexp_from5k_to6k_s${SEED}_${STAMP}"

run_train_variant() {
  local run_name="$1"
  shift
  echo "[TRAIN] ${run_name}"
  "${PY}" "${TRAIN_PY}" \
    --task "${TASK}" \
    --headless \
    --seed "${SEED}" \
    --num_envs "${TRAIN_NUM_ENVS}" \
    --resume \
    --load_run "${LOAD_RUN}" \
    --checkpoint "${LOAD_CKPT}" \
    --max_iterations "${TRAIN_MAX_ITERS}" \
    --run_name "${run_name}" \
    --train_fault_mode single_motor_random \
    --train_fault_pair_uniform \
    --train_fault_hold_steps 1000 \
    "$@" \
    "${CURR_ARGS[@]}"
}

find_latest_run_dir() {
  local suffix="$1"
  local d
  d="$(ls -dt "${LOG_ROOT}"/*_"${suffix}" 2>/dev/null | head -n1 || true)"
  echo "${d}"
}

find_ckpt() {
  local run_dir="$1"
  local ckpt=""
  if [[ -f "${run_dir}/model_5999.pt" ]]; then
    ckpt="${run_dir}/model_5999.pt"
  elif [[ -f "${run_dir}/model_6000.pt" ]]; then
    ckpt="${run_dir}/model_6000.pt"
  else
    ckpt="$(ls "${run_dir}"/model_*.pt | sort -V | tail -n1)"
  fi
  echo "${ckpt}"
}

run_eval_sweep() {
  local variant="$1"      # C0/C1/C2
  local run_suffix="$2"   # run name suffix
  local group_root="$3"
  local run_dir ckpt out_variant

  run_dir="$(find_latest_run_dir "${run_suffix}")"
  if [[ -z "${run_dir}" ]]; then
    echo "[WARN] run not found for suffix=${run_suffix}, skip eval"
    return 0
  fi
  ckpt="$(find_ckpt "${run_dir}")"
  out_variant="${group_root}/${variant}"
  mkdir -p "${out_variant}"

  echo "[EVAL] ${variant} run=$(basename "${run_dir}") ckpt=$(basename "${ckpt}")"
  for mid in $(seq 0 11); do
    local outdir="${out_variant}/m${mid}"
    mkdir -p "${outdir}"
    "${PY}" "${EVAL_PY}" \
      --task "${TASK}" \
      --checkpoint "${ckpt}" \
      --num_envs "${EVAL_NUM_ENVS}" \
      --num_episodes "${EVAL_NUM_EPISODES}" \
      --headless \
      --output_dir "${outdir}" \
      --eval_protocol_mode "${EVAL_PROTOCOL_MODE}" \
      --eval_cmd_profile "${EVAL_CMD_PROFILE}" \
      --eval_fault_mode single_motor_fixed \
      --eval_fault_motor_id "${mid}" \
      >"${outdir}.log" 2>&1
  done
}

if [[ "${MODE}" == "train" || "${MODE}" == "all" ]]; then
  # C0: uniform-only (control)
  run_train_variant "${RUN_C0}" \
    --no-train_fault_pair_weighted_enable \
    --no-train_fault_pair_adaptive_enable \
    --train_fault_focus_prob 0.0

  # C1: bounded adaptive (recommended candidate)
  run_train_variant "${RUN_C1}" \
    --train_fault_pair_weighted_enable \
    --train_fault_pair_prob_floor 0.08 \
    --train_fault_pair_prob_cap 0.24 \
    --train_fault_pair_adaptive_enable \
    --train_fault_pair_adaptive_mix 1.0 \
    --train_fault_pair_adaptive_beta 3.0 \
    --train_fault_pair_adaptive_ema 0.95 \
    --train_fault_pair_adaptive_min_episode_per_pair 20 \
    --train_fault_pair_adaptive_w_fail 0.45 \
    --train_fault_pair_adaptive_w_sat 0.30 \
    --train_fault_pair_adaptive_w_latch 0.20 \
    --train_fault_pair_adaptive_sat_scale 1.0 \
    --train_fault_focus_ramp_start_iter 5300 \
    --train_fault_focus_ramp_end_iter 5900 \
    --train_fault_focus_ramp_start_prob 0.0 \
    --train_fault_focus_ramp_end_prob 0.35 \
    --train_fault_focus_ramp_segment_iters 50

  # C2: aggressive adaptive (stress candidate)
  run_train_variant "${RUN_C2}" \
    --train_fault_pair_weighted_enable \
    --train_fault_pair_prob_floor 0.08 \
    --train_fault_pair_prob_cap 0.24 \
    --train_fault_pair_adaptive_enable \
    --train_fault_pair_adaptive_mix 1.0 \
    --train_fault_pair_adaptive_beta 4.0 \
    --train_fault_pair_adaptive_ema 0.95 \
    --train_fault_pair_adaptive_min_episode_per_pair 20 \
    --train_fault_pair_adaptive_w_fail 0.45 \
    --train_fault_pair_adaptive_w_sat 0.30 \
    --train_fault_pair_adaptive_w_latch 0.20 \
    --train_fault_pair_adaptive_sat_scale 1.0 \
    --train_fault_focus_ramp_start_iter 5300 \
    --train_fault_focus_ramp_end_iter 5900 \
    --train_fault_focus_ramp_start_prob 0.0 \
    --train_fault_focus_ramp_end_prob 0.50 \
    --train_fault_focus_ramp_segment_iters 50
fi

if [[ "${MODE}" == "eval" || "${MODE}" == "all" ]]; then
  GROUP_ROOT="${EVAL_ROOT_BASE}/c012_defaultexp_${STAMP}"
  mkdir -p "${GROUP_ROOT}"

  run_eval_sweep "C0" "${RUN_C0}" "${GROUP_ROOT}"
  run_eval_sweep "C1" "${RUN_C1}" "${GROUP_ROOT}"
  run_eval_sweep "C2" "${RUN_C2}" "${GROUP_ROOT}"

  "${PY}" "${COMPARE_PY}" \
    --variant "C0=c012_defaultexp_${STAMP}/C0/m*" \
    --variant "C1=c012_defaultexp_${STAMP}/C1/m*" \
    --variant "C2=c012_defaultexp_${STAMP}/C2/m*" \
    --tb_log_root "${LOG_ROOT}" \
    --tb_run "C0=*_${RUN_C0}" \
    --tb_run "C1=*_${RUN_C1}" \
    --tb_run "C2=*_${RUN_C2}" \
    --pair_prob_floor 0.08 \
    --pair_prob_cap 0.24 \
    --control C0 \
    --used_aged_tolerance 0.02 \
    --out_json "${GROUP_ROOT}/c012_default_policy_compare.json" \
    --out_csv "${GROUP_ROOT}/c012_default_policy_compare.csv"

  echo "[DONE] compare outputs:"
  echo "  - ${GROUP_ROOT}/c012_default_policy_compare.json"
  echo "  - ${GROUP_ROOT}/c012_default_policy_compare.csv"
fi

echo "[DONE] MODE=${MODE} STAMP=${STAMP}"
echo "[INFO] Run names:"
echo "  C0: ${RUN_C0}"
echo "  C1: ${RUN_C1}"
echo "  C2: ${RUN_C2}"
