#!/usr/bin/env bash
set -euo pipefail

# Runtime env
CONDA_SH="${CONDA_SH:-/home/iamjaehka13/miniforge3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-isaaclab}"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[ERROR] conda init script not found: $CONDA_SH" >&2
  exit 1
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Defaults (can be overridden by env vars)
TASK="${TASK:-Unitree-Go2-RealObs-v1}"
RUN_DIR="${RUN_DIR:-logs/rsl_rl/unitree_go2_realobs/2026-02-25_00-44-11_realobs_fromscratch_8k_s45}"
SCENARIO="${SCENARIO:-used}"
VX="${VX:-0.30}"
VY="${VY:-0.00}"
WZ="${WZ:-0.00}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"   # 20 sec at dt=0.02
NUM_ENVS="${NUM_ENVS:-1}"
SEED="${SEED:-45}"
PARALLEL_JOBS="${PARALLEL_JOBS:-3}"
FOLLOW_CAMERA="${FOLLOW_CAMERA:-1}"
FOLLOW_CAM_OFFSET_X="${FOLLOW_CAM_OFFSET_X:--0.8}"
FOLLOW_CAM_OFFSET_Y="${FOLLOW_CAM_OFFSET_Y:--3.6}"
FOLLOW_CAM_OFFSET_Z="${FOLLOW_CAM_OFFSET_Z:-1.3}"
FOLLOW_CAM_LOOKAT_Z="${FOLLOW_CAM_LOOKAT_Z:-0.35}"
FOLLOW_CAM_USE_YAW_ONLY="${FOLLOW_CAM_USE_YAW_ONLY:-1}"
FOLLOW_CAM_SMOOTH_ALPHA="${FOLLOW_CAM_SMOOTH_ALPHA:-0.18}"

# Checkpoints to render (space/comma separated via env). Default: 5000-only.
CKPTS_RAW="${CKPTS_RAW:-5000}"
CKPTS_RAW="${CKPTS_RAW//,/ }"
read -r -a CKPTS <<< "$CKPTS_RAW"
if [[ "${#CKPTS[@]}" -eq 0 ]]; then
  CKPTS=(5000)
fi
MOTORS=(0 1 2 3 4 5 6 7 8 9 10 11)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$SCRIPT_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
CKPT_TAG="$(printf "%s_" "${CKPTS[@]}")"
CKPT_TAG="${CKPT_TAG%_}"
BATCH_ID="${BATCH_ID:-used_m0_11_vx${VX}_ck${CKPT_TAG}_${TS}}"
OUT_ROOT="$ROOT_DIR/unitree_go2_phm/scripts/rsl_rl/eval_results/video_batches/$BATCH_ID"
mkdir -p "$OUT_ROOT"
VIDEO_FLAT_DIR="${VIDEO_FLAT_DIR:-$OUT_ROOT/videos_all}"
mkdir -p "$VIDEO_FLAT_DIR"

MANIFEST="$OUT_ROOT/manifest.csv"
LOG_ROOT="$OUT_ROOT/logs"
mkdir -p "$LOG_ROOT"
echo "task,run_dir,scenario,motor,ckpt,checkpoint_path,video_length,cmd_vx,cmd_vy,cmd_wz,video_flat_dir,status,elapsed_s,log_file,video_file" > "$MANIFEST"

echo "[BATCH] $BATCH_ID"
echo "[OUT ] $OUT_ROOT"
echo "[PAR ] parallel_jobs=$PARALLEL_JOBS"
echo "[VID ] flat_video_dir=$VIDEO_FLAT_DIR"

_fmt_tag_float() {
  local x="$1"
  local s="${x/-/m}"
  s="${s/./p}"
  if [[ "$s" != m* ]]; then
    s="p${s}"
  fi
  echo "$s"
}

VX_TAG="$(_fmt_tag_float "$VX")"
VY_TAG="$(_fmt_tag_float "$VY")"
WZ_TAG="$(_fmt_tag_float "$WZ")"

run_one() {
  local ck="$1"
  local m="$2"
  local ckpt_path="$RUN_DIR/model_${ck}.pt"
  local log_file="$LOG_ROOT/play_used_m${m}_ck${ck}.log"
  local start_ts
  local end_ts
  local elapsed
  local status
  local video_tag="${SCENARIO}_m${m}_vx${VX_TAG}_vy${VY_TAG}_wz${WZ_TAG}_model_${ck}"
  local video_file=""
  if [[ -n "$VIDEO_FLAT_DIR" ]]; then
    video_file="$VIDEO_FLAT_DIR/play_${video_tag}-step-0.mp4"
  else
    video_file="$RUN_DIR/videos/play/${video_tag}/play_${video_tag}-step-0.mp4"
  fi

  if [[ ! -f "$ckpt_path" ]]; then
    status="missing_ckpt"
    elapsed=0
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$TASK" "$RUN_DIR" "$SCENARIO" "$m" "$ck" "$ckpt_path" "$VIDEO_LENGTH" "$VX" "$VY" "$WZ" \
      "$VIDEO_FLAT_DIR" "$status" "$elapsed" "$log_file" "$video_file" >> "$MANIFEST"
    echo "[FAIL] motor=$m ckpt=$ck status=$status"
    return
  fi

  if [[ -s "$video_file" ]]; then
    status="skip_existing"
    elapsed=0
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
      "$TASK" "$RUN_DIR" "$SCENARIO" "$m" "$ck" "$ckpt_path" "$VIDEO_LENGTH" "$VX" "$VY" "$WZ" \
      "$VIDEO_FLAT_DIR" "$status" "$elapsed" "$log_file" "$video_file" >> "$MANIFEST"
    echo "[SKIP] motor=$m ckpt=$ck status=$status"
    return
  fi

  start_ts="$(date +%s)"
  echo "[RUN ] motor=$m ckpt=$ck"
  local cmd=(
    "$PYTHON_BIN" play.py
    --task "$TASK"
    --checkpoint "$ckpt_path"
    --num_envs "$NUM_ENVS"
    --video --video_length "$VIDEO_LENGTH"
    --video_flat_folder "$VIDEO_FLAT_DIR"
    --force_fault_scenario "$SCENARIO"
    --force_fault_motor_id "$m"
    --force_walk_command
    --play_cmd_lin_x "$VX" --play_cmd_lin_y "$VY" --play_cmd_ang_z "$WZ"
    --follow_cam_offset_x "$FOLLOW_CAM_OFFSET_X"
    --follow_cam_offset_y "$FOLLOW_CAM_OFFSET_Y"
    --follow_cam_offset_z "$FOLLOW_CAM_OFFSET_Z"
    --follow_cam_lookat_z "$FOLLOW_CAM_LOOKAT_Z"
    --follow_cam_smooth_alpha "$FOLLOW_CAM_SMOOTH_ALPHA"
    --seed "$SEED"
    --disable_fabric
    --headless
  )
  if [[ "$FOLLOW_CAMERA" == "1" ]]; then
    cmd+=(--follow_camera)
  fi
  if [[ "$FOLLOW_CAM_USE_YAW_ONLY" == "1" ]]; then
    cmd+=(--follow_cam_use_yaw_only)
  else
    cmd+=(--no-follow-cam-use-yaw-only)
  fi

  if "${cmd[@]}" > "$log_file" 2>&1; then
    status="ok"
  else
    status="fail"
  fi
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))
  echo "[DONE] motor=$m ckpt=$ck status=$status elapsed=${elapsed}s"
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$TASK" "$RUN_DIR" "$SCENARIO" "$m" "$ck" "$ckpt_path" "$VIDEO_LENGTH" "$VX" "$VY" "$WZ" \
    "$VIDEO_FLAT_DIR" "$status" "$elapsed" "$log_file" "$video_file" >> "$MANIFEST"
}

active_jobs=0

for ck in "${CKPTS[@]}"; do
  for m in "${MOTORS[@]}"; do
    run_one "$ck" "$m" &
    active_jobs=$((active_jobs + 1))
    if (( active_jobs >= PARALLEL_JOBS )); then
      wait -n || true
      active_jobs=$((active_jobs - 1))
    fi
  done
done

wait

echo "[FINISH] manifest: $MANIFEST"

echo "[INFO] Generated video dirs under:" 
find "$RUN_DIR/videos/play" -maxdepth 1 -type d -name "${SCENARIO}_m*_vx*_*" | sort || true
