#!/usr/bin/env bash
set -euo pipefail

# One-round reward tuning helper (analysis-only for reward edits).
# This script NEVER edits reward config files automatically.
#
# Flow:
#   1) evaluate.py on a given checkpoint
#   2) reward_tuning_advisor.py recommend
#   3) optional gate check against baseline eval JSON
#
# Usage example:
#   bash run_reward_tuning_round.sh \
#     --checkpoint logs/rsl_rl/unitree_go2_realobs/<run>/model_2999.pt \
#     --task Unitree-Go2-RealObs-v1 \
#     --num_envs 512 \
#     --num_episodes 100 \
#     --output_root ./eval_results/reward_tuning
#
# Optional gate:
#   --baseline_eval_json <path/to/baseline_eval.json>

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
  cat <<'USAGE'
run_reward_tuning_round.sh

Required:
  --checkpoint <path>            Policy checkpoint to evaluate.

Optional:
  --task <name>                  Default: Unitree-Go2-RealObs-v1
  --agent <entry>                Default: rsl_rl_cfg_entry_point
  --num_envs <int>               Default: 512
  --num_episodes <int>           Default: 100
  --seed <int>                   Default: 42
  --device <str>                 Default: (env default)
  --headless <0|1>               Default: 1
  --output_root <dir>            Default: ./eval_results/reward_tuning

Advisor options:
  --env_cfg <path>               Reward cfg python file path.
                                 Default: RealObs env cfg file.
  --rewards_class <name>         Default: RealObsRewardsCfg
  --max_step_change <float>      Default: 0.10 (must be <= 0.15)
  --max_terms_per_round <int>    Default: 2
  --down_step_change <float>     Default: 0.05

Gate options (optional):
  --baseline_eval_json <path>    If set, runs gate check vs candidate eval JSON.
  --max_survival_drop_abs <f>    Default: 0.02
  --max_tracking_increase_abs <f> Default: 0.01
  --max_power_increase_ratio <f> Default: 0.05
  --max_critical_survival_drop_abs <f> Default: 0.03

Notes:
  - This script does NOT modify reward files.
  - Apply only top 1-2 advisor recommendations manually, then rerun.
USAGE
}

TASK="Unitree-Go2-RealObs-v1"
AGENT="rsl_rl_cfg_entry_point"
CHECKPOINT=""
NUM_ENVS=512
NUM_EPISODES=100
SEED=42
DEVICE=""
HEADLESS=1
OUTPUT_ROOT="./eval_results/reward_tuning"

ENV_CFG="${SCRIPT_DIR}/../../source/unitree_go2_phm/unitree_go2_phm/tasks/manager_based/unitree_go2_phm/unitree_go2_realobs_env_cfg.py"
REWARDS_CLASS="RealObsRewardsCfg"
MAX_STEP_CHANGE="0.10"
MAX_TERMS_PER_ROUND=2
DOWN_STEP_CHANGE="0.05"

BASELINE_EVAL_JSON=""
MAX_SURVIVAL_DROP_ABS="0.02"
MAX_TRACKING_INCREASE_ABS="0.01"
MAX_POWER_INCREASE_RATIO="0.05"
MAX_CRITICAL_SURVIVAL_DROP_ABS="0.03"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --task) TASK="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --num_envs) NUM_ENVS="$2"; shift 2 ;;
    --num_episodes) NUM_EPISODES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --headless) HEADLESS="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --env_cfg) ENV_CFG="$2"; shift 2 ;;
    --rewards_class) REWARDS_CLASS="$2"; shift 2 ;;
    --max_step_change) MAX_STEP_CHANGE="$2"; shift 2 ;;
    --max_terms_per_round) MAX_TERMS_PER_ROUND="$2"; shift 2 ;;
    --down_step_change) DOWN_STEP_CHANGE="$2"; shift 2 ;;
    --baseline_eval_json) BASELINE_EVAL_JSON="$2"; shift 2 ;;
    --max_survival_drop_abs) MAX_SURVIVAL_DROP_ABS="$2"; shift 2 ;;
    --max_tracking_increase_abs) MAX_TRACKING_INCREASE_ABS="$2"; shift 2 ;;
    --max_power_increase_ratio) MAX_POWER_INCREASE_RATIO="$2"; shift 2 ;;
    --max_critical_survival_drop_abs) MAX_CRITICAL_SURVIVAL_DROP_ABS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERR] Unknown argument: $1"; usage; exit 2 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "[ERR] --checkpoint is required."
  usage
  exit 2
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[ERR] Checkpoint not found: $CHECKPOINT"
  exit 2
fi

if [[ ! -f "$ENV_CFG" ]]; then
  echo "[ERR] Env cfg not found: $ENV_CFG"
  exit 2
fi

python3 - <<PY
v = float("${MAX_STEP_CHANGE}")
if not (0.0 < v <= 0.15):
    raise SystemExit("[ERR] --max_step_change must be in (0, 0.15].")
PY

STAMP="$(date +%Y%m%d_%H%M%S)"
ROUND_DIR="${OUTPUT_ROOT%/}/round_${STAMP}"
mkdir -p "$ROUND_DIR"

echo "[INFO] Reward tuning round dir: $ROUND_DIR"
echo "[INFO] Task: $TASK"
echo "[INFO] Checkpoint: $CHECKPOINT"

EVAL_CMD=(
  python3 evaluate.py
  --task "$TASK"
  --agent "$AGENT"
  --checkpoint "$CHECKPOINT"
  --num_envs "$NUM_ENVS"
  --num_episodes "$NUM_EPISODES"
  --seed "$SEED"
  --output_dir "$ROUND_DIR"
  # Reward tuning is exploratory; don't enforce paper fixed-fault protocol.
  --no-paper-protocol-strict
)
if [[ -n "$DEVICE" ]]; then
  EVAL_CMD+=(--device "$DEVICE")
fi
if [[ "$HEADLESS" == "1" ]]; then
  EVAL_CMD+=(--headless)
fi

echo "[RUN] ${EVAL_CMD[*]}"
"${EVAL_CMD[@]}"

EVAL_JSON="$(ls -1t "$ROUND_DIR"/eval_*.json 2>/dev/null | head -n1 || true)"
if [[ -z "$EVAL_JSON" ]]; then
  echo "[ERR] evaluate.py did not produce eval_*.json under $ROUND_DIR"
  exit 3
fi
REWARD_CSV="$(ls -1t "$ROUND_DIR"/reward_breakdown_*.csv 2>/dev/null | head -n1 || true)"

RECOMMEND_JSON="$ROUND_DIR/recommendation.json"
REC_CMD=(
  python3 reward_tuning_advisor.py recommend
  --eval_json "$EVAL_JSON"
  --env_cfg "$ENV_CFG"
  --rewards_class "$REWARDS_CLASS"
  --max_step_change "$MAX_STEP_CHANGE"
  --max_terms_per_round "$MAX_TERMS_PER_ROUND"
  --down_step_change "$DOWN_STEP_CHANGE"
  --output_json "$RECOMMEND_JSON"
)
if [[ -n "$REWARD_CSV" ]]; then
  REC_CMD+=(--reward_csv "$REWARD_CSV")
fi

echo "[RUN] ${REC_CMD[*]}"
"${REC_CMD[@]}"

if [[ -n "$BASELINE_EVAL_JSON" ]]; then
  if [[ ! -f "$BASELINE_EVAL_JSON" ]]; then
    echo "[ERR] baseline_eval_json not found: $BASELINE_EVAL_JSON"
    exit 4
  fi
  GATE_JSON="$ROUND_DIR/gate.json"
  GATE_CMD=(
    python3 reward_tuning_advisor.py gate
    --baseline_eval_json "$BASELINE_EVAL_JSON"
    --candidate_eval_json "$EVAL_JSON"
    --max_survival_drop_abs "$MAX_SURVIVAL_DROP_ABS"
    --max_tracking_increase_abs "$MAX_TRACKING_INCREASE_ABS"
    --max_power_increase_ratio "$MAX_POWER_INCREASE_RATIO"
    --max_critical_survival_drop_abs "$MAX_CRITICAL_SURVIVAL_DROP_ABS"
    --output_json "$GATE_JSON"
  )
  echo "[RUN] ${GATE_CMD[*]}"
  "${GATE_CMD[@]}"
fi

echo ""
echo "[DONE] Round complete."
echo "  eval_json:        $EVAL_JSON"
echo "  reward_csv:       ${REWARD_CSV:-<not generated>}"
echo "  recommendation:   $RECOMMEND_JSON"
if [[ -n "$BASELINE_EVAL_JSON" ]]; then
  echo "  gate_json:        $GATE_JSON"
fi
echo ""
echo "Next:"
echo "  1) Apply only top 1-2 recommendation edits manually (<=10~15%)."
echo "  2) Retrain and rerun this script."
echo "  3) If gate fails, rollback previous reward set."
