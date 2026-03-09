# Paper B Experiment Guide

Paper B의 메인 질문은 다음입니다.

`Baseline -> RealObs-ObsOnly -> RealObs-Full -> Strategic upper bound`

즉, measurable-only 관측이 실제로 안전성 이득을 만드는지, 그리고 그 이득이 reward shaping과 분리되는지를 보여주는 실험 체계를 기본으로 둡니다.

## Public Task Surface

- `Unitree-Go2-Baseline-v1`
- `Unitree-Go2-ObsOnly-v1`
- `Unitree-Go2-RealObs-v1`
- `Unitree-Go2-TempDose-v1`
- `Unitree-Go2-Strategic-v1`

메인 본문:
- `Baseline`
- `RealObs-ObsOnly`
- `RealObs-Full`
- `Strategic`

사이드 ablation:
- `TempDose`
- `RealObs-Full` with governor `on/off`

## Core Training

기본 8k 학습:

```bash
cd <repo-root>/unitree_go2_realobs/scripts/rsl_rl

bash run_paper_b_core_train.sh baseline
bash run_paper_b_core_train.sh obsonly
bash run_paper_b_core_train.sh realobs
bash run_paper_b_core_train.sh strategic
```

한 번에 모두:

```bash
bash run_paper_b_core_train.sh all
```

기본값:
- pair-uniform fault exposure on
- hold steps `1000`
- adaptive pair/motor off
- governor off
- `MAX_ITERS=8000`
- `SEED=45`

주요 환경변수:

```bash
SEED=45
NUM_ENVS=4096
MAX_ITERS=8000
OBS_ABLATION=none
SENSOR_PRESET=full
CRITICAL_GOVERNOR=off
```

## Core Evaluation

고정 motor sweep:

```bash
cd <repo-root>/unitree_go2_realobs/scripts/rsl_rl

bash run_paper_b_core_eval.sh baseline
bash run_paper_b_core_eval.sh obsonly
bash run_paper_b_core_eval.sh realobs
bash run_paper_b_core_eval.sh strategic
```

사이드 ablation:

```bash
bash run_paper_b_core_eval.sh tempdose
GOVERNOR_MODE=both bash run_paper_b_core_eval.sh realobs
```

기본 평가 프로토콜:
- `paper_protocol_strict=true`
- `eval_fault_mode=single_motor_fixed`
- motor id `0..11` sweep
- governor off

주요 환경변수:

```bash
NUM_ENVS=64
NUM_EPISODES=50
EVAL_PROTOCOL_MODE=combined
EVAL_CMD_PROFILE=from_env
EVAL_SAFETY_CMD_PROFILE=stand
OBS_ABLATION=none
SENSOR_PRESET=full
GOVERNOR_MODE=off
```

## Paper B Ablations

관측 채널 ablation:

```bash
OBS_ABLATION=no_voltage bash run_paper_b_core_train.sh realobs
OBS_ABLATION=no_thermal bash run_paper_b_core_train.sh realobs
OBS_ABLATION=no_vibration bash run_paper_b_core_train.sh realobs
```

sensor realism ablation:

```bash
SENSOR_PRESET=ideal bash run_paper_b_core_train.sh realobs
SENSOR_PRESET=voltage_only bash run_paper_b_core_train.sh realobs
SENSOR_PRESET=encoder_transport bash run_paper_b_core_train.sh realobs
```

지원되는 runtime flag:
- `--critical_governor_enable` / `--no-critical_governor_enable`
- `--realobs_obs_ablation {none,no_voltage,no_thermal,no_vibration}`
- `--realobs_sensor_preset {full,ideal,voltage_only,encoder_transport}`

## Result Aggregation

motor sweep 결과 요약:

```bash
python summarize_motor_sweep.py \
  --eval_results_dir unitree_go2_realobs/scripts/rsl_rl/eval_results/paper_b_core_<stamp> \
  --variant baseline='baseline_govoff/*' \
  --variant obsonly='obsonly_govoff/*' \
  --variant realobs='realobs_govoff/*' \
  --variant strategic='strategic_govoff/*' \
  --out_json /tmp/paper_b_summary.json \
  --out_csv /tmp/paper_b_summary.csv
```

## Deprecated Workflows

다음 워크플로는 Paper B primary surface에서 제외합니다.

- legacy tuning workflows
- legacy distillation workflow
- replay/debug pipeline
- legacy comparison scripts

이들은 historical reference로만 취급하고, 메인 README와 메인 실험 표에서는 사용하지 않습니다.
