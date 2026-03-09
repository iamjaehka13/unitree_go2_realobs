# Paper B 실행 및 실로봇 가이드 (KR)

작성일: 2026-03-09  
논문 제목: `Real-Observable Safety Locomotion under Motor Degradation`

이 문서는 현재 코드베이스의 active execution guide다. 목적은 Paper B 기준의 학습, 평가, 실로봇 로그 수집, governor 검증 절차를 하나의 문서로 고정하는 것이다.

## 1) 현재 프로젝트의 메인 질문

핵심 질문은 하나다.

`measurable-only real-observable channels만으로 motor degradation 하의 safety benefit을 얼마나 회복할 수 있는가`

메인 비교 사다리는 아래 4개다.

1. `Baseline`
2. `RealObs-ObsOnly`
3. `RealObs-Full`
4. `Strategic PHM`

사이드 실험은 아래 2개만 유지한다.

1. `RealObs-TempDose`
2. `RealObs-Full governor on/off`

## 2) 현재 canonical task surface

현재 public task ID는 아래 다섯 개만 쓴다.

1. `Unitree-Go2-Baseline-v1`
2. `Unitree-Go2-ObsOnly-v1`
3. `Unitree-Go2-RealObs-v1`
4. `Unitree-Go2-Strategic-v1`
5. `Unitree-Go2-TempDose-v1`

해석은 다음처럼 고정한다.

1. `Baseline`
   same degraded physics, no hidden degradation observations, no degradation-aware rewards
2. `RealObs-ObsOnly`
   measurable-only observations, baseline rewards
3. `RealObs-Full`
   measurable-only observations, observable safety rewards
4. `Strategic PHM`
   privileged upper bound
5. `RealObs-TempDose`
   thermal-dose side ablation only

등록 시작점은 아래 파일이다.

- `unitree_go2_realobs/source/unitree_go2_realobs/unitree_go2_realobs/tasks/manager_based/unitree_go2_realobs/__init__.py`

## 3) 메인 실행 파일

실험 표면은 아래 파일들로 고정한다.

- 메인 실험 가이드: `EXPERIMENT_GUIDE.md`
- 코어 학습 wrapper: `unitree_go2_realobs/scripts/rsl_rl/run_paper_b_core_train.sh`
- 코어 평가 wrapper: `unitree_go2_realobs/scripts/rsl_rl/run_paper_b_core_eval.sh`
- 학습 엔트리: `unitree_go2_realobs/scripts/rsl_rl/train.py`
- 평가 엔트리: `unitree_go2_realobs/scripts/rsl_rl/evaluate.py`
- runtime override helper: `unitree_go2_realobs/scripts/rsl_rl/paper_b_runtime.py`
- motor sweep 요약: `unitree_go2_realobs/scripts/rsl_rl/summarize_motor_sweep.py`
- 실로봇 가이드: `unitree_go2_realobs/scripts/real/README.md`
- 실로봇 수집 런북: `third_party/PAPER_B_REAL_LOG_RUNBOOK_KR.txt`

## 4) 기본 학습 프로토콜

작업 위치:

```bash
cd <repo-root>/unitree_go2_realobs/scripts/rsl_rl
```

개별 학습:

```bash
bash run_paper_b_core_train.sh baseline
bash run_paper_b_core_train.sh obsonly
bash run_paper_b_core_train.sh realobs
bash run_paper_b_core_train.sh strategic
```

전체 학습:

```bash
bash run_paper_b_core_train.sh all
```

기본값은 아래처럼 둔다.

1. `MAX_ITERS=8000`
2. `NUM_ENVS=4096`
3. `SEED=45`
4. pair-uniform exposure on
5. hold steps `1000`
6. adaptive pair/motor off
7. governor off

주요 환경변수:

```bash
SEED=45
NUM_ENVS=4096
MAX_ITERS=8000
HOLD_STEPS=1000
OBS_ABLATION=none
SENSOR_PRESET=full
CRITICAL_GOVERNOR=off
```

직접 실행이 필요하면 아래 형식을 쓴다.

```bash
python3 train.py \
  --task Unitree-Go2-RealObs-v1 \
  --agent rsl_rl_cfg_entry_point \
  --num_envs 4096 \
  --max_iterations 8000 \
  --train_fault_mode single_motor_random \
  --train_fault_pair_uniform \
  --train_fault_hold_steps 1000 \
  --realobs_obs_ablation none \
  --realobs_sensor_preset full \
  --no-critical_governor_enable \
  --headless
```

## 5) 기본 평가 프로토콜

평가는 fixed 12-motor sweep를 기본으로 둔다.

```bash
cd <repo-root>/unitree_go2_realobs/scripts/rsl_rl

bash run_paper_b_core_eval.sh baseline
bash run_paper_b_core_eval.sh obsonly
bash run_paper_b_core_eval.sh realobs
bash run_paper_b_core_eval.sh strategic
```

사이드 실험:

```bash
bash run_paper_b_core_eval.sh tempdose
GOVERNOR_MODE=both bash run_paper_b_core_eval.sh realobs
```

기본 평가 규칙:

1. `paper_protocol_strict=true`
2. `eval_fault_mode=single_motor_fixed`
3. `eval_fault_motor_id=0..11`
4. `governor=off`
5. `EVAL_PROTOCOL_MODE=combined`
6. `OBS_ABLATION=none`
7. `SENSOR_PRESET=full`

주요 환경변수:

```bash
NUM_ENVS=64
NUM_EPISODES=50
EVAL_PROTOCOL_MODE=combined
EVAL_CMD_PROFILE=from_env
EVAL_SAFETY_CMD_PROFILE=stand
GOVERNOR_MODE=off
```

결과 요약:

```bash
python3 summarize_motor_sweep.py \
  --eval_results_dir ./eval_results/paper_b_core_<stamp> \
  --variant baseline='baseline_govoff/*' \
  --variant obsonly='obsonly_govoff/*' \
  --variant realobs='realobs_govoff/*' \
  --variant strategic='strategic_govoff/*' \
  --out_json /tmp/paper_b_summary.json \
  --out_csv /tmp/paper_b_summary.csv
```

## 6) Paper B ablation 규칙

채널 ablation은 `RealObs`에만 건다.

```bash
OBS_ABLATION=no_voltage bash run_paper_b_core_train.sh realobs
OBS_ABLATION=no_thermal bash run_paper_b_core_train.sh realobs
OBS_ABLATION=no_vibration bash run_paper_b_core_train.sh realobs
```

sensor realism ablation도 `RealObs` 중심으로만 돈다.

```bash
SENSOR_PRESET=ideal bash run_paper_b_core_train.sh realobs
SENSOR_PRESET=voltage_only bash run_paper_b_core_train.sh realobs
SENSOR_PRESET=encoder_transport bash run_paper_b_core_train.sh realobs
```

CLI override는 `train.py`, `evaluate.py`에서 동일하게 지원한다.

1. `--critical_governor_enable` / `--no-critical_governor_enable`
2. `--realobs_obs_ablation {none,no_voltage,no_thermal,no_vibration}`
3. `--realobs_sensor_preset {full,ideal,voltage_only,encoder_transport}`

## 7) 실로봇 연동 절차

실행 위치:

```bash
cd <repo-root>
```

### 7.1 SDK 확인

```bash
python3 unitree_go2_realobs/scripts/real/check_sdk_setup.py
```

### 7.2 UDP bridge 빌드/실행

```bash
cmake -S unitree_go2_realobs/scripts/real/sdk2_bridge -B /tmp/go2_udp_bridge_build
cmake --build /tmp/go2_udp_bridge_build -j
/tmp/go2_udp_bridge_build/go2_udp_bridge enp3s0 --auto-stand-up
```

### 7.3 라이브 governor 실행

기본 command schedule 예시는 아래 파일을 쓴다.

- `unitree_go2_realobs/scripts/real/command_schedule_example.yaml`

실행:

```bash
python3 unitree_go2_realobs/scripts/real/run_governor_live_template.py \
  --command_file unitree_go2_realobs/scripts/real/command_schedule_example.yaml \
  --state_host 127.0.0.1 --state_port 17001 \
  --cmd_host 127.0.0.1 --cmd_port 17002 \
  --out_dir ./real_runs/$(date +%Y%m%d_%H%M%S)_paperb
```

거버너 공통 로직은 아래 파일을 기준으로 본다.

- `unitree_go2_realobs/scripts/rsl_rl/governor_utils.py`

### 7.4 raw log와 50Hz step log를 같이 확보

논문용 최소 산출물:

1. raw `LowState` log
2. `steps.csv`
3. `summary.json`
4. run metadata

raw log 수집 프로토콜은 아래 런북을 따른다.

- `third_party/PAPER_B_REAL_LOG_RUNBOOK_KR.txt`

### 7.5 오프라인 governor 재평가

```bash
python3 unitree_go2_realobs/scripts/real/offline_governor_eval_from_log.py \
  --input_csv ./real_runs/<run_dir>/steps.csv \
  --output_json ./real_runs/<run_dir>/offline_eval.json \
  --governor
```

필요하면 raw 500Hz 로그를 50Hz command schedule CSV로 변환한다.

```bash
python3 unitree_go2_realobs/scripts/real/log_to_replay_csv.py \
  --input_csv ./go2_full_log.csv \
  --output_csv ./go2_replay_50hz.csv
```

## 8) 실로봇-시뮬 정렬에서 꼭 지킬 것

1. 메인 논문 claim은 시뮬 `Baseline / ObsOnly / RealObs / Strategic` quartet에서 낸다.
2. 실로봇 섹션은 threshold 적합성, governor on/off, metric alignment를 보강하는 용도다.
3. `Strategic`는 실배치 정책이라고 주장하지 않는다.
4. `TempDose`는 메인 표가 아니라 side ablation이다.
5. 실로봇에서도 가능한 한 동일 command schedule을 유지한다.

## 9) 지금 쓰지 않는 옛 흐름

다음은 active guide에서 뺀다.

1. distillation workflow
2. legacy pairuniform wrapper scripts
3. old replay/debug comparison scripts
4. legacy tuned-baseline comparison

pair-uniform exposure 자체는 현재 코어 학습 wrapper의 내부 설정으로만 쓴다. 별도 legacy wrapper를 메인 문서에서 다시 끌어오지 않는다.

## 10) 같이 보는 문서

1. `EXPERIMENT_GUIDE.md`
2. `docs/PAPER_B_MANUSCRIPT_BLUEPRINT_KR.txt`
3. `docs/PAPER_B_PRESENTATION_MASTER_KR.txt`
4. `docs/PAPER_B_REAL_CONSTANTS_CALIBRATION_KR.txt`
5. `unitree_go2_realobs/scripts/real/README.md`
6. `third_party/PAPER_B_REAL_LOG_RUNBOOK_KR.txt`
