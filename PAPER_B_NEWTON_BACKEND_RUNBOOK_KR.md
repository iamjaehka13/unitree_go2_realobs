# Paper B Newton(MuJoCo-Warp) 실행 런북

작성일: 2026-02-16

이 문서는 **Isaac Lab에서 Unitree Go2 RealObs 정책을 Newton(MuJoCo-Warp 백엔드)**로 실행/재생하기 위한 최소 절차를 정리합니다.

## 0) 전제 경로

- IsaacLab 루트: `/home/iamjaehka13/IsaacLab`
- 이 프로젝트 루트: `<repo-root>`
- 확장 패키지 경로: `<repo-root>/unitree_go2_realobs/source/unitree_go2_realobs`
- 재생 스크립트 경로: `<repo-root>/unitree_go2_realobs/scripts/rsl_rl/play.py`

## 1) IsaacLab 루트에서 Newton 브랜치 준비

Newton 통합은 실험 브랜치(`feature/newton`) 기준으로 사용하는 것을 권장합니다.

```bash
source /home/iamjaehka13/miniforge3/etc/profile.d/conda.sh
conda activate isaaclab

cd /home/iamjaehka13/IsaacLab

git fetch origin
git switch --track origin/feature/newton   # 이미 로컬 브랜치가 있으면: git switch feature/newton
```

## 2) 의존성 설치

```bash
cd /home/iamjaehka13/IsaacLab

# 선택: 캐시 이슈가 있을 때만 수행
pip cache purge

./isaaclab.sh -i
```

## 3) Go2 RealObs 확장 설치 (IsaacLab Python으로)

```bash
cd /home/iamjaehka13/IsaacLab

./isaaclab.sh -p -m pip install -e \
  <repo-root>/unitree_go2_realobs/source/unitree_go2_realobs
```

## 4) 태스크 등록 확인

```bash
cd /home/iamjaehka13/IsaacLab

./isaaclab.sh -p <repo-root>/unitree_go2_realobs/scripts/list_envs.py
```

아래 태스크가 보이면 정상입니다.

- `Unitree-Go2-Baseline-v1`
- `Unitree-Go2-ObsOnly-v1`
- `Unitree-Go2-RealObs-v1`
- `Unitree-Go2-Strategic-v1`
- `Unitree-Go2-TempDose-v1`

## 5) Newton으로 체크포인트 재생

```bash
cd /home/iamjaehka13/IsaacLab

./isaaclab.sh -p <repo-root>/unitree_go2_realobs/scripts/rsl_rl/play.py \
  --task Unitree-Go2-RealObs-v1 \
  --num_envs 32 \
  --checkpoint /absolute/path/to/model_2999.pt \
  --newton_visualizer
```

원격/저사양 환경에서 렌더를 끄려면:

```bash
./isaaclab.sh -p <repo-root>/unitree_go2_realobs/scripts/rsl_rl/play.py \
  --task Unitree-Go2-RealObs-v1 \
  --num_envs 32 \
  --checkpoint /absolute/path/to/model_2999.pt \
  --newton_visualizer \
  --headless
```

## 6) (선택) Newton에서 바로 학습

```bash
cd /home/iamjaehka13/IsaacLab

./isaaclab.sh -p <repo-root>/unitree_go2_realobs/scripts/rsl_rl/train.py \
  --task Unitree-Go2-RealObs-v1 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --headless \
  --newton_visualizer
```

## 7) 자주 나는 오류

1. `python: command not found`
- `conda activate isaaclab`를 먼저 수행하지 않은 경우가 대부분입니다.

2. `Task ... not found`
- 3번의 editable install이 안 된 상태입니다.
- `./isaaclab.sh -p -m pip install -e <path>`를 다시 실행하세요.

3. 체크포인트를 못 찾음
- `--checkpoint`는 가급적 절대경로를 사용하세요.

4. PhysX와 Newton을 혼용하려다 충돌
- Newton 실험은 `feature/newton`, PhysX 실험은 `main` 또는 `v2.3.1`로 분리해 운영하는 것이 안전합니다.

## 8) 브랜치 전환 메모

Newton 실험 후 PhysX 파이프라인으로 돌아갈 때:

```bash
cd /home/iamjaehka13/IsaacLab
git switch main   # 또는 git switch --detach v2.3.1
./isaaclab.sh -i
```
