# unitree_go2_phm/phm/buffers/long_term.py
from __future__ import annotations
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class LongTermHealthBuffer:
    """
    [PHM Core] 로봇의 장기적 건강 상태 변화와 열화 추이를 추적하는 통계 버퍼.

    Notes:
    1. fill_count 기반 slope 분모로 cold-start 왜곡을 줄인다.
    2. 환경 내부 스텝 카운터에 의존하지 않고 자체 타이머를 사용한다.
    3. 관절 개수를 인자로 받아 태스크별로 재사용할 수 있다.
    """
    def __init__(self, num_envs: int, num_joints: int, device: str, history_length: int = 10, snapshot_interval: int = 100):
        """
        Args:
            num_envs: 환경 개수
            num_joints: 관절 개수
            device: 디바이스
            history_length: 윈도우 크기
            snapshot_interval: 스냅샷 주기
        """
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = device
        self.history_length = history_length
        self.snapshot_interval = snapshot_interval

        # 1. 생애 주기 통계
        self.thermal_overload_duration = torch.zeros((num_envs, num_joints), device=device)

        # 2. 노화 히스토리
        self.fatigue_snapshots = torch.zeros((num_envs, history_length, num_joints), device=device)
        self.snapshot_index = torch.zeros(num_envs, dtype=torch.long, device=device)

        # 환경 독립적인 자체 타이머 (Reset 시 1로 초기화)
        # [Fix] 1부터 시작: 0%interval==0 방지 (리셋 직후 불필요한 스냅샷 트리거 차단)
        self.step_timer = torch.ones(num_envs, dtype=torch.long, device=device)
        
        # [Fix] fill_count: 실제로 기록된 고유 스냅샷 수 (0 ~ history_length)
        # slope 계산 시 history_length 대신 max(fill_count-1, 1)로 나누어 정확한 기울기 산출.
        self.fill_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        
        # 버퍼가 한 바퀴 돌았는지 확인하는 플래그 (Cold Start 방지)
        self.is_buffer_filled = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def update(self, env: ManagerBasedRLEnv, dt: float, env_ids=None):
        """매 step 업데이트. env_ids가 지정되면 해당 환경만 갱신."""
        from ..constants import TEMP_WARN_THRESHOLD

        if env_ids is None:
            env_ids = slice(None)
        
        # [Thermal Trace]
        # [Fix] temperature -> coil_temp (state.py 정의와 일치화)
        # [Fix] env_ids 범위만 갱신하여 미갱신 환경의 이중 누적 방지.
        current_temp = env.phm_state.coil_temp[env_ids]
        
        is_overheating = (current_temp > TEMP_WARN_THRESHOLD).float()
        self.thermal_overload_duration[env_ids] += is_overheating * dt

        # [Degradation Snapshot]
        # 환경별로 타이머가 다르므로 마스킹 연산 수행 (Vectorized)
        # snapshot_interval에 도달한 환경들만 선택
        timer_subset = self.step_timer[env_ids]
        trigger_mask = (timer_subset % self.snapshot_interval == 0)
        
        if torch.any(trigger_mask):
            if isinstance(env_ids, slice):
                triggered_env_ids = torch.nonzero(trigger_mask, as_tuple=True)[0]
            else:
                triggered_env_ids = env_ids[trigger_mask]
            self._take_snapshot(env.phm_state.fatigue_index, triggered_env_ids)

        # [Internal Timer Update] 스냅샷 판정 후 증가 (1-based timer)
        self.step_timer[env_ids] += 1

    def _take_snapshot(self, current_fatigue: torch.Tensor, env_ids: torch.Tensor):
        """선택된 환경들에 대해 스냅샷 저장"""
        # (N_subset, Joints)
        subset_fatigue = current_fatigue[env_ids]
        
        # (N_subset, 1, 1) -> (N_subset, 1, Joints)
        # 해당 환경들의 현재 스냅샷 인덱스 가져오기
        current_indices = self.snapshot_index[env_ids]
        scatter_indices = current_indices.view(-1, 1, 1).expand(-1, 1, self.num_joints)

        # 값 기록 (In-place scatter)
        self.fatigue_snapshots[env_ids].scatter_(1, scatter_indices, subset_fatigue.unsqueeze(1))
        
        # 인덱스 순환 업데이트
        new_indices = (current_indices + 1) % self.history_length
        self.snapshot_index[env_ids] = new_indices

        # [Fix] fill_count 증가 (history_length 상한)
        self.fill_count[env_ids] = torch.clamp(
            self.fill_count[env_ids] + 1, max=self.history_length
        )

        # 인덱스가 0으로 돌아왔다면, 버퍼가 가득 찬 것임
        filled_mask = (new_indices == 0)
        if torch.any(filled_mask):
            self.is_buffer_filled[env_ids[filled_mask]] = True

    def get_degradation_slope(self) -> torch.Tensor:
        """
        [PHM Insight] 노화 속도(기울기) 반환.
        
        [Fix] 버퍼가 완전히 차지 않은 경우에도 정확한 기울기를 반환:
        - 최신 스냅샷(idx_now)과 가장 오래된 스냅샷(idx_old) 사이의 차이를
          실제 스냅샷 간격(fill_count - 1)으로 나누어 정확한 단위 기울기 산출.
        - fill_count < 2이면 기울기를 계산할 수 없으므로 0.0 반환.
        """
        # 현재(최신) 포인터: index - 1 (Circular Buffer 고려)
        idx_now = (self.snapshot_index - 1) % self.history_length
        
        # 가장 오래된 스냅샷 위치:
        # - 버퍼가 찼으면(is_buffer_filled): 현재 포인터 위치가 가장 오래된 것
        # - 버퍼가 안 찼으면: 항상 0번이 가장 오래된 것 (순차 기록이므로)
        idx_old = torch.where(
            self.is_buffer_filled,
            self.snapshot_index,
            torch.zeros_like(self.snapshot_index)
        )

        # 차원 맞추기: (N, 1, 1)
        idx_now_3d = idx_now.view(-1, 1, 1).expand(-1, 1, self.num_joints)
        idx_old_3d = idx_old.view(-1, 1, 1).expand(-1, 1, self.num_joints)

        latest = torch.gather(self.fatigue_snapshots, 1, idx_now_3d).squeeze(1)
        oldest = torch.gather(self.fatigue_snapshots, 1, idx_old_3d).squeeze(1)

        # [Fix] 실제 스냅샷 간격으로 나눔 (fill_count - 1, 최소 1)
        # 버퍼가 찼으면 history_length, 안 찼으면 실제 기록된 개수 - 1
        effective_span = torch.where(
            self.is_buffer_filled,
            torch.full_like(self.fill_count, self.history_length),
            torch.clamp(self.fill_count - 1, min=1)
        ).unsqueeze(-1).float()  # (N, 1)

        slope = (latest - oldest) / effective_span

        # [Fix] fill_count < 2이면 기울기 계산 불가 (데이터 포인트 부족) → 0.0
        has_enough_data = (self.fill_count >= 2).unsqueeze(-1)
        return torch.where(has_enough_data, slope, torch.zeros_like(slope))

    def reset(self, env_ids: torch.Tensor):
        """에피소드 리셋 시 초기화"""
        self.thermal_overload_duration[env_ids] = 0.0
        self.fatigue_snapshots[env_ids] = 0.0
        self.snapshot_index[env_ids] = 0
        
        # 타이머 및 플래그 리셋
        # [Fix] step_timer를 1로 시작 (0%interval==0 방지)
        self.step_timer[env_ids] = 1
        self.fill_count[env_ids] = 0
        self.is_buffer_filled[env_ids] = False
