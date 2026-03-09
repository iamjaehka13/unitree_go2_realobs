# ---------------------------------------------------------------------
# unitree_go2_realobs/mdp/commands/command.py
# ---------------------------------------------------------------------
# Copyright (c) 2024, Unitree Go2 MotorDeg Project.
# All rights reserved.

"""
[Isaac Lab 2.3.1 / 2.1 Compatible]
Custom Command Generator for Scalar Values.
Purpose: MotorDeg 연구를 위한 Risk Factor, Temperature Limit 등의 단일 변수(Scalar) 제어.
Visualization: 로봇 상단에 스칼라 값에 비례하는 크기의 구(Sphere)를 표시합니다.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

# [Isaac Lab Core]
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils.configclass import configclass

# [Isaac Lab Utils & Markers]
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

# [Isaac Lab Assets]
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformScalarCommand(CommandTerm):
    """
    [구현 클래스] 설정된 범위 내에서 스칼라 값을 샘플링하는 커맨드 생성기.
    """

    cfg: UniformScalarCommandCfg

    def __init__(self, cfg: UniformScalarCommandCfg, env: ManagerBasedEnv):
        # Keep marker config but create marker lazily only when debug vis is enabled.
        self._marker_cfg = cfg.marker_cfg
        self._markers: VisualizationMarkers | None = None
        self._debug_vis = bool(cfg.debug_vis)

        # 상위 클래스 초기화
        super().__init__(cfg, env)

        # [CRITICAL 3] 내부 버퍼 생성
        self._command_buf = torch.zeros(self.num_envs, 1, device=self.device)

        # [CRITICAL 4] Iteration Protocol Error Fix (KeyError '0' 방지)
        # env.scene에서 직접 가져오기
        try:
            self.robot: Articulation = env.scene[cfg.asset_name]
        except KeyError:
            # 혹시나 해서 예외처리 추가
            raise ValueError(f"[UniformScalarCommand] Asset '{cfg.asset_name}' not found in env.scene keys: {list(env.scene.keys())}")
        
        # 초기 가시성 설정
        self._set_debug_vis_impl(cfg.debug_vis)

    # -------------------------------------------------------------------------
    # [Essential] Abstract Interface Implementation
    # -------------------------------------------------------------------------
    @property
    def command(self) -> torch.Tensor:
        """The command tensor (Required by Isaac Lab)."""
        return self._command_buf

    def _resample_command(self, env_ids: Sequence[int]):
        """
        [Resample Logic] (Required by Isaac Lab)
        """
        num_resets = len(env_ids)
        
        # 0~1 랜덤 생성
        r = torch.rand(num_resets, 1, device=self.device)
        
        # 선형 변환 (Min ~ Max)
        scale = self.cfg.maximum - self.cfg.minimum
        val = scale * r + self.cfg.minimum
        
        # 값 할당
        self._command_buf[env_ids] = val

    def _update_metrics(self):
        """Metrics logging (Required by Isaac Lab)."""
        pass

    # -------------------------------------------------------------------------
    # [Safety Fix] Visualization Update with 'dt' Argument
    # -------------------------------------------------------------------------
    def _update_command(self, dt: float = 0.0): 
        """
        [Update Logic] 시각화 업데이트.
        """
        # _debug_vis 변수가 없어서 죽는 일은 이제 없습니다.
        if not self._debug_vis:
            return
        if self._markers is None:
            return

        # 1. 로봇 위치
        root_pos = self.robot.data.root_pos_w 

        # 2. 마커 위치 (머리 위 0.6m)
        marker_pos = root_pos.clone()
        marker_pos[:, 2] += 0.6 

        # 3. 마커 크기 (값에 비례)
        scale_factor = 0.2 + self._command_buf
        marker_scale = scale_factor.repeat(1, 3)

        # 4. 시각화 수행
        self._markers.visualize(translations=marker_pos, scales=marker_scale)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # [CRITICAL 5] 상태 동기화
        self._debug_vis = bool(debug_vis)
        if self._debug_vis and self._markers is None:
            self._markers = VisualizationMarkers(self._marker_cfg)
        if self._markers is not None:
            self._markers.set_visibility(self._debug_vis)


@configclass
class UniformScalarCommandCfg(CommandTermCfg):
    """
    [설정 클래스] 반드시 구현 클래스보다 아래에 위치해야 함.
    """
    class_type: type = UniformScalarCommand
    
    # [Scalar Range]
    minimum: float = 0.0
    maximum: float = 1.0
    
    # [Target Asset]
    asset_name: str = "robot"

    # [Debug Visualization]
    debug_vis: bool = False 

    # [VISUALIZATION CONFIG]
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/scalar_value",
        markers={
            "value_indicator": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0), # 녹색
                    emissive_color=(0.0, 0.2, 0.0) 
                ),
            ),
        },
    )
