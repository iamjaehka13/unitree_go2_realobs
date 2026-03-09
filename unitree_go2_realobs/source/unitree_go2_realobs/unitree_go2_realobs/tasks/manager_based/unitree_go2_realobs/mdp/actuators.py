from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.actuators import ImplicitActuator, ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from ..motor_deg.state import MotorDegState

class MotorDegRealismActuator(ImplicitActuator):
    
    def __init__(self, cfg: ImplicitActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        

        self.nominal_kp = self.stiffness.clone()
        self.nominal_kd = self.damping.clone()
        
        self._motor_deg_state: MotorDegState | None = None
        self._asset: Articulation | None = None

    def bind_motor_deg_state(self, motor_deg_state: MotorDegState):
        self._motor_deg_state = motor_deg_state

    def bind_asset(self, asset: Articulation):
        self._asset = asset

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor

    ) -> ArticulationActions:
        
        # [Note] Stiffness/Damping injection is handled by
        # UnitreeGo2MotorDegEnv._apply_physical_degradation() in the physics loop.
        # Writing here would cause a redundant double-write at substep 0.

        return super().compute(control_action, joint_pos, joint_vel)
