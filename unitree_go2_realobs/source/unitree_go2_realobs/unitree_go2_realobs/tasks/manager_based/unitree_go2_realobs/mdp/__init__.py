"""MDP terms for the Unitree Go2 real-observable locomotion tasks."""

from __future__ import annotations

import isaaclab.envs.mdp as _isaac_mdp

from .actuators import MotorDegRealismActuator
from .commands import UniformScalarCommand, UniformScalarCommandCfg
from . import observations as _observations
from . import rewards as _rewards
from . import terminations as _terminations

# Re-export Isaac Lab standard MDP API without wildcard imports.
__all__ = []
for _name in dir(_isaac_mdp):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_isaac_mdp, _name)
    __all__.append(_name)

# Re-export MotorDeg custom API.
_custom_exports = {
    "MotorDegRealismActuator": MotorDegRealismActuator,
    "UniformScalarCommand": UniformScalarCommand,
    "UniformScalarCommandCfg": UniformScalarCommandCfg,
}
for _name, _obj in _custom_exports.items():
    globals()[_name] = _obj
    __all__.append(_name)

for _module in (_observations, _rewards, _terminations):
    _module_names = getattr(_module, "__all__", None)
    if _module_names is None:
        _module_names = [
            n
            for n, obj in vars(_module).items()
            if (not n.startswith("_")) and (getattr(obj, "__module__", None) == _module.__name__)
        ]
    for _name in _module_names:
        globals()[_name] = getattr(_module, _name)
    __all__.extend(_module_names)
