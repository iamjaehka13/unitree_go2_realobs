import torch
from isaaclab.managers import SceneEntityCfg

from . import contact as _contact
from . import imu as _imu
from . import observable_signals as _observable_signals
from . import proprioception as _proprioception

from ...motor_deg.models.thermal import get_thermal_stress_index as _get_thermal_stress_index
from ...motor_deg.models.degradation import get_mechanical_health_index as _get_mechanical_health_index

__all__ = []
for _module in (_contact, _imu, _observable_signals, _proprioception):
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


def get_thermal_stress_index(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if not hasattr(env, "motor_deg_state"):
        asset = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if joint_ids is None:
            joint_ids = slice(None)
        return torch.zeros((env.num_envs, asset.data.joint_pos[:, joint_ids].shape[1]), device=env.device)
    out = _get_thermal_stress_index(env, env_ids=None)
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return out
    if isinstance(joint_ids, slice):
        return out[:, joint_ids]
    idx = torch.as_tensor(joint_ids, device=out.device)
    if torch.any(idx < 0) or torch.any(idx >= out.shape[1]):
        raise RuntimeError(
            f"Invalid joint_ids for get_thermal_stress_index: {joint_ids}. "
            f"Valid range is [0, {out.shape[1] - 1}]."
        )
    return out.index_select(1, idx)


def get_mechanical_health_index(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    if not hasattr(env, "motor_deg_state"):
        asset = env.scene[asset_cfg.name]
        joint_ids = asset_cfg.joint_ids
        if joint_ids is None:
            joint_ids = slice(None)
        return torch.zeros((env.num_envs, asset.data.joint_pos[:, joint_ids].shape[1]), device=env.device)
    out = _get_mechanical_health_index(env, env_ids=None)
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        return out
    if isinstance(joint_ids, slice):
        return out[:, joint_ids]
    idx = torch.as_tensor(joint_ids, device=out.device)
    if torch.any(idx < 0) or torch.any(idx >= out.shape[1]):
        raise RuntimeError(
            f"Invalid joint_ids for get_mechanical_health_index: {joint_ids}. "
            f"Valid range is [0, {out.shape[1] - 1}]."
        )
    return out.index_select(1, idx)


__all__.extend(["get_thermal_stress_index", "get_mechanical_health_index"])
