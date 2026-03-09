from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_CASE_TEMPERATURE_NAMES = (
    "motor_case_temp",
    "case_temp",
    "motor_temp_case",
    "housing_temp",
    "motor_housing_temp",
)

_CASE_TEMPERATURE_DERIVATIVE_NAMES = (
    "case_temp_derivative",
    "motor_case_temp_derivative",
    "housing_temp_derivative",
)


def _cfg_bool(env: ManagerBasedEnv, name: str, default: bool) -> bool:
    cfg = getattr(env, "cfg", None)
    return bool(getattr(cfg, name, default))


def resolve_realobs_voltage_tensor(
    env: ManagerBasedEnv,
    *,
    allow_true_voltage_fallback: bool | None = None,
    require_voltage_sensor: bool | None = None,
) -> tuple[torch.Tensor | None, str]:
    """Resolve the policy-facing RealObs voltage channel under an explicit contract."""
    if not hasattr(env, "motor_deg_state"):
        return None, "missing_state"

    if allow_true_voltage_fallback is None:
        allow_true_voltage_fallback = _cfg_bool(env, "realobs_allow_true_voltage_fallback", False)
    if require_voltage_sensor is None:
        require_voltage_sensor = _cfg_bool(env, "realobs_require_voltage_sensor", False)

    deg_state = env.motor_deg_state
    if hasattr(deg_state, "battery_voltage"):
        val = getattr(deg_state, "battery_voltage")
        if isinstance(val, torch.Tensor):
            return val.unsqueeze(-1), "battery_voltage"

    if allow_true_voltage_fallback and hasattr(deg_state, "battery_voltage_true"):
        val = getattr(deg_state, "battery_voltage_true")
        if isinstance(val, torch.Tensor):
            return val.unsqueeze(-1), "battery_voltage_true_fallback"

    if require_voltage_sensor:
        raise RuntimeError(
            "RealObs contract violation: required measured voltage sensor channel `battery_voltage` is missing. "
            "Current config disallows falling back to hidden `battery_voltage_true`."
        )

    if hasattr(deg_state, "battery_voltage_true"):
        val = getattr(deg_state, "battery_voltage_true")
        if isinstance(val, torch.Tensor):
            return val.unsqueeze(-1), "battery_voltage_true_fallback"

    return None, "missing_voltage"


def resolve_realobs_case_temperature_tensor(
    env: ManagerBasedEnv,
    *,
    env_ids: torch.Tensor | None = None,
    allow_coil_fallback: bool | None = None,
    require_case_proxy: bool | None = None,
    coil_to_case_delta_c: float = 5.0,
) -> tuple[torch.Tensor | None, str]:
    """Resolve the policy-facing RealObs case-temperature proxy under an explicit contract."""
    if not hasattr(env, "motor_deg_state"):
        return None, "missing_state"

    if allow_coil_fallback is None:
        allow_coil_fallback = _cfg_bool(env, "realobs_allow_case_temperature_from_coil_fallback", False)
    if require_case_proxy is None:
        require_case_proxy = _cfg_bool(env, "realobs_require_case_temperature_proxy", False)

    deg_state = env.motor_deg_state
    for name in _CASE_TEMPERATURE_NAMES:
        if hasattr(deg_state, name):
            val = getattr(deg_state, name)
            if isinstance(val, torch.Tensor):
                return (val if env_ids is None else val[env_ids]), name

    if allow_coil_fallback and hasattr(deg_state, "coil_temp"):
        val = getattr(deg_state, "coil_temp")
        if isinstance(val, torch.Tensor):
            case_like = torch.clamp(val - float(coil_to_case_delta_c), min=0.0)
            return (case_like if env_ids is None else case_like[env_ids]), "coil_temp_offset_fallback"

    if require_case_proxy:
        raise RuntimeError(
            "RealObs contract violation: explicit case/housing temperature proxy tensor is required. "
            "Current config disallows falling back to hidden `coil_temp`-derived proxy."
        )

    if hasattr(deg_state, "coil_temp"):
        val = getattr(deg_state, "coil_temp")
        if isinstance(val, torch.Tensor):
            case_like = torch.clamp(val - float(coil_to_case_delta_c), min=0.0)
            return (case_like if env_ids is None else case_like[env_ids]), "coil_temp_offset_fallback"

    return None, "missing_case_proxy"


def resolve_realobs_case_temperature_rate_tensor(
    env: ManagerBasedEnv,
    *,
    env_ids: torch.Tensor | None = None,
    allow_coil_fallback: bool | None = None,
    require_case_proxy: bool | None = None,
) -> tuple[torch.Tensor | None, str]:
    """Resolve the policy-facing RealObs case-temperature-rate proxy under an explicit contract."""
    if not hasattr(env, "motor_deg_state"):
        return None, "missing_state"

    if allow_coil_fallback is None:
        allow_coil_fallback = _cfg_bool(env, "realobs_allow_case_temperature_from_coil_fallback", False)
    if require_case_proxy is None:
        require_case_proxy = _cfg_bool(env, "realobs_require_case_temperature_proxy", False)

    deg_state = env.motor_deg_state
    for name in _CASE_TEMPERATURE_DERIVATIVE_NAMES:
        if hasattr(deg_state, name):
            val = getattr(deg_state, name)
            if isinstance(val, torch.Tensor):
                return (val if env_ids is None else val[env_ids]), name

    if allow_coil_fallback and hasattr(deg_state, "temp_derivative"):
        val = getattr(deg_state, "temp_derivative")
        if isinstance(val, torch.Tensor):
            return (val if env_ids is None else val[env_ids]), "temp_derivative_fallback"

    if require_case_proxy:
        raise RuntimeError(
            "RealObs contract violation: explicit case/housing temperature-rate proxy tensor is required. "
            "Current config disallows falling back to hidden `temp_derivative`."
        )

    if hasattr(deg_state, "temp_derivative"):
        val = getattr(deg_state, "temp_derivative")
        if isinstance(val, torch.Tensor):
            return (val if env_ids is None else val[env_ids]), "temp_derivative_fallback"

    return None, "missing_case_rate_proxy"
