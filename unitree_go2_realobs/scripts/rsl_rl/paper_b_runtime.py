from __future__ import annotations

from typing import Any


_REALOBS_OBS_ABLATIONS: dict[str, tuple[str, ...]] = {
    "none": (),
    "no_voltage": ("energy_budget",),
    "no_thermal": ("thermal_stress",),
    "no_vibration": ("vibration_level",),
}

_REALOBS_SENSOR_PRESETS = ("full", "ideal", "voltage_only", "encoder_transport")


def _set_if_present(obj: object, name: str, value: Any) -> None:
    if hasattr(obj, name):
        setattr(obj, name, value)


def _disable_obs_term(env_cfg: object, term_name: str) -> bool:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return False

    changed = False
    for group_name in ("policy", "critic"):
        group_cfg = getattr(observations, group_name, None)
        if group_cfg is None or not hasattr(group_cfg, term_name):
            continue
        setattr(group_cfg, term_name, None)
        changed = True
    return changed


def _set_policy_noise(env_cfg: object, term_name: str, noise_cfg: Any) -> bool:
    observations = getattr(env_cfg, "observations", None)
    policy_cfg = getattr(observations, "policy", None) if observations is not None else None
    if policy_cfg is None or not hasattr(policy_cfg, term_name):
        return False

    term_cfg = getattr(policy_cfg, term_name)
    if term_cfg is None:
        return False

    term_cfg.noise = noise_cfg
    return True


def _fixed_midpoint_range(bounds: tuple[float, float]) -> tuple[float, float]:
    lo = float(bounds[0])
    hi = float(bounds[1])
    mid = 0.5 * (lo + hi)
    return (mid, mid)


def apply_paper_b_runtime_overrides(
    env_cfg: object,
    *,
    critical_governor_enable: bool | None = None,
    realobs_obs_ablation: str = "none",
    realobs_sensor_preset: str = "full",
) -> dict[str, Any]:
    """Apply Paper-B-specific runtime overrides to a parsed env cfg."""
    obs_ablation = str(realobs_obs_ablation).strip().lower()
    sensor_preset = str(realobs_sensor_preset).strip().lower()

    if obs_ablation not in _REALOBS_OBS_ABLATIONS:
        raise ValueError(
            f"Invalid --realobs_obs_ablation={realobs_obs_ablation!r} "
            f"(expected one of {tuple(_REALOBS_OBS_ABLATIONS.keys())})."
        )
    if sensor_preset not in _REALOBS_SENSOR_PRESETS:
        raise ValueError(
            f"Invalid --realobs_sensor_preset={realobs_sensor_preset!r} "
            f"(expected one of {_REALOBS_SENSOR_PRESETS})."
        )

    if critical_governor_enable is not None and hasattr(env_cfg, "critical_governor_enable"):
        setattr(env_cfg, "critical_governor_enable", bool(critical_governor_enable))

    if obs_ablation != "none":
        disabled_any = False
        for term_name in _REALOBS_OBS_ABLATIONS[obs_ablation]:
            disabled_any = _disable_obs_term(env_cfg, term_name) or disabled_any
        if not disabled_any:
            raise ValueError(
                f"Observation ablation '{obs_ablation}' requested, but matching RealObs observation terms "
                "were not found in the active task cfg."
            )

    if sensor_preset != "full":
        if sensor_preset in ("ideal", "voltage_only"):
            _set_policy_noise(env_cfg, "base_ang_vel", None)
            _set_policy_noise(env_cfg, "projected_gravity", None)
        if sensor_preset in ("ideal", "voltage_only", "encoder_transport"):
            _set_policy_noise(env_cfg, "thermal_stress", None)

        if sensor_preset == "ideal":
            _set_if_present(env_cfg, "voltage_sensor_bias_range_v", (0.0, 0.0))
            _set_if_present(env_cfg, "encoder_pos_noise_std_rad", 0.0)
            _set_if_present(env_cfg, "encoder_vel_noise_std_rads", 0.0)
            _set_if_present(env_cfg, "friction_bias_range", (1.0, 1.0))
            _set_if_present(env_cfg, "imu_gyro_drift_sensitivity", 0.0)
            _set_if_present(env_cfg, "imu_accel_drift_sensitivity", 0.0)
            _set_if_present(env_cfg, "cmd_transport_dr_enable", False)
            _set_if_present(env_cfg, "cmd_delay_max_steps", 0)
            _set_if_present(env_cfg, "cmd_dropout_prob", 0.0)
            _set_if_present(env_cfg, "encoder_sample_hold_prob", 0.0)
            _set_if_present(env_cfg, "case_temp_quant_step_c", 0.0)
            _set_if_present(env_cfg, "battery_voltage_quant_step_v", 0.0)
            _set_if_present(env_cfg, "cell_voltage_quant_step_v", 0.0)
            _set_if_present(env_cfg, "cell_ocv_bias_range_v", (0.0, 0.0))
            if hasattr(env_cfg, "cell_ir_range_ohm"):
                env_cfg.cell_ir_range_ohm = _fixed_midpoint_range(tuple(env_cfg.cell_ir_range_ohm))
            _set_if_present(env_cfg, "cell_sensor_bias_range_v", (0.0, 0.0))
        elif sensor_preset == "voltage_only":
            _set_if_present(env_cfg, "encoder_pos_noise_std_rad", 0.0)
            _set_if_present(env_cfg, "encoder_vel_noise_std_rads", 0.0)
            _set_if_present(env_cfg, "friction_bias_range", (1.0, 1.0))
            _set_if_present(env_cfg, "imu_gyro_drift_sensitivity", 0.0)
            _set_if_present(env_cfg, "imu_accel_drift_sensitivity", 0.0)
            _set_if_present(env_cfg, "cmd_transport_dr_enable", False)
            _set_if_present(env_cfg, "cmd_delay_max_steps", 0)
            _set_if_present(env_cfg, "cmd_dropout_prob", 0.0)
            _set_if_present(env_cfg, "encoder_sample_hold_prob", 0.0)
            _set_if_present(env_cfg, "case_temp_quant_step_c", 0.0)
        elif sensor_preset == "encoder_transport":
            _set_if_present(env_cfg, "voltage_sensor_bias_range_v", (0.0, 0.0))
            _set_if_present(env_cfg, "friction_bias_range", (1.0, 1.0))
            _set_if_present(env_cfg, "case_temp_quant_step_c", 0.0)
            _set_if_present(env_cfg, "battery_voltage_quant_step_v", 0.0)
            _set_if_present(env_cfg, "cell_voltage_quant_step_v", 0.0)
            _set_if_present(env_cfg, "cell_ocv_bias_range_v", (0.0, 0.0))
            if hasattr(env_cfg, "cell_ir_range_ohm"):
                env_cfg.cell_ir_range_ohm = _fixed_midpoint_range(tuple(env_cfg.cell_ir_range_ohm))
            _set_if_present(env_cfg, "cell_sensor_bias_range_v", (0.0, 0.0))

    setattr(env_cfg, "paper_b_obs_ablation", obs_ablation)
    setattr(env_cfg, "paper_b_sensor_preset", sensor_preset)

    return {
        "paper_b_obs_ablation": obs_ablation,
        "paper_b_sensor_preset": sensor_preset,
        "critical_governor_enable": getattr(env_cfg, "critical_governor_enable", None),
    }
