from __future__ import annotations

from typing import Any

from isaaclab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg


_LOCOMOTION_POLICY_OBS = frozenset(
    {
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "last_action",
    }
)
_LOCOMOTION_CRITIC_OBS = frozenset(
    {
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "joint_effort",
        "last_action",
    }
)
_MEASURABLE_EXTRA_OBS = frozenset({"energy_budget", "thermal_stress", "vibration_level"})
_PRIVILEGED_EXTRA_OBS = frozenset({"energy_budget", "thermal_stress", "mech_health", "vibration_level", "degradation_trend", "thermal_overload"})
_TEMPDOSE_EXTRA_OBS = frozenset({"thermal_rate", "thermal_dose"})

_LOCOMOTION_REWARDS = frozenset(
    {
        "track_lin_vel_xy",
        "track_ang_vel_z",
        "feet_air_time",
        "feet_slide",
        "undesired_contacts",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "flat_orientation",
        "joint_vel_l2",
        "joint_acc_l2",
        "joint_torques_l2",
        "action_rate",
        "dof_pos_limits",
    }
)
_MEASURABLE_PROXY_REWARDS = frozenset({"energy_efficiency", "thermal_safety", "saturation_prevention"})
_PRIVILEGED_EXTRA_REWARDS = frozenset({"bearing_health"})

_SOFT_SAFETY_TERMINATIONS = frozenset({"time_out", "base_contact", "bad_orientation", "motor_stall"})
_HARD_THERM_TERMINATIONS = frozenset({"thermal_failure"})

_SUPPORTED_OBS_ABLATIONS = frozenset({"none", "no_voltage", "no_thermal", "no_vibration"})
_OBS_ABLATION_TO_TERMS = {
    "none": frozenset(),
    "no_voltage": frozenset({"energy_budget"}),
    "no_thermal": frozenset({"thermal_stress", "thermal_rate", "thermal_dose"}),
    "no_vibration": frozenset({"vibration_level"}),
}


def _cfg_items(cfg: object) -> tuple[tuple[str, Any], ...]:
    try:
        return tuple(vars(cfg).items())
    except TypeError:
        return ()


def _active_term_names(cfg: object, term_type: type[Any]) -> tuple[str, ...]:
    names = [name for name, value in _cfg_items(cfg) if isinstance(value, term_type)]
    return tuple(sorted(names))


def _active_obs_terms(group_cfg: object) -> tuple[str, ...]:
    return _active_term_names(group_cfg, ObservationTermCfg)


def _active_reward_terms(reward_cfg: object) -> tuple[str, ...]:
    return _active_term_names(reward_cfg, RewardTermCfg)


def _active_termination_terms(termination_cfg: object) -> tuple[str, ...]:
    return _active_term_names(termination_cfg, TerminationTermCfg)


def _expected_terms(base_terms: frozenset[str], obs_ablation: str) -> frozenset[str]:
    return base_terms - _OBS_ABLATION_TO_TERMS[obs_ablation]


def summarize_paper_b_task_cfg(cfg: object) -> dict[str, Any]:
    terminations = getattr(cfg, "terminations", None)
    commands = getattr(cfg, "commands", None)
    observations = getattr(cfg, "observations", None)
    rewards = getattr(cfg, "rewards", None)
    risk_factor = getattr(commands, "risk_factor", None) if commands is not None else None
    thermal_failure = getattr(terminations, "thermal_failure", None) if terminations is not None else None
    policy_cfg = getattr(observations, "policy", None) if observations is not None else None
    critic_cfg = getattr(observations, "critic", None) if observations is not None else None

    return {
        "paper_b_family": str(getattr(cfg, "paper_b_family", "")).strip().lower(),
        "paper_b_variant": str(getattr(cfg, "paper_b_variant", "")).strip().lower(),
        "paper_b_observation_scope": str(getattr(cfg, "paper_b_observation_scope", "")).strip().lower(),
        "paper_b_reward_scope": str(getattr(cfg, "paper_b_reward_scope", "")).strip().lower(),
        "paper_b_deployable": bool(getattr(cfg, "paper_b_deployable", False)),
        "paper_b_obs_ablation": str(getattr(cfg, "paper_b_obs_ablation", "none")).strip().lower(),
        "paper_b_sensor_preset": str(getattr(cfg, "paper_b_sensor_preset", "full")).strip().lower(),
        "critical_governor_enable": bool(getattr(cfg, "critical_governor_enable", False)),
        "brownout_voltage_source": str(getattr(cfg, "brownout_voltage_source", "")).strip().lower(),
        "temperature_metric_semantics": str(getattr(cfg, "temperature_metric_semantics", "")).strip().lower(),
        "thermal_failure_enabled": thermal_failure is not None,
        "risk_factor_min": getattr(risk_factor, "minimum", None),
        "risk_factor_max": getattr(risk_factor, "maximum", None),
        "realobs_require_voltage_sensor": bool(getattr(cfg, "realobs_require_voltage_sensor", False)),
        "realobs_allow_true_voltage_fallback": bool(getattr(cfg, "realobs_allow_true_voltage_fallback", False)),
        "realobs_require_case_temperature_proxy": bool(
            getattr(cfg, "realobs_require_case_temperature_proxy", False)
        ),
        "realobs_allow_case_temperature_from_coil_fallback": bool(
            getattr(cfg, "realobs_allow_case_temperature_from_coil_fallback", False)
        ),
        "policy_observation_terms": _active_obs_terms(policy_cfg),
        "critic_observation_terms": _active_obs_terms(critic_cfg),
        "reward_terms": _active_reward_terms(rewards),
        "termination_terms": _active_termination_terms(terminations),
    }


def _expect(summary: dict[str, Any], condition: bool, detail: str) -> None:
    if condition:
        return
    raise ValueError(
        f"Paper B contract violation for variant='{summary['paper_b_variant']}': {detail}. "
        f"summary={summary}"
    )


def _validate_main_ladder(summary: dict[str, Any]) -> None:
    _expect(summary, summary["paper_b_family"] == "main_ladder", "paper_b_family must be 'main_ladder'")
    _expect(summary, summary["paper_b_deployable"], "main ladder variants must be deployable")
    _expect(summary, not summary["critical_governor_enable"], "main ladder must keep governor disabled")
    _expect(
        summary,
        summary["brownout_voltage_source"] == "sensor_voltage",
        "main ladder must use deployable sensor-voltage brownout semantics",
    )
    _expect(summary, not summary["thermal_failure_enabled"], "main ladder must disable hard thermal stop")
    _expect(
        summary,
        summary["temperature_metric_semantics"] == "case_proxy",
        "main ladder must report case-proxy temperature metrics",
    )
    if summary["risk_factor_min"] is not None and summary["risk_factor_max"] is not None:
        _expect(
            summary,
            float(summary["risk_factor_min"]) == 1.0 and float(summary["risk_factor_max"]) == 1.0,
            "main ladder must clamp risk_factor to 1.0",
        )


def _validate_realobs_contract(summary: dict[str, Any]) -> None:
    _expect(summary, summary["realobs_require_voltage_sensor"], "measurable-only variants must require voltage sensor")
    _expect(
        summary,
        not summary["realobs_allow_true_voltage_fallback"],
        "measurable-only variants must not allow hidden true-voltage fallback",
    )
    _expect(
        summary,
        summary["realobs_require_case_temperature_proxy"],
        "measurable-only variants must require explicit case proxy",
    )
    _expect(
        summary,
        not summary["realobs_allow_case_temperature_from_coil_fallback"],
        "measurable-only variants must not allow hidden coil-derived proxy fallback",
    )


def _validate_obs_ablation(summary: dict[str, Any], *, allowed: bool) -> str:
    obs_ablation = summary["paper_b_obs_ablation"]
    _expect(
        summary,
        obs_ablation in _SUPPORTED_OBS_ABLATIONS,
        f"unsupported paper_b_obs_ablation='{obs_ablation}'",
    )
    if not allowed:
        _expect(summary, obs_ablation == "none", "this variant must not apply Paper B observation ablation")
    return obs_ablation


def _validate_structure(
    summary: dict[str, Any],
    *,
    expected_policy_obs: frozenset[str],
    expected_critic_obs: frozenset[str],
    expected_rewards: frozenset[str],
    expected_terminations: frozenset[str],
) -> None:
    _expect(
        summary,
        set(summary["policy_observation_terms"]) == expected_policy_obs,
        f"policy observation terms mismatch: expected={sorted(expected_policy_obs)} actual={list(summary['policy_observation_terms'])}",
    )
    _expect(
        summary,
        set(summary["critic_observation_terms"]) == expected_critic_obs,
        f"critic observation terms mismatch: expected={sorted(expected_critic_obs)} actual={list(summary['critic_observation_terms'])}",
    )
    _expect(
        summary,
        set(summary["reward_terms"]) == expected_rewards,
        f"reward terms mismatch: expected={sorted(expected_rewards)} actual={list(summary['reward_terms'])}",
    )
    _expect(
        summary,
        set(summary["termination_terms"]) == expected_terminations,
        f"termination terms mismatch: expected={sorted(expected_terminations)} actual={list(summary['termination_terms'])}",
    )


def validate_paper_b_task_cfg(cfg: object) -> dict[str, Any]:
    summary = summarize_paper_b_task_cfg(cfg)
    variant = summary["paper_b_variant"]

    _expect(summary, bool(variant), "paper_b_variant must be set")
    _expect(summary, bool(summary["paper_b_family"]), "paper_b_family must be set")
    _expect(summary, bool(summary["paper_b_observation_scope"]), "paper_b_observation_scope must be set")
    _expect(summary, bool(summary["paper_b_reward_scope"]), "paper_b_reward_scope must be set")

    if variant == "baseline":
        _validate_obs_ablation(summary, allowed=False)
        _validate_main_ladder(summary)
        _expect(summary, summary["paper_b_observation_scope"] == "locomotion_only", "baseline observation scope mismatch")
        _expect(summary, summary["paper_b_reward_scope"] == "locomotion_only", "baseline reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_LOCOMOTION_POLICY_OBS,
            expected_critic_obs=_LOCOMOTION_CRITIC_OBS,
            expected_rewards=_LOCOMOTION_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS,
        )
    elif variant == "obsonly":
        obs_ablation = _validate_obs_ablation(summary, allowed=True)
        _validate_main_ladder(summary)
        _validate_realobs_contract(summary)
        _expect(summary, summary["paper_b_observation_scope"] == "measurable_only", "ObsOnly observation scope mismatch")
        _expect(summary, summary["paper_b_reward_scope"] == "locomotion_only", "ObsOnly reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_expected_terms(_LOCOMOTION_POLICY_OBS | _MEASURABLE_EXTRA_OBS, obs_ablation),
            expected_critic_obs=_expected_terms(_LOCOMOTION_CRITIC_OBS | _MEASURABLE_EXTRA_OBS, obs_ablation),
            expected_rewards=_LOCOMOTION_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS,
        )
    elif variant == "realobs":
        obs_ablation = _validate_obs_ablation(summary, allowed=True)
        _validate_main_ladder(summary)
        _validate_realobs_contract(summary)
        _expect(summary, summary["paper_b_observation_scope"] == "measurable_only", "RealObs observation scope mismatch")
        _expect(summary, summary["paper_b_reward_scope"] == "measurable_proxy", "RealObs reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_expected_terms(_LOCOMOTION_POLICY_OBS | _MEASURABLE_EXTRA_OBS, obs_ablation),
            expected_critic_obs=_expected_terms(_LOCOMOTION_CRITIC_OBS | _MEASURABLE_EXTRA_OBS, obs_ablation),
            expected_rewards=_LOCOMOTION_REWARDS | _MEASURABLE_PROXY_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS,
        )
    elif variant == "strategic":
        _validate_obs_ablation(summary, allowed=False)
        _expect(summary, summary["paper_b_family"] == "upper_bound", "Strategic family mismatch")
        _expect(summary, not summary["paper_b_deployable"], "Strategic must not be marked deployable")
        _expect(summary, summary["critical_governor_enable"], "Strategic must enable governor")
        _expect(summary, summary["brownout_voltage_source"] == "bms_pred", "Strategic must use privileged brownout semantics")
        _expect(summary, summary["thermal_failure_enabled"], "Strategic must keep hard thermal stop enabled")
        _expect(summary, summary["temperature_metric_semantics"] == "coil_hotspot", "Strategic metric semantics mismatch")
        _expect(summary, summary["paper_b_observation_scope"] == "privileged", "Strategic observation scope mismatch")
        _expect(summary, summary["paper_b_reward_scope"] == "privileged", "Strategic reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_LOCOMOTION_POLICY_OBS | _PRIVILEGED_EXTRA_OBS,
            expected_critic_obs=_LOCOMOTION_CRITIC_OBS | _PRIVILEGED_EXTRA_OBS,
            expected_rewards=_LOCOMOTION_REWARDS | _MEASURABLE_PROXY_REWARDS | _PRIVILEGED_EXTRA_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS | _HARD_THERM_TERMINATIONS,
        )
    elif variant == "strategic_nogov":
        _validate_obs_ablation(summary, allowed=False)
        _expect(summary, summary["paper_b_family"] == "upper_bound", "Strategic-noGov family mismatch")
        _expect(summary, not summary["paper_b_deployable"], "Strategic-noGov must not be marked deployable")
        _expect(summary, not summary["critical_governor_enable"], "Strategic-noGov must disable governor")
        _expect(summary, summary["brownout_voltage_source"] == "bms_pred", "Strategic-noGov brownout semantics mismatch")
        _expect(summary, summary["thermal_failure_enabled"], "Strategic-noGov must keep hard thermal stop enabled")
        _expect(summary, summary["temperature_metric_semantics"] == "coil_hotspot", "Strategic-noGov metric semantics mismatch")
        _expect(summary, summary["paper_b_observation_scope"] == "privileged", "Strategic-noGov observation scope mismatch")
        _expect(summary, summary["paper_b_reward_scope"] == "privileged", "Strategic-noGov reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_LOCOMOTION_POLICY_OBS | _PRIVILEGED_EXTRA_OBS,
            expected_critic_obs=_LOCOMOTION_CRITIC_OBS | _PRIVILEGED_EXTRA_OBS,
            expected_rewards=_LOCOMOTION_REWARDS | _MEASURABLE_PROXY_REWARDS | _PRIVILEGED_EXTRA_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS | _HARD_THERM_TERMINATIONS,
        )
    elif variant == "strategic_softtherm":
        _validate_obs_ablation(summary, allowed=False)
        _expect(summary, summary["paper_b_family"] == "upper_bound", "Strategic-SoftTherm family mismatch")
        _expect(summary, not summary["paper_b_deployable"], "Strategic-SoftTherm must not be marked deployable")
        _expect(summary, summary["critical_governor_enable"], "Strategic-SoftTherm must enable governor")
        _expect(
            summary,
            summary["brownout_voltage_source"] == "bms_pred",
            "Strategic-SoftTherm brownout semantics mismatch",
        )
        _expect(summary, not summary["thermal_failure_enabled"], "Strategic-SoftTherm must disable hard thermal stop")
        _expect(
            summary,
            summary["temperature_metric_semantics"] == "coil_hotspot",
            "Strategic-SoftTherm metric semantics mismatch",
        )
        _expect(summary, summary["paper_b_observation_scope"] == "privileged", "Strategic-SoftTherm observation scope mismatch")
        _expect(summary, summary["paper_b_reward_scope"] == "privileged", "Strategic-SoftTherm reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_LOCOMOTION_POLICY_OBS | _PRIVILEGED_EXTRA_OBS,
            expected_critic_obs=_LOCOMOTION_CRITIC_OBS | _PRIVILEGED_EXTRA_OBS,
            expected_rewards=_LOCOMOTION_REWARDS | _MEASURABLE_PROXY_REWARDS | _PRIVILEGED_EXTRA_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS,
        )
    elif variant == "realobs_hardtherm":
        obs_ablation = _validate_obs_ablation(summary, allowed=True)
        _expect(summary, summary["paper_b_family"] == "side_ablation", "RealObs-HardTherm family mismatch")
        _validate_realobs_contract(summary)
        _expect(summary, summary["paper_b_deployable"], "RealObs-HardTherm should keep deployable channels")
        _expect(summary, not summary["critical_governor_enable"], "RealObs-HardTherm must keep governor disabled")
        _expect(
            summary,
            summary["brownout_voltage_source"] == "sensor_voltage",
            "RealObs-HardTherm must use deployable brownout semantics",
        )
        _expect(summary, summary["thermal_failure_enabled"], "RealObs-HardTherm must enable hard thermal stop")
        _expect(
            summary,
            summary["temperature_metric_semantics"] == "coil_hotspot",
            "RealObs-HardTherm metric semantics mismatch",
        )
        _expect(
            summary,
            summary["paper_b_observation_scope"] == "measurable_only",
            "RealObs-HardTherm observation scope mismatch",
        )
        _expect(
            summary,
            summary["paper_b_reward_scope"] == "measurable_proxy",
            "RealObs-HardTherm reward scope mismatch",
        )
        _validate_structure(
            summary,
            expected_policy_obs=_expected_terms(_LOCOMOTION_POLICY_OBS | _MEASURABLE_EXTRA_OBS, obs_ablation),
            expected_critic_obs=_expected_terms(_LOCOMOTION_CRITIC_OBS | _MEASURABLE_EXTRA_OBS, obs_ablation),
            expected_rewards=_LOCOMOTION_REWARDS | _MEASURABLE_PROXY_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS | _HARD_THERM_TERMINATIONS,
        )
    elif variant == "tempdose":
        obs_ablation = _validate_obs_ablation(summary, allowed=True)
        _expect(summary, summary["paper_b_family"] == "side_ablation", "TempDose family mismatch")
        _validate_realobs_contract(summary)
        _expect(summary, summary["paper_b_deployable"], "TempDose should keep deployable measurable channels")
        _expect(summary, not summary["critical_governor_enable"], "TempDose must keep governor disabled")
        _expect(summary, summary["brownout_voltage_source"] == "sensor_voltage", "TempDose brownout semantics mismatch")
        _expect(summary, summary["thermal_failure_enabled"], "TempDose must enable hard thermal stop")
        _expect(summary, summary["temperature_metric_semantics"] == "coil_hotspot", "TempDose metric semantics mismatch")
        _expect(
            summary,
            summary["paper_b_observation_scope"] == "measurable_plus_thermal",
            "TempDose observation scope mismatch",
        )
        _expect(summary, summary["paper_b_reward_scope"] == "measurable_proxy", "TempDose reward scope mismatch")
        _validate_structure(
            summary,
            expected_policy_obs=_expected_terms(_LOCOMOTION_POLICY_OBS | _MEASURABLE_EXTRA_OBS | _TEMPDOSE_EXTRA_OBS, obs_ablation),
            expected_critic_obs=_expected_terms(_LOCOMOTION_CRITIC_OBS | _MEASURABLE_EXTRA_OBS | _TEMPDOSE_EXTRA_OBS, obs_ablation),
            expected_rewards=_LOCOMOTION_REWARDS | _MEASURABLE_PROXY_REWARDS,
            expected_terminations=_SOFT_SAFETY_TERMINATIONS | _HARD_THERM_TERMINATIONS,
        )
    else:
        raise ValueError(f"Unknown paper_b_variant='{variant}'. summary={summary}")

    return summary
