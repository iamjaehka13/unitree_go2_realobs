from __future__ import annotations

PAPER_SCENARIO_LABELS = {
    "fresh": "nominal",
    "used": "moderate",
    "aged": "severe",
    "critical": "critical",
}

SCENARIO_CLI_ALIASES = {
    "fresh": "fresh",
    "used": "used",
    "aged": "aged",
    "critical": "critical",
    "nominal": "fresh",
    "moderate": "used",
    "severe": "aged",
    "safety_critical": "critical",
}

SCENARIO_CLI_CHOICES = sorted(
    set(["fresh", "used", "aged", "critical", "nominal", "moderate", "severe", "safety_critical"])
)
SCENARIO_CLI_CHOICES_WITH_NONE = sorted(set(["none", *SCENARIO_CLI_CHOICES]))


def scenario_key(name: str) -> str:
    key = SCENARIO_CLI_ALIASES.get(str(name).strip().lower())
    if key is None:
        valid = ", ".join(sorted(SCENARIO_CLI_ALIASES.keys()))
        raise ValueError(f"Unknown scenario name '{name}'. Expected one of: {valid}")
    return key


def scenario_label(name: str) -> str:
    return PAPER_SCENARIO_LABELS.get(scenario_key(name), str(name).strip().lower())


def scenario_labels(names: list[str]) -> list[str]:
    return [scenario_label(name) for name in names]


def scenario_lookup_keys(name: str) -> tuple[str, ...]:
    key = scenario_key(name)
    label = scenario_label(key)
    if label == key:
        return (key,)
    return (label, key)
