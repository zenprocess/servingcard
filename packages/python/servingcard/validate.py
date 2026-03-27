"""Validation for servingcard YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from servingcard.schema import ServingCard

REQUIRED_FIELDS = [
    "servingcard",
    "model",
    "variant",
    "hardware",
    "framework",
    "author",
    "created",
    "method",
]


def validate_card(path: Path) -> list[str]:
    """Validate a servingcard YAML file.

    Returns a list of errors. Empty list means the card is valid.
    """
    errors: list[str] = []
    path = Path(path)

    if not path.exists():
        return [f"File not found: {path}"]

    if not path.suffix in (".yaml", ".yml"):
        errors.append(f"Expected .yaml or .yml extension, got: {path.suffix}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return [f"Invalid YAML: {e}"]

    if not isinstance(data, dict):
        return ["YAML root must be a mapping"]

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    # Validate against Pydantic schema
    try:
        card = ServingCard.model_validate(data)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return errors

    # Semantic validations
    if card.benchmark is None and card.benchmarks is None:
        errors.append("Missing benchmark section — at least one benchmark entry is recommended")

    # Validate benchmarks[] entries
    if card.benchmarks:
        _METRIC_FIELDS = {
            "tok_s", "peak_tok_s", "peak_concurrency", "ttft_ms",
            "quality_score", "cacp_compliance", "tool_call_accuracy",
            "useful_token_ratio", "steering_success_rate",
        }
        for i, obs in enumerate(card.benchmarks):
            if not obs.author:
                errors.append(f"benchmarks[{i}]: missing required field 'author'")
            if not obs.date:
                errors.append(f"benchmarks[{i}]: missing required field 'date'")
            has_metric = any(getattr(obs, f, None) is not None for f in _METRIC_FIELDS)
            if not has_metric:
                errors.append(f"benchmarks[{i}]: must have at least one metric (tok_s, ttft_ms, quality_score, etc.)")

    if card.capacity is None:
        errors.append("Missing capacity section")

    if (
        card.capacity
        and card.capacity.gpu_memory_utilization is not None
        and not (0.0 < card.capacity.gpu_memory_utilization <= 1.0)
    ):
        errors.append(
            f"gpu_memory_utilization must be in (0, 1], got {card.capacity.gpu_memory_utilization}"
        )

    if card.speculative_decoding and not card.benchmark and not card.benchmarks:
        errors.append("Speculative decoding config present but no benchmarks to validate it")

    return errors
