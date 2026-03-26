"""Tests for servingcard.schema — Pydantic models and YAML I/O."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from servingcard.schema import (
    BenchmarkEntry,
    BenchmarkSection,
    CapacitySection,
    PawBenchResults,
    ServingCard,
    ServingSection,
    SpeculativeDecodingSection,
)


# ---------------------------------------------------------------------------
# 1. from_yaml loads real config
# ---------------------------------------------------------------------------


def test_from_yaml_loads_real_config(real_eagle3_config_path: Path) -> None:
    if not real_eagle3_config_path.exists():
        pytest.skip("Real Eagle3 config not found in registry")
    card = ServingCard.from_yaml(real_eagle3_config_path)
    assert card.model == "qwen3-coder"
    assert card.variant == "fp8-eagle3-spec3"
    assert card.hardware == "nvidia-gb10"
    assert card.author == "zenprocess"


# ---------------------------------------------------------------------------
# 2. to_yaml round-trips correctly
# ---------------------------------------------------------------------------


def test_to_yaml_round_trip(tmp_valid_yaml: Path) -> None:
    card = ServingCard.from_yaml(tmp_valid_yaml)
    yaml_str = card.to_yaml()
    reloaded = ServingCard.model_validate(yaml.safe_load(yaml_str))
    assert reloaded.model == card.model
    assert reloaded.variant == card.variant
    assert reloaded.framework == card.framework


# ---------------------------------------------------------------------------
# 3. Required fields missing raises ValidationError
# ---------------------------------------------------------------------------


def test_required_fields_missing_raises() -> None:
    with pytest.raises(ValidationError):
        ServingCard(model="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 4. PawBenchResults validates score bounds (0-1)
# ---------------------------------------------------------------------------


def test_pawbench_valid_scores() -> None:
    pb = PawBenchResults(
        single_stream_tok_s=50.0,
        ttft_ms=200.0,
        quality_score=0.85,
        cacp_compliance=0.95,
    )
    assert pb.quality_score == 0.85
    assert pb.cacp_compliance == 0.95


# ---------------------------------------------------------------------------
# 5. PawBenchResults rejects quality_score > 1
# ---------------------------------------------------------------------------


def test_pawbench_rejects_quality_above_one() -> None:
    with pytest.raises(ValidationError, match="quality_score"):
        PawBenchResults(
            single_stream_tok_s=50.0,
            ttft_ms=200.0,
            quality_score=1.5,
            cacp_compliance=0.9,
        )


# ---------------------------------------------------------------------------
# 6. PawBenchResults rejects negative scores
# ---------------------------------------------------------------------------


def test_pawbench_rejects_negative_quality() -> None:
    with pytest.raises(ValidationError, match="quality_score"):
        PawBenchResults(
            single_stream_tok_s=50.0,
            ttft_ms=200.0,
            quality_score=-0.1,
            cacp_compliance=0.9,
        )


def test_pawbench_rejects_negative_cacp() -> None:
    with pytest.raises(ValidationError, match="cacp_compliance"):
        PawBenchResults(
            single_stream_tok_s=50.0,
            ttft_ms=200.0,
            quality_score=0.5,
            cacp_compliance=-0.01,
        )


# ---------------------------------------------------------------------------
# 7. BenchmarkEntry with all fields
# ---------------------------------------------------------------------------


def test_benchmark_entry_all_fields() -> None:
    entry = BenchmarkEntry(
        tok_s=69.0,
        ttft_ms=1541.0,
        p99_latency_ms=2000.0,
        peak_tok_s=469.0,
        peak_concurrency=8,
        context="Full load test",
    )
    assert entry.tok_s == 69.0
    assert entry.peak_concurrency == 8
    assert entry.context == "Full load test"


# ---------------------------------------------------------------------------
# 8. BenchmarkEntry with minimal fields
# ---------------------------------------------------------------------------


def test_benchmark_entry_minimal() -> None:
    entry = BenchmarkEntry()
    assert entry.tok_s is None
    assert entry.ttft_ms is None
    assert entry.context is None


# ---------------------------------------------------------------------------
# 9. CapacitySection defaults
# ---------------------------------------------------------------------------


def test_capacity_defaults() -> None:
    cap = CapacitySection(context_limit=131072)
    assert cap.context_limit == 131072
    assert cap.context_soft_limit is None
    assert cap.max_concurrent is None
    assert cap.gpu_memory_utilization is None


# ---------------------------------------------------------------------------
# 10. ServingSection with engine_args
# ---------------------------------------------------------------------------


def test_serving_section_engine_args() -> None:
    serving = ServingSection(
        engine_args={"model": "my-model", "quantization": "fp8"},
        sampling_defaults={"temperature": 0.2},
    )
    assert serving.engine_args["model"] == "my-model"
    assert serving.sampling_defaults["temperature"] == 0.2


# ---------------------------------------------------------------------------
# 11. SpeculativeDecodingSection
# ---------------------------------------------------------------------------


def test_speculative_decoding_section() -> None:
    spec = SpeculativeDecodingSection(
        method="eagle3",
        draft_tokens=3,
        draft_model="aurora-spec",
        acceptance_rate_healthy=0.65,
        acceptance_rate_alert_below=0.50,
    )
    assert spec.method == "eagle3"
    assert spec.draft_tokens == 3
    assert spec.draft_model == "aurora-spec"


# ---------------------------------------------------------------------------
# 12. from_yaml with nonexistent file raises
# ---------------------------------------------------------------------------


def test_from_yaml_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ServingCard.from_yaml("/nonexistent/path.yaml")


# ---------------------------------------------------------------------------
# 13. from_yaml with invalid YAML raises
# ---------------------------------------------------------------------------


def test_from_yaml_invalid_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("{{{{invalid: yaml: [")
    with pytest.raises(Exception):
        ServingCard.from_yaml(bad)


# ---------------------------------------------------------------------------
# 14. Extra fields accepted (model_config extra=allow on BenchmarkEntry)
# ---------------------------------------------------------------------------


def test_benchmark_entry_extra_fields() -> None:
    entry = BenchmarkEntry(tok_s=50.0, custom_metric=99.9)  # type: ignore[call-arg]
    assert entry.model_extra["custom_metric"] == 99.9  # type: ignore[index]


# ---------------------------------------------------------------------------
# 15. ServingCard with PawBenchResults
# ---------------------------------------------------------------------------


def test_serving_card_with_pawbench(full_card_dict: dict) -> None:
    card = ServingCard.model_validate(full_card_dict)
    assert card.pawbench is not None
    assert card.pawbench.quality_score == 0.85
    assert card.pawbench.cacp_compliance == 0.95
    assert card.pawbench.suite == "full"


# ---------------------------------------------------------------------------
# 16. PawBench boundary values
# ---------------------------------------------------------------------------


def test_pawbench_boundary_zero() -> None:
    pb = PawBenchResults(
        single_stream_tok_s=1.0,
        ttft_ms=1.0,
        quality_score=0.0,
        cacp_compliance=0.0,
    )
    assert pb.quality_score == 0.0


def test_pawbench_boundary_one() -> None:
    pb = PawBenchResults(
        single_stream_tok_s=1.0,
        ttft_ms=1.0,
        quality_score=1.0,
        cacp_compliance=1.0,
    )
    assert pb.quality_score == 1.0


# ---------------------------------------------------------------------------
# 17. ServingCard from full dict
# ---------------------------------------------------------------------------


def test_serving_card_full(full_card_dict: dict) -> None:
    card = ServingCard.model_validate(full_card_dict)
    assert card.quantization is not None
    assert card.quantization.method == "fp8"
    assert card.speculative_decoding is not None
    assert card.speculative_decoding.draft_tokens == 3
    assert card.notes == ["Note 1", "Note 2"]
    assert card.readiness is not None
    assert card.readiness.warmup_requests == 3
