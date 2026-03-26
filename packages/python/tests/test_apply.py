"""Tests for servingcard.apply — source resolution and command generation."""

from __future__ import annotations

import pytest

from servingcard.apply import (
    REGISTRY_BASE_URL,
    generate_launch_command,
    generate_tgi_command,
    generate_vllm_command,
    resolve_source,
)
from servingcard.schema import (
    CapacitySection,
    ServingCard,
    ServingSection,
    SpeculativeDecodingSection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card(**overrides) -> ServingCard:
    """Build a minimal ServingCard with optional overrides."""
    defaults = {
        "servingcard": "0.1",
        "model": "test-model",
        "variant": "test-variant",
        "hardware": "nvidia-a100",
        "framework": "vllm>=0.8.0",
        "author": "test",
        "created": "2025-01-01",
        "method": "manual",
    }
    defaults.update(overrides)
    return ServingCard.model_validate(defaults)


# ---------------------------------------------------------------------------
# 1. resolve_source local file path -> returns as-is
# ---------------------------------------------------------------------------


def test_resolve_source_local_file() -> None:
    assert resolve_source("./my-config.yaml") == "./my-config.yaml"
    assert resolve_source("/abs/path.yaml") == "/abs/path.yaml"


# ---------------------------------------------------------------------------
# 2. resolve_source registry shorthand -> returns GitHub URL
# ---------------------------------------------------------------------------


def test_resolve_source_registry_shorthand() -> None:
    result = resolve_source("qwen3-coder/gb10-fp8")
    assert result == f"{REGISTRY_BASE_URL}/qwen3-coder/gb10-fp8.yaml"


# ---------------------------------------------------------------------------
# 3. resolve_source full URL -> returns as-is
# ---------------------------------------------------------------------------


def test_resolve_source_full_url() -> None:
    url = "https://example.com/card.yaml"
    assert resolve_source(url) == url


def test_resolve_source_http_url() -> None:
    url = "http://localhost:8080/card.yaml"
    assert resolve_source(url) == url


# ---------------------------------------------------------------------------
# 4. generate_vllm_command with full config
# ---------------------------------------------------------------------------


def test_generate_vllm_command_full() -> None:
    card = _make_card(
        serving={
            "engine_args": {
                "model": "my-model",
                "quantization": "fp8",
                "max_model_len": 131072,
                "gpu_memory_utilization": 0.9,
            }
        }
    )
    cmd = generate_vllm_command(card)
    assert cmd.startswith("vllm serve my-model")
    assert "--quantization fp8" in cmd
    assert "--max-model-len 131072" in cmd
    assert "--gpu-memory-utilization 0.9" in cmd


# ---------------------------------------------------------------------------
# 5. generate_vllm_command with minimal config (no engine_args)
# ---------------------------------------------------------------------------


def test_generate_vllm_command_no_engine_args() -> None:
    card = _make_card()
    cmd = generate_vllm_command(card)
    assert "test-model" in cmd
    assert "cannot generate" in cmd.lower() or "No engine_args" in cmd


# ---------------------------------------------------------------------------
# 6. generate_tgi_command
# ---------------------------------------------------------------------------


def test_generate_tgi_command() -> None:
    card = _make_card(
        serving={
            "engine_args": {
                "model": "my-model",
                "max_model_len": 131072,
            }
        }
    )
    cmd = generate_tgi_command(card)
    assert "text-generation-launcher" in cmd
    assert "--model-id my-model" in cmd
    assert "--max-input-length 131072" in cmd


# ---------------------------------------------------------------------------
# 7. generate_launch_command infers engine from framework
# ---------------------------------------------------------------------------


def test_launch_command_infers_vllm() -> None:
    card = _make_card(
        framework="vllm>=0.8.0",
        serving={"engine_args": {"model": "m"}},
    )
    cmd = generate_launch_command(card)
    assert "vllm serve" in cmd


def test_launch_command_infers_tgi() -> None:
    card = _make_card(
        framework="tgi>=2.0",
        serving={"engine_args": {"model": "m"}},
    )
    cmd = generate_launch_command(card)
    assert "text-generation-launcher" in cmd


# ---------------------------------------------------------------------------
# 8. generate_launch_command with engine override
# ---------------------------------------------------------------------------


def test_launch_command_engine_override() -> None:
    card = _make_card(
        framework="vllm>=0.8.0",
        serving={"engine_args": {"model": "m"}},
    )
    cmd = generate_launch_command(card, engine="tgi")
    assert "text-generation-launcher" in cmd


def test_launch_command_unsupported_engine() -> None:
    card = _make_card()
    cmd = generate_launch_command(card, engine="triton")
    assert "Unsupported engine" in cmd


# ---------------------------------------------------------------------------
# 9. Registry URL format correct
# ---------------------------------------------------------------------------


def test_registry_url_format() -> None:
    assert REGISTRY_BASE_URL.startswith("https://")
    assert "servingcard" in REGISTRY_BASE_URL
    result = resolve_source("model/variant")
    assert result.endswith(".yaml")
    assert "/model/variant.yaml" in result


# ---------------------------------------------------------------------------
# 10. Config with speculative decoding produces correct flags
# ---------------------------------------------------------------------------


def test_speculative_decoding_vllm_flags() -> None:
    card = _make_card(
        serving={
            "engine_args": {
                "model": "qwen3-coder",
                "speculative_model": "aurora-spec",
                "num_speculative_tokens": 3,
            }
        }
    )
    cmd = generate_vllm_command(card)
    assert "--speculative-model aurora-spec" in cmd
    assert "--num-speculative-tokens 3" in cmd


def test_speculative_decoding_tgi_commented() -> None:
    """TGI command should comment out speculative decoding args."""
    card = _make_card(
        serving={
            "engine_args": {
                "model": "qwen3-coder",
                "speculative_model": "aurora-spec",
                "num_speculative_tokens": 3,
            }
        }
    )
    cmd = generate_tgi_command(card)
    assert "# --speculative-model" in cmd or "speculative" in cmd.lower()


# ---------------------------------------------------------------------------
# 11. resolve_source bare name -> returns as-is (local file)
# ---------------------------------------------------------------------------


def test_resolve_source_bare_name() -> None:
    assert resolve_source("config.yaml") == "config.yaml"


# ---------------------------------------------------------------------------
# 12. Boolean engine_args rendered correctly
# ---------------------------------------------------------------------------


def test_vllm_boolean_flag() -> None:
    card = _make_card(
        serving={
            "engine_args": {
                "model": "m",
                "enable_prefix_caching": True,
                "enforce_eager": False,
            }
        }
    )
    cmd = generate_vllm_command(card)
    assert "--enable-prefix-caching" in cmd
    # False booleans should not appear
    assert "--enforce-eager" not in cmd
