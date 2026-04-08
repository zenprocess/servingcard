"""Shared fixtures for servingcard tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REAL_EAGLE3_CONFIG = (
    Path(__file__).resolve().parents[3] / "registry" / "qwen3-coder" / "gb10-fp8-eagle3-spec3.yaml"
)


MINIMAL_CARD_DICT: dict = {
    "servingcard": "0.1",
    "model": "test-model",
    "variant": "test-variant",
    "hardware": "nvidia-a100",
    "framework": "vllm>=0.8.0",
    "author": "test-author",
    "created": "2025-01-01",
    "method": "manual",
}


@pytest.fixture()
def minimal_card_dict() -> dict:
    """Return a minimal valid card as a plain dict."""
    return MINIMAL_CARD_DICT.copy()


@pytest.fixture()
def real_eagle3_config_path() -> Path:
    """Return the path to the real Eagle3 servingcard YAML in the registry."""
    return REAL_EAGLE3_CONFIG


@pytest.fixture()
def tmp_valid_yaml(tmp_path: Path, minimal_card_dict: dict) -> Path:
    """Write a minimal valid servingcard YAML to tmp_path and return its path."""
    card_dict = minimal_card_dict.copy()
    card_dict["benchmark"] = {
        "single_stream": {"tok_s": 50.0, "ttft_ms": 200.0},
    }
    card_dict["capacity"] = {
        "context_limit": 131072,
        "gpu_memory_utilization": 0.9,
    }
    p = tmp_path / "valid-card.yaml"
    p.write_text(yaml.dump(card_dict, sort_keys=False))
    return p


@pytest.fixture()
def full_card_dict(minimal_card_dict: dict) -> dict:
    """Return a card dict with all optional sections populated."""
    d = minimal_card_dict.copy()
    d.update(
        {
            "description": "Full test card",
            "hardware_details": {
                "gpu": "nvidia-a100",
                "memory_gb": 80,
                "memory_type": "hbm2e",
                "architecture": "ampere",
            },
            "quantization": {"method": "fp8", "bits": 8},
            "speculative_decoding": {
                "method": "eagle3",
                "draft_tokens": 3,
                "draft_model": "draft-model",
                "acceptance_rate_healthy": 0.65,
                "acceptance_rate_alert_below": 0.50,
            },
            "benchmark": {
                "single_stream": {"tok_s": 69, "ttft_ms": 1541},
                "parallel": {"peak_tok_s": 469, "peak_concurrency": 8},
                "latency": {"ttft_ms": 1541},
            },
            "pawbench": {
                "suite": "full",
                "single_stream_tok_s": 69.0,
                "parallel_peak_tok_s": 469.0,
                "peak_concurrency": 8,
                "ttft_ms": 1541.0,
                "quality_score": 0.85,
                "cacp_compliance": 0.95,
                "useful_token_ratio": 0.7,
                "tokens_per_turn": 150.0,
                "adaptability_score": 0.8,
            },
            "capacity": {
                "context_limit": 131072,
                "context_soft_limit": 110000,
                "max_concurrent": 8,
                "gpu_memory_utilization": 0.90,
            },
            "serving": {
                "engine_args": {
                    "model": "test-model",
                    "quantization": "fp8",
                    "speculative_model": "draft-model",
                    "num_speculative_tokens": 3,
                    "gpu_memory_utilization": 0.90,
                    "max_model_len": 131072,
                },
                "sampling_defaults": {"temperature": 0.2, "top_p": 1.0},
                "sampling_notes": ["note1"],
            },
            "prerequisites": {
                "models": [{"path": "~/models/draft", "description": "Draft head"}],
                "serve_script": "~/bin/serve.sh",
            },
            "readiness": {
                "health_endpoint": "/health",
                "warmup_requests": 3,
                "warmup_prompt": "Say ok.",
                "warmup_max_tokens": 5,
            },
            "notes": ["Note 1", "Note 2"],
        }
    )
    return d
