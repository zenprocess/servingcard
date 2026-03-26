"""Tests for servingcard.backends — benchmark backend plugins."""

from __future__ import annotations

from abc import ABC

import pytest

from servingcard.backends import (
    BenchmarkBackend,
    ManualBackend,
    PawBenchBackend,
    get_backend,
)


# ---------------------------------------------------------------------------
# 1. ManualBackend.run returns expected keys (mock input)
# ---------------------------------------------------------------------------


def test_manual_backend_returns_expected_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["50.0", "200.0", "0.85", "0.95", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    backend = ManualBackend()
    result = backend.run(endpoint="http://localhost:8000", model="test-model")

    assert "single_stream_tok_s" in result
    assert "ttft_ms" in result
    assert "quality_score" in result
    assert "cacp_compliance" in result
    assert result["single_stream_tok_s"] == 50.0
    assert result["ttft_ms"] == 200.0
    assert result["quality_score"] == 0.85
    assert result["cacp_compliance"] == 0.95
    assert result["suite"] == "manual"


# ---------------------------------------------------------------------------
# 2. PawBenchBackend falls back gracefully when pawbench not installed
# ---------------------------------------------------------------------------


def test_pawbench_backend_not_available() -> None:
    assert PawBenchBackend.is_available() is False


# ---------------------------------------------------------------------------
# 3. get_backend returns ManualBackend when pawbench unavailable
# ---------------------------------------------------------------------------


def test_get_backend_returns_manual() -> None:
    backend = get_backend()
    assert isinstance(backend, ManualBackend)


# ---------------------------------------------------------------------------
# 4. Backend results include required fields
# ---------------------------------------------------------------------------


def test_manual_backend_result_has_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["100", "150", "0.9", "0.8", "200", "4"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = ManualBackend().run(endpoint="http://localhost:8000", model="m")
    required_keys = {"single_stream_tok_s", "ttft_ms", "quality_score", "cacp_compliance"}
    assert required_keys.issubset(result.keys())
    # Optional fields provided
    assert result["parallel_peak_tok_s"] == 200.0
    assert result["peak_concurrency"] == 4


# ---------------------------------------------------------------------------
# 5. ManualBackend with invalid input handled (retries)
# ---------------------------------------------------------------------------


def test_manual_backend_retries_on_bad_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # First provide "abc" (invalid), then a valid number, repeat for all prompts
    inputs = iter(["abc", "50.0", "200.0", "0.85", "0.95", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = ManualBackend().run(endpoint="http://localhost:8000", model="m")
    assert result["single_stream_tok_s"] == 50.0


# ---------------------------------------------------------------------------
# 6. PawBenchBackend.is_available() returns False without pawbench
# ---------------------------------------------------------------------------


def test_pawbench_is_available_false() -> None:
    # Pawbench is not installed in this environment
    assert PawBenchBackend.is_available() is False


# ---------------------------------------------------------------------------
# 7. Backend interface check
# ---------------------------------------------------------------------------


def test_backend_is_abstract() -> None:
    assert issubclass(BenchmarkBackend, ABC)
    assert hasattr(BenchmarkBackend, "run")
    # Concrete classes implement the interface
    assert issubclass(ManualBackend, BenchmarkBackend)
    assert issubclass(PawBenchBackend, BenchmarkBackend)


# ---------------------------------------------------------------------------
# 8. Results have correct types
# ---------------------------------------------------------------------------


def test_manual_backend_result_types(monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = iter(["42.5", "180.3", "0.7", "0.6", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = ManualBackend().run(endpoint="http://localhost:8000", model="m")
    assert isinstance(result["single_stream_tok_s"], float)
    assert isinstance(result["ttft_ms"], float)
    assert isinstance(result["quality_score"], float)
    assert isinstance(result["cacp_compliance"], float)
    assert isinstance(result["suite"], str)


# ---------------------------------------------------------------------------
# 9. ManualBackend rejects out-of-range quality score (retries)
# ---------------------------------------------------------------------------


def test_manual_backend_rejects_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # quality_score: first 1.5 (rejected), then 0.5 (accepted)
    inputs = iter(["50.0", "200.0", "1.5", "0.5", "0.9", "", ""])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(inputs))

    result = ManualBackend().run(endpoint="http://localhost:8000", model="m")
    assert result["quality_score"] == 0.5


# ---------------------------------------------------------------------------
# 10. PawBenchBackend.run raises without pawbench
# ---------------------------------------------------------------------------


def test_pawbench_run_raises_without_install() -> None:
    backend = PawBenchBackend()
    with pytest.raises(RuntimeError, match="PawBench not found"):
        backend.run(endpoint="http://localhost:8000", model="m")
