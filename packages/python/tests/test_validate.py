"""Tests for servingcard.validate — card validation logic."""

from __future__ import annotations

from pathlib import Path

import yaml

from servingcard.validate import validate_card

# ---------------------------------------------------------------------------
# 1. Valid config returns empty errors list
# ---------------------------------------------------------------------------


def test_valid_config_no_errors(tmp_valid_yaml: Path) -> None:
    errors = validate_card(tmp_valid_yaml)
    assert errors == []


# ---------------------------------------------------------------------------
# 2. Missing model field -> error
# ---------------------------------------------------------------------------


def test_missing_model_field(tmp_path: Path, minimal_card_dict: dict) -> None:
    d = minimal_card_dict.copy()
    del d["model"]
    p = tmp_path / "card.yaml"
    p.write_text(yaml.dump(d))
    errors = validate_card(p)
    assert any("model" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 3. Missing benchmark -> error (semantic warning)
# ---------------------------------------------------------------------------


def test_missing_benchmark_warns(tmp_path: Path, minimal_card_dict: dict) -> None:
    p = tmp_path / "card.yaml"
    # Add capacity to avoid that warning too
    d = minimal_card_dict.copy()
    d["capacity"] = {"context_limit": 131072}
    p.write_text(yaml.dump(d))
    errors = validate_card(p)
    assert any("benchmark" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 4. Missing author -> error
# ---------------------------------------------------------------------------


def test_missing_author(tmp_path: Path, minimal_card_dict: dict) -> None:
    d = minimal_card_dict.copy()
    del d["author"]
    p = tmp_path / "card.yaml"
    p.write_text(yaml.dump(d))
    errors = validate_card(p)
    assert any("author" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 5. Invalid gpu_memory_utilization (>1.0) -> error
# ---------------------------------------------------------------------------


def test_invalid_gpu_memory_utilization(tmp_path: Path, minimal_card_dict: dict) -> None:
    d = minimal_card_dict.copy()
    d["benchmark"] = {"single_stream": {"tok_s": 50}}
    d["capacity"] = {"context_limit": 131072, "gpu_memory_utilization": 1.5}
    p = tmp_path / "card.yaml"
    p.write_text(yaml.dump(d))
    errors = validate_card(p)
    assert any("gpu_memory_utilization" in e for e in errors)


# ---------------------------------------------------------------------------
# 6. Valid config with all optional fields
# ---------------------------------------------------------------------------


def test_valid_full_config(tmp_path: Path, full_card_dict: dict) -> None:
    p = tmp_path / "full.yaml"
    p.write_text(yaml.dump(full_card_dict, sort_keys=False))
    errors = validate_card(p)
    assert errors == []


# ---------------------------------------------------------------------------
# 7. Nonexistent file -> error
# ---------------------------------------------------------------------------


def test_nonexistent_file() -> None:
    errors = validate_card(Path("/nonexistent/file.yaml"))
    assert len(errors) == 1
    assert "not found" in errors[0].lower()


# ---------------------------------------------------------------------------
# 8. Invalid YAML syntax -> error
# ---------------------------------------------------------------------------


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("{{{{ not valid yaml: [")
    errors = validate_card(p)
    assert len(errors) >= 1
    assert any("yaml" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 9. Empty file -> error
# ---------------------------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    errors = validate_card(p)
    assert len(errors) >= 1


# ---------------------------------------------------------------------------
# 10. Config with only required fields passes (with semantic warnings)
# ---------------------------------------------------------------------------


def test_required_only_has_semantic_warnings(tmp_path: Path, minimal_card_dict: dict) -> None:
    p = tmp_path / "minimal.yaml"
    p.write_text(yaml.dump(minimal_card_dict))
    errors = validate_card(p)
    # Should have semantic warnings (benchmark, capacity) but no hard errors
    assert all("required" not in e.lower() for e in errors)
    assert any("benchmark" in e.lower() for e in errors)
    assert any("capacity" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# 11. gpu_memory_utilization zero -> error
# ---------------------------------------------------------------------------


def test_gpu_memory_utilization_zero(tmp_path: Path, minimal_card_dict: dict) -> None:
    d = minimal_card_dict.copy()
    d["benchmark"] = {"single_stream": {"tok_s": 50}}
    d["capacity"] = {"context_limit": 131072, "gpu_memory_utilization": 0.0}
    p = tmp_path / "card.yaml"
    p.write_text(yaml.dump(d))
    errors = validate_card(p)
    assert any("gpu_memory_utilization" in e for e in errors)


# ---------------------------------------------------------------------------
# 12. Speculative decoding without benchmark -> error
# ---------------------------------------------------------------------------


def test_speculative_without_benchmark(tmp_path: Path, minimal_card_dict: dict) -> None:
    d = minimal_card_dict.copy()
    d["speculative_decoding"] = {"method": "eagle3", "draft_tokens": 3}
    d["capacity"] = {"context_limit": 131072}
    p = tmp_path / "card.yaml"
    p.write_text(yaml.dump(d))
    errors = validate_card(p)
    assert any("speculative" in e.lower() for e in errors)
