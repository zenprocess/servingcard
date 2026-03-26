"""Pydantic models for the servingcard YAML format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BenchmarkEntry(BaseModel):
    """A single benchmark measurement."""

    model_config = ConfigDict(extra="allow")

    tok_s: float | None = None
    ttft_ms: float | None = None
    p99_latency_ms: float | None = None
    peak_tok_s: float | None = None
    peak_concurrency: int | None = None
    context: str | None = None


class BenchmarkSection(BaseModel):
    """Benchmark results across different scenarios."""

    single_stream: BenchmarkEntry | None = None
    parallel: BenchmarkEntry | None = None
    latency: BenchmarkEntry | None = None


class PawBenchResults(BaseModel):
    """PawBench benchmark results."""

    suite: str = "full"  # quick | standard | full
    single_stream_tok_s: float
    parallel_peak_tok_s: float | None = None
    peak_concurrency: int | None = None
    ttft_ms: float
    quality_score: float = Field(ge=0, le=1)
    cacp_compliance: float = Field(ge=0, le=1)
    useful_token_ratio: float | None = Field(default=None, ge=0, le=1)
    tokens_per_turn: float | None = None
    adaptability_score: float | None = Field(default=None, ge=0, le=1)


class HardwareDetails(BaseModel):
    """Hardware specification."""

    gpu: str
    memory_gb: int
    memory_type: str | None = None
    architecture: str | None = None


class QuantizationSection(BaseModel):
    """Quantization configuration."""

    method: str
    bits: int


class SpeculativeDecodingSection(BaseModel):
    """Speculative decoding configuration."""

    method: str
    draft_tokens: int
    draft_model: str | None = None
    acceptance_rate_healthy: float | None = None
    acceptance_rate_alert_below: float | None = None


class CapacitySection(BaseModel):
    """Capacity constraints."""

    context_limit: int
    context_soft_limit: int | None = None
    max_concurrent: int | None = None
    gpu_memory_utilization: float | None = None


class ServingSection(BaseModel):
    """Serving engine configuration."""

    engine_args: dict[str, Any] | None = None
    sampling_defaults: dict[str, Any] | None = None
    sampling_notes: list[str] | None = None


class PrerequisiteModel(BaseModel):
    """A prerequisite model dependency."""

    path: str
    description: str | None = None


class PrerequisitesSection(BaseModel):
    """Prerequisites for running this config."""

    models: list[PrerequisiteModel] | None = None
    serve_script: str | None = None


class ReadinessSection(BaseModel):
    """Readiness and health check configuration."""

    health_endpoint: str | None = None
    warmup_requests: int = 0
    warmup_prompt: str | None = None
    warmup_max_tokens: int | None = None


class ServingCard(BaseModel):
    """A complete servingcard definition."""

    servingcard: str = Field(description="Schema version")
    model: str = Field(description="Model identifier")
    variant: str = Field(description="Configuration variant name")
    hardware: str = Field(description="Target hardware identifier")
    framework: str = Field(description="Serving framework and version constraint")
    author: str = Field(description="Card author")
    created: str = Field(description="Creation date (YYYY-MM-DD)")
    method: str = Field(description="How the config was derived")
    method_iterations: int | None = Field(
        default=None, description="Number of optimization iterations"
    )

    description: str | None = None

    hardware_details: HardwareDetails | None = None
    quantization: QuantizationSection | None = None
    speculative_decoding: SpeculativeDecodingSection | None = None
    benchmark: BenchmarkSection | None = None
    pawbench: PawBenchResults | None = None
    capacity: CapacitySection | None = None
    serving: ServingSection | None = None
    prerequisites: PrerequisitesSection | None = None
    readiness: ReadinessSection | None = None
    notes: list[str] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServingCard:
        """Load a ServingCard from a YAML file."""
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        """Serialize the card to YAML string."""
        data = self.model_dump(exclude_none=True)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
