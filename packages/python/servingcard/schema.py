"""Pydantic models for the servingcard YAML format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BenchmarkObservation(BaseModel):
    """A single benchmark observation — one author's results for this config."""

    author: str = Field(description="GitHub username of the benchmarker")
    status: str = Field(default="verified", description="verified | claim | independent")
    date: str = Field(description="Benchmark date YYYY-MM-DD")
    method: str = Field(default="pawbench", description="Benchmark tool used")
    source: str | None = Field(default=None, description="URL to source (for claims)")
    source_label: str | None = None

    # Results
    tok_s: float | None = None
    peak_tok_s: float | None = None
    peak_concurrency: int | None = None
    ttft_ms: float | None = None
    quality_score: float | None = None
    cacp_compliance: float | None = None
    tool_call_accuracy: float | None = None
    useful_token_ratio: float | None = None
    steering_success_rate: float | None = None

    notes: str | None = None


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


_QUANT_BITS: dict[str, int] = {
    "fp4": 4,
    "nvfp4": 4,
    "int4": 4,
    "fp8": 8,
    "int8": 8,
    "fp16": 16,
    "bf16": 16,
    "fp32": 32,
}


class QuantizationSection(BaseModel):
    """Quantization configuration.

    Accepts either a structured form (`{method: fp8, bits: 8}`) or a bare
    string shorthand (`fp8`) coerced via `from_shorthand`. Bit width is
    inferred from the well-known method name; unknown shorthands default
    to 0 bits with a warning rather than failing the load.
    """

    method: str
    bits: int = 0

    @classmethod
    def from_shorthand(cls, value: str) -> "QuantizationSection":
        return cls(method=value, bits=_QUANT_BITS.get(value.lower(), 0))


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


class HuggingFaceSection(BaseModel):
    """HuggingFace model references."""

    base_model: str | None = None
    quantized_model: str | None = None
    base_url: str | None = None
    quantized_url: str | None = None


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

    # Verification status: verified (PawBench), claim (published numbers), independent (re-run)
    status: str = Field(default="verified", description="verified | claim | independent")
    source: str | None = Field(default=None, description="URL to original benchmark source (for claims)")
    source_label: str | None = Field(default=None, description="Human-readable source description")

    model_type: str | None = Field(default=None, description="Model type: dense-general, coding-specialist, etc.")
    description: str | None = None

    hardware_details: HardwareDetails | None = None
    quantization: QuantizationSection | None = None

    @field_validator("quantization", mode="before")
    @classmethod
    def _coerce_quantization(cls, v: object) -> object:
        if isinstance(v, str):
            return QuantizationSection.from_shorthand(v)
        return v
    speculative_decoding: SpeculativeDecodingSection | None = None
    benchmark: BenchmarkSection | None = None
    benchmarks: list[BenchmarkObservation] | None = None
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
