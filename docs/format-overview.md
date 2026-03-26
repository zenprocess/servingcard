# Format Overview

A serving card is a YAML file with structured sections. This guide explains
each section in detail. For the formal schema, see the
[specification](../spec/servingcard-v1.md).

## Required Fields

Every serving card must include these top-level fields:

| Field | Type | Description |
|-------|------|-------------|
| `servingcard` | string | Spec version. Must be `"1.0"` |
| `model` | string | Model identifier (HuggingFace ID or short name) |
| `variant` | string | Configuration variant name |
| `hardware` | string | Target hardware slug |
| `framework` | string | Inference framework |
| `author` | string | Who created and verified this card |
| `created` | string | ISO 8601 date |
| `method` | string | How the config was tuned |
| `benchmark` | object | Verified benchmark results (see below) |

## Identity Section

The identity fields answer "what is this config for?"

```yaml
servingcard: "1.0"
model: qwen3-coder                    # What model
variant: fp8-eagle3-spec3             # Which configuration
hardware: nvidia-gb10                 # What hardware
framework: vllm                       # What inference framework
framework_version: ">=0.8.0"          # Optional version constraint
```

**Hardware identifiers** follow the pattern `{vendor}-{chip}[-{memory}]`:
- `nvidia-rtx4090` (24 GB, no suffix needed -- only one memory config)
- `nvidia-a100-80g` (suffix needed to distinguish from A100-40G)
- `nvidia-gb10` (128 GB unified memory)
- `amd-mi300x`
- `apple-m4-ultra`

## Accountability Section

The accountability fields answer "who made this and how?"

```yaml
author: zenprocess                    # Required
created: "2026-03-26"                 # Required
method: autoresearch                  # Required
method_detail: "378 iterations"       # Optional
license: MIT                          # Optional
```

**Standard method values**: `manual`, `autoresearch`, `auto-tune-vllm`,
`upstream`, `community`. Custom values are permitted.

## Serving Section

Framework-specific parameters. Keys correspond directly to the framework's
CLI flags or config options. This section is intentionally not standardized
across frameworks.

```yaml
# vLLM example
serving:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  max_model_len: 131072
  quantization: fp8
  speculative_decoding:
    method: eagle3
    draft_model: aurora-spec-qwen3-coder
    num_speculative_tokens: 3
```

```yaml
# TGI example
serving:
  sharded: true
  num_shard: 4
  quantize: bitsandbytes-fp4
  max_input_length: 8192
```

```yaml
# llama.cpp example
serving:
  n_gpu_layers: 99
  n_ctx: 32768
  flash_attn: true
```

## Sampling Section

Documents which sampling parameters the configuration supports and their
valid ranges. Uses constraint syntax:

```yaml
sampling:
  temperature: {min: 0, max: 2, default: 0.2}    # Numeric range with default
  top_p: {min: 0, max: 1, default: 1}             # Numeric range with default
  logit_bias: unsupported                          # Not supported
  logprobs: supported                              # Supported, no constraints
```

Consumers should validate sampling parameters against these constraints
before sending requests.

## Capacity Section

Tested, verified limits for this specific configuration:

```yaml
capacity:
  context_limit: 131072       # Max context length in tokens
  max_concurrent: 8           # Max concurrent requests
  parallel_tool_calls:
    max_reliable: 3           # Max parallel tool calls that work reliably
```

These are measured limits, not theoretical maximums.

## Benchmark Section

The only required subsection: `benchmark.single_stream.tok_s`.

```yaml
benchmark:
  single_stream:
    tok_s: 69.0               # Required: output tokens/sec
    ttft_ms: 1541             # Time to first token
    p99_latency_ms: 2200      # 99th percentile latency
    input_tokens: 4096        # Input size used
    output_tokens: 512        # Output size used
  parallel:
    peak_tok_s: 469           # Aggregate tok/s at peak
    concurrency: 8            # What concurrency level
  methodology:
    tool: switchyard-bench    # What benchmark tool
    prompt_distribution: coding-tasks
    num_runs: 10
    confidence_interval: 0.95
```

See the [Benchmark Guide](benchmarks.md) for methodology details.

## Transforms Section

Documents known model output quirks and how to fix them:

```yaml
transforms:
  - type: regex_strip
    pattern: "<think>.*?</think>"
    description: "Strip reasoning tags from output"
  - type: coerce_float_to_int
    scope: tool_call_arguments
    description: "Fix float tool args (42.0 -> 42)"
```

Standard transform types: `regex_strip`, `regex_replace`,
`coerce_float_to_int`, `json_repair`, `truncate`.

## Readiness Section

Warmup and health check configuration:

```yaml
readiness:
  warmup_requests: 3         # Requests to send before production traffic
  warmup_prompt: "Say ok."
  warmup_max_tokens: 5
  health_endpoint: /health
  ready_timeout_s: 300
```

## Prerequisites Section

System requirements:

```yaml
prerequisites:
  models:
    - path: ~/models/aurora-spec-qwen3-coder
      description: "Eagle3 draft head"
  gpu_memory_gb: 110
  disk_gb: 60
  cuda_version: ">=12.4"
  driver_version: ">=550"
```

## Notes Section

Free-text operational guidance:

```yaml
notes:
  - "CUDA graphs enabled. First 2-3 requests after restart are 3-5x slower."
  - "Draft head uses 32K/151K vocab. Non-Latin tokens fall back to baseline."
  - "gpu_memory_utilization above 0.92 causes OOM under concurrent load."
```

Notes capture the knowledge that usually lives in Slack threads and
Reddit comments. They are the most human part of a serving card.
