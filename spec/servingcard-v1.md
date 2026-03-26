# ServingCard v1 Specification

**Version**: 1.0
**Status**: Draft
**Date**: 2026-03-26
**Authors**: zenprocess

---

## Abstract

A ServingCard is a structured, machine-readable YAML document that captures the complete serving configuration for a large language model on specific hardware. It is the serving counterpart to a model card: where model cards describe what a model *is*, ServingCards describe how a model *runs*.

A ServingCard answers one question: **"What is the optimal way to serve model X on hardware Y with framework Z, and how do we know it works?"**

Every ServingCard includes verified benchmark results. A configuration without benchmarks is an untested recipe, not a ServingCard.

---

## Motivation

Model serving configurations are tribal knowledge. Teams discover optimal tensor parallelism, quantization settings, speculative decoding parameters, and sampling constraints through expensive trial and error --- then store them in scattered shell scripts, Docker Compose files, and Slack threads.

ServingCard standardizes this knowledge into a single portable file that is:

- **Human-readable** --- YAML, not protobuf or binary formats
- **Machine-parseable** --- strict schema, validatable with JSON Schema
- **Hardware-specific** --- the same model has different optimal configs on an RTX 4090 vs an A100 vs a GB10
- **Benchmarked** --- every card carries proof that the configuration actually works
- **Auditable** --- author, date, tuning method, and benchmark methodology are all required

---

## Terminology

| Term | Definition |
|------|-----------|
| **ServingCard** | A YAML file conforming to this specification |
| **Variant** | A named serving configuration for a model (e.g., `fp8-eagle3-spec3`) |
| **Hardware identifier** | A short slug identifying a GPU or accelerator class (e.g., `nvidia-a100-80g`, `nvidia-gb10`, `nvidia-rtx4090`) |
| **Framework** | The inference engine used to serve the model (e.g., `vllm`, `tgi`, `sglang`, `llamacpp`, `tensorrt-llm`) |
| **Transform** | A post-processing rule applied to model output to fix known quirks |

---

## File Naming Convention

ServingCard files MUST use the following naming pattern:

```
{model}-{hardware}-{variant}.yaml
```

Examples:
- `qwen3-coder-gb10-eagle3-spec3.yaml`
- `llama4-scout-a100-80g-fp16-baseline.yaml`
- `deepseek-r1-h100-srt-tp4-fp8.yaml`

The filename components are derived from the `model`, `hardware`, and `variant` fields in the document. Use lowercase with hyphens as separators.

---

## Schema

### Top-Level Structure

A ServingCard is a YAML document with the following top-level keys:

| Key | Type | Required | Description |
|-----|------|----------|-------------|
| `servingcard` | string | YES | Specification version. Must be `"1.0"` for this version. |
| `model` | string | YES | HuggingFace model ID or local model name. |
| `variant` | string | YES | Name for this serving variant. |
| `hardware` | string | YES | Hardware identifier slug. |
| `framework` | string | YES | Inference framework name. |
| `framework_version` | string | no | Version constraint (e.g., `">=0.8.0"`, `"0.8.2"`, `">=0.7,<0.9"`). |
| `author` | string | YES | Person or organization that created and verified this card. |
| `created` | string (date) | YES | ISO 8601 date when the card was created (e.g., `"2026-03-26"`). |
| `method` | string | YES | How this configuration was tuned. See Method Values. |
| `method_detail` | string | no | Free-text detail about the tuning process. |
| `license` | string | no | License for the ServingCard file itself (not the model). |
| `serving` | object | no | Framework-specific serving parameters. |
| `sampling` | object | no | Sampling parameter constraints. |
| `capacity` | object | no | Capacity and concurrency limits. |
| `benchmark` | object | YES | Verified benchmark results. |
| `transforms` | list | no | Response post-processing transforms. |
| `readiness` | object | no | Warmup and health check configuration. |
| `prerequisites` | object | no | System requirements for this configuration. |
| `notes` | list of strings | no | Free-text operational notes. |

### Required Fields

Five top-level fields plus the `benchmark` section are required for a valid ServingCard:

```yaml
servingcard: "1.0"           # REQUIRED - spec version
model: qwen3-coder           # REQUIRED - what model
variant: fp8-eagle3-spec3    # REQUIRED - which config
hardware: nvidia-gb10        # REQUIRED - what hardware
framework: vllm              # REQUIRED - what framework
author: zenprocess            # REQUIRED - who made this
created: "2026-03-26"        # REQUIRED - when
method: autoresearch          # REQUIRED - how was this tuned

benchmark:                    # REQUIRED section
  single_stream:
    tok_s: 69.0               # REQUIRED - at minimum, tokens/second
```

### Method Values

The `method` field indicates how the serving configuration was discovered and validated. Standard values:

| Value | Meaning |
|-------|---------|
| `manual` | Hand-tuned by a human through experimentation |
| `autoresearch` | Automated research across parameter space (e.g., grid search, Bayesian optimization) |
| `auto-tune-vllm` | Tuned using vLLM's auto-tune facility |
| `upstream` | Configuration provided by the model author or framework maintainer |
| `community` | Contributed by community members, verified by card author |

Custom values are permitted. The field is free-text, but the standard values above SHOULD be used when applicable.

---

## Section Definitions

### `serving` --- Framework-Specific Parameters

The `serving` section contains parameters passed to the inference framework. These are **not standardized** across frameworks because each framework has different configuration surfaces. The keys in this section correspond directly to framework flags, config file keys, or API parameters.

```yaml
# vLLM example
serving:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  max_model_len: 131072
  max_num_seqs: 8
  quantization: fp8
  speculative_decoding:
    method: eagle3
    draft_model: aurora-spec-qwen3-coder
    num_speculative_tokens: 3
  enforce_eager: false
  enable_prefix_caching: true
  kv_cache_dtype: auto
```

```yaml
# TGI example
serving:
  sharded: true
  num_shard: 4
  quantize: bitsandbytes-fp4
  max_input_length: 8192
  max_total_tokens: 16384
  max_batch_prefill_tokens: 16384
```

```yaml
# llama.cpp example
serving:
  n_gpu_layers: 99
  n_ctx: 32768
  n_batch: 2048
  flash_attn: true
  mlock: true
  type_k: q8_0
  type_v: q8_0
```

Tools consuming ServingCards SHOULD pass recognized keys to the framework and ignore unrecognized keys with a warning.

### `sampling` --- Sampling Parameter Constraints

The `sampling` section documents which sampling parameters the model and framework combination supports and their valid ranges. This section uses **constraint syntax**.

#### Constraint Syntax

Each sampling parameter is expressed as one of:

| Form | Meaning | Example |
|------|---------|---------|
| `{min, max, default}` | Numeric range with a default | `{min: 0, max: 2, default: 0.6}` |
| `{min, max}` | Numeric range, no default specified | `{min: 0, max: 1}` |
| `{values, default}` | Enumerated options | `{values: [greedy, sample], default: sample}` |
| `"supported"` | Parameter is accepted (no constraints documented) | `logprobs: supported` |
| `"unsupported"` | Parameter is NOT supported by this configuration | `logit_bias: unsupported` |

```yaml
sampling:
  temperature: {min: 0, max: 2, default: 0.2}
  top_p: {min: 0, max: 1, default: 1}
  top_k: {min: -1, max: 1000, default: -1}
  repetition_penalty: {min: 0, max: 2, default: 1.0}
  logit_bias: unsupported
  min_p: unsupported
  logprobs: supported
  frequency_penalty: {min: -2, max: 2, default: 0}
```

Consumers SHOULD validate user-requested sampling parameters against these constraints before sending requests.

### `capacity` --- Limits

```yaml
capacity:
  context_limit: 131072          # Maximum context length in tokens
  max_concurrent: 8              # Maximum concurrent requests
  parallel_tool_calls:
    max_reliable: 3              # Max parallel tool calls that work reliably
```

All fields are optional. Values represent tested, verified limits for this specific configuration, not theoretical maximums.

### `benchmark` --- Verified Results

The `benchmark` section is **required**. At minimum, `single_stream.tok_s` must be present.

```yaml
benchmark:
  single_stream:
    tok_s: 69.0                  # Tokens per second (output), single request
    ttft_ms: 1541                # Time to first token, milliseconds
    p99_latency_ms: 2200         # 99th percentile end-to-end latency
    input_tokens: 4096           # Input size used for benchmark
    output_tokens: 512           # Output size used for benchmark
  parallel:
    peak_tok_s: 469              # Aggregate tok/s at peak concurrency
    concurrency: 8               # Concurrency level for parallel benchmark
  methodology:
    tool: benchmark_serving       # Benchmarking tool used
    prompt_distribution: coding-tasks  # Type of prompts used
    num_runs: 10                 # Number of benchmark runs
    confidence_interval: 0.95    # Statistical confidence level
    date: "2026-03-25"           # When benchmarks were run
    notes: "Measured after warmup, CUDA graphs compiled"
```

#### Benchmark Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `single_stream.tok_s` | float | YES | Output tokens per second for a single request |
| `single_stream.ttft_ms` | float | no | Time to first token in milliseconds |
| `single_stream.p99_latency_ms` | float | no | 99th percentile end-to-end latency |
| `single_stream.input_tokens` | int | no | Input token count used in benchmark |
| `single_stream.output_tokens` | int | no | Output token count used in benchmark |
| `parallel.peak_tok_s` | float | no | Aggregate throughput at stated concurrency |
| `parallel.concurrency` | int | no | Concurrency level tested |
| `methodology` | object | no | How benchmarks were conducted |

### `transforms` --- Response Post-Processing

Transforms describe post-processing rules that consumers SHOULD apply to model output. Each transform is an object with a `type` and type-specific fields.

```yaml
transforms:
  - type: regex_strip
    pattern: "<think>.*?</think>"
    flags: dotall
    description: "Strip Qwen3-coder reasoning tags from output"

  - type: coerce_float_to_int
    scope: tool_call_arguments
    description: "Fix Qwen3-coder float tool args (42.0 -> 42)"

  - type: json_repair
    scope: tool_call_arguments
    description: "Attempt to fix malformed JSON in tool call arguments"
```

#### Standard Transform Types

| Type | Fields | Description |
|------|--------|-------------|
| `regex_strip` | `pattern`, `flags` (optional) | Remove text matching regex from output |
| `regex_replace` | `pattern`, `replacement`, `flags` (optional) | Replace text matching regex |
| `coerce_float_to_int` | `scope` | Convert float values to integers where schema expects int |
| `json_repair` | `scope` | Attempt to repair malformed JSON |
| `truncate` | `max_tokens` | Truncate output beyond a token limit |

Custom transform types are permitted. Consumers SHOULD ignore unrecognized transform types with a warning.

### `readiness` --- Warmup and Health

```yaml
readiness:
  warmup_requests: 3            # Number of warmup requests to send
  warmup_prompt: "Say ok."      # Prompt text for warmup requests
  warmup_max_tokens: 5          # Max tokens per warmup request
  health_endpoint: /health      # HTTP health check endpoint path
  ready_timeout_s: 300          # Max seconds to wait for readiness
```

Consumers SHOULD send the specified warmup requests after framework startup before routing production traffic.

### `prerequisites` --- System Requirements

```yaml
prerequisites:
  models:
    - path: ~/models/aurora-spec-qwen3-coder
      description: "Eagle3 draft head for speculative decoding"
    - path: ~/models/qwen3-coder
      description: "Base model weights"
  gpu_memory_gb: 110            # Minimum GPU memory required
  disk_gb: 60                   # Minimum disk space for model files
  cuda_version: ">=12.4"        # CUDA version constraint
  driver_version: ">=550"       # GPU driver version constraint
```

### `notes` --- Operational Notes

A list of free-text strings for human-readable operational guidance that does not fit in structured fields.

```yaml
notes:
  - "CUDA graphs enabled. First 2-3 requests after restart are 3-5x slower."
  - "Draft head uses 32K/151K vocab. Non-Latin tokens fall back to baseline speed."
  - "gpu_memory_utilization above 0.92 causes OOM under concurrent load."
```

---

## Complete Example

```yaml
servingcard: "1.0"
model: qwen3-coder
variant: fp8-eagle3-spec3
hardware: nvidia-gb10
framework: vllm
framework_version: ">=0.8.0"

author: zenprocess
created: "2026-03-26"
method: autoresearch
method_detail: "378 iterations, Karpathy pattern"
license: MIT

serving:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  max_model_len: 131072
  max_num_seqs: 8
  quantization: fp8
  speculative_decoding:
    method: eagle3
    draft_model: aurora-spec-qwen3-coder
    num_speculative_tokens: 3
  enforce_eager: false
  enable_prefix_caching: true
  kv_cache_dtype: auto

sampling:
  temperature: {min: 0, max: 2, default: 0.2}
  top_p: {min: 0, max: 1, default: 1}
  logit_bias: unsupported
  min_p: unsupported

capacity:
  context_limit: 131072
  max_concurrent: 8
  parallel_tool_calls: {max_reliable: 3}

benchmark:
  single_stream:
    tok_s: 69.0
    ttft_ms: 1541
    p99_latency_ms: 2200
  parallel:
    peak_tok_s: 469
    concurrency: 8
  methodology:
    tool: benchmark_serving
    prompt_distribution: coding-tasks
    num_runs: 10
    confidence_interval: 0.95

transforms:
  - type: regex_strip
    pattern: "<think>.*?</think>"
    description: "Strip Qwen3-coder think tags"
  - type: coerce_float_to_int
    scope: tool_call_arguments
    description: "Fix float tool args (42.0 -> 42)"

readiness:
  warmup_requests: 3
  warmup_prompt: "Say ok."
  warmup_max_tokens: 5
  health_endpoint: /health

prerequisites:
  models:
    - path: ~/models/aurora-spec-qwen3-coder
      description: "Eagle3 draft head"
  gpu_memory_gb: 110

notes:
  - "CUDA graphs enabled. First 2-3 requests after restart are 3-5x slower."
  - "Draft head uses 32K/151K vocab. Non-Latin tokens fall back to baseline speed."
```

---

## Anti-Examples

### Not a ServingCard: missing benchmarks

```yaml
# INVALID - no benchmark section
servingcard: "1.0"
model: llama4-scout
variant: fp16-baseline
hardware: nvidia-a100-80g
framework: vllm
author: someone
created: "2026-03-20"
method: manual

serving:
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.95
  max_model_len: 65536
```

This is an untested recipe. Without `benchmark.single_stream.tok_s`, it cannot be validated as a ServingCard. Write it as a regular YAML config file until benchmarks are collected.

### Not a ServingCard: no hardware specificity

```yaml
# INVALID - hardware is missing
servingcard: "1.0"
model: deepseek-r1
variant: default
framework: vllm
author: someone
created: "2026-03-20"
method: manual

benchmark:
  single_stream:
    tok_s: 45.0
```

A configuration that claims to work on any hardware is making an untestable claim. Benchmark results are hardware-specific by definition. Always specify the hardware.

### Not a ServingCard: no accountability

```yaml
# INVALID - no author, no created date, no method
servingcard: "1.0"
model: qwen3-coder
variant: default
hardware: nvidia-rtx4090
framework: vllm

benchmark:
  single_stream:
    tok_s: 35.0
```

Who tuned this? When? How? Without accountability fields, configurations cannot be traced or verified. All three of `author`, `created`, and `method` are required.

### Not a ServingCard: multi-hardware in one file

```yaml
# INVALID - one file per hardware configuration
servingcard: "1.0"
model: qwen3-coder
variant: multi-gpu
hardware:                     # Don't do this
  - nvidia-a100-80g
  - nvidia-h100
framework: vllm
```

One file = one serving configuration. If the same model needs different configs on A100 vs H100, create two ServingCards:
- `qwen3-coder-a100-80g-fp16.yaml`
- `qwen3-coder-h100-fp8.yaml`

### Not a ServingCard: framework-agnostic

```yaml
# INVALID - framework is missing
servingcard: "1.0"
model: llama4-maverick
variant: default
hardware: nvidia-h100
author: someone
created: "2026-03-20"
method: manual

benchmark:
  single_stream:
    tok_s: 120.0
```

Serving parameters and performance characteristics are framework-dependent. The same model on the same hardware will behave differently on vLLM vs TGI vs SGLang. Always specify the framework.

---

## Design Rationale

### Why YAML, not JSON or protobuf?

ServingCards are written and read by humans first. YAML supports comments, multi-line strings, and readable nested structures. JSON Schema is provided for programmatic validation, but the source of truth is YAML.

### Why is hardware required?

Optimal serving parameters are hardware-dependent. `gpu_memory_utilization: 0.95` may work on a 128 GB GB200 but OOM on a 24 GB RTX 4090. `tensor_parallel_size: 4` is meaningless on a single-GPU system. Benchmark numbers without hardware context are useless.

### Why are benchmarks required?

The gap between "this config starts without errors" and "this config actually performs well" is enormous. Speculative decoding can halve throughput if misconfigured. Quantization can degrade quality silently. Without benchmarks, a ServingCard is just a guess with formatting.

### Why one file per configuration?

Split configs create synchronization bugs. If your serving params are in one file and your sampling constraints in another, they will drift apart. One file per configuration means one unit of truth that can be versioned, diffed, and shared atomically.

### Why is `serving` not standardized?

Inference frameworks evolve rapidly and have genuinely different parameter surfaces. Attempting to standardize `tensor_parallel_size` across vLLM, TGI, SGLang, llama.cpp, and TensorRT-LLM would either miss framework-specific features or create a leaky abstraction. Instead, the `serving` section is framework-specific by design, while `sampling`, `capacity`, and `benchmark` use universal concepts.

### Why `transforms`?

Many models have known output quirks: Qwen3-coder emits `<think>` tags, some models return floats where integers are expected in tool calls, others produce malformed JSON. Documenting these quirks alongside the serving config means consumers can apply the right fixes without rediscovering them.

---

## Versioning

The `servingcard` field indicates the spec version. This document defines version `"1.0"`.

Future versions will follow semantic versioning:
- **Patch** (1.0 -> 1.1): new optional fields, no breaking changes
- **Major** (1.0 -> 2.0): changes to required fields, structural changes, removed fields

Consumers SHOULD accept any card where the major version matches their supported version.

---

## Appendix A: JSON Schema

The following JSON Schema can be used for programmatic validation of ServingCard v1 documents.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://github.com/zenprocess/servingcard/spec/v1.0/schema.json",
  "title": "ServingCard v1.0",
  "description": "Hardware-specific LLM model serving configuration with verified benchmarks.",
  "type": "object",
  "required": [
    "servingcard",
    "model",
    "variant",
    "hardware",
    "framework",
    "author",
    "created",
    "method",
    "benchmark"
  ],
  "properties": {
    "servingcard": {
      "type": "string",
      "const": "1.0",
      "description": "Specification version."
    },
    "model": {
      "type": "string",
      "description": "HuggingFace model ID or local model name.",
      "minLength": 1
    },
    "variant": {
      "type": "string",
      "description": "Serving variant name.",
      "minLength": 1
    },
    "hardware": {
      "type": "string",
      "description": "Hardware identifier slug.",
      "minLength": 1
    },
    "framework": {
      "type": "string",
      "description": "Inference framework name.",
      "minLength": 1
    },
    "framework_version": {
      "type": "string",
      "description": "Framework version constraint."
    },
    "author": {
      "type": "string",
      "description": "Person or organization that created this card.",
      "minLength": 1
    },
    "created": {
      "type": "string",
      "format": "date",
      "description": "ISO 8601 date when the card was created."
    },
    "method": {
      "type": "string",
      "description": "How this configuration was tuned.",
      "minLength": 1
    },
    "method_detail": {
      "type": "string",
      "description": "Free-text detail about the tuning process."
    },
    "license": {
      "type": "string",
      "description": "License for the ServingCard file itself."
    },
    "serving": {
      "type": "object",
      "description": "Framework-specific serving parameters. Keys are framework-dependent.",
      "additionalProperties": true
    },
    "sampling": {
      "type": "object",
      "description": "Sampling parameter constraints.",
      "additionalProperties": {
        "oneOf": [
          {
            "type": "object",
            "properties": {
              "min": { "type": "number" },
              "max": { "type": "number" },
              "default": { "type": "number" }
            },
            "required": ["min", "max"]
          },
          {
            "type": "object",
            "properties": {
              "values": {
                "type": "array",
                "items": { "type": "string" }
              },
              "default": { "type": "string" }
            },
            "required": ["values"]
          },
          {
            "type": "string",
            "enum": ["supported", "unsupported"]
          }
        ]
      }
    },
    "capacity": {
      "type": "object",
      "description": "Capacity and concurrency limits.",
      "properties": {
        "context_limit": {
          "type": "integer",
          "description": "Maximum context length in tokens."
        },
        "max_concurrent": {
          "type": "integer",
          "description": "Maximum concurrent requests."
        },
        "parallel_tool_calls": {
          "type": "object",
          "properties": {
            "max_reliable": {
              "type": "integer",
              "description": "Max parallel tool calls that work reliably."
            }
          }
        }
      },
      "additionalProperties": true
    },
    "benchmark": {
      "type": "object",
      "description": "Verified benchmark results.",
      "required": ["single_stream"],
      "properties": {
        "single_stream": {
          "type": "object",
          "required": ["tok_s"],
          "properties": {
            "tok_s": {
              "type": "number",
              "description": "Output tokens per second, single request.",
              "exclusiveMinimum": 0
            },
            "ttft_ms": {
              "type": "number",
              "description": "Time to first token in milliseconds.",
              "exclusiveMinimum": 0
            },
            "p99_latency_ms": {
              "type": "number",
              "description": "99th percentile end-to-end latency in milliseconds.",
              "exclusiveMinimum": 0
            },
            "input_tokens": {
              "type": "integer",
              "description": "Input token count used in benchmark."
            },
            "output_tokens": {
              "type": "integer",
              "description": "Output token count used in benchmark."
            }
          },
          "additionalProperties": true
        },
        "parallel": {
          "type": "object",
          "properties": {
            "peak_tok_s": {
              "type": "number",
              "description": "Aggregate throughput at stated concurrency.",
              "exclusiveMinimum": 0
            },
            "concurrency": {
              "type": "integer",
              "description": "Concurrency level tested.",
              "minimum": 1
            }
          },
          "additionalProperties": true
        },
        "methodology": {
          "type": "object",
          "properties": {
            "tool": {
              "type": "string",
              "description": "Benchmarking tool used."
            },
            "prompt_distribution": {
              "type": "string",
              "description": "Type of prompts used."
            },
            "num_runs": {
              "type": "integer",
              "description": "Number of benchmark runs.",
              "minimum": 1
            },
            "confidence_interval": {
              "type": "number",
              "description": "Statistical confidence level.",
              "minimum": 0,
              "maximum": 1
            },
            "date": {
              "type": "string",
              "format": "date",
              "description": "When benchmarks were run."
            },
            "notes": {
              "type": "string",
              "description": "Additional benchmark context."
            }
          },
          "additionalProperties": true
        }
      },
      "additionalProperties": true
    },
    "transforms": {
      "type": "array",
      "description": "Response post-processing transforms.",
      "items": {
        "type": "object",
        "required": ["type"],
        "properties": {
          "type": {
            "type": "string",
            "description": "Transform type identifier."
          },
          "description": {
            "type": "string",
            "description": "Human-readable description of the transform."
          }
        },
        "additionalProperties": true
      }
    },
    "readiness": {
      "type": "object",
      "description": "Warmup and health check configuration.",
      "properties": {
        "warmup_requests": {
          "type": "integer",
          "description": "Number of warmup requests to send.",
          "minimum": 0
        },
        "warmup_prompt": {
          "type": "string",
          "description": "Prompt text for warmup requests."
        },
        "warmup_max_tokens": {
          "type": "integer",
          "description": "Max tokens per warmup request.",
          "minimum": 1
        },
        "health_endpoint": {
          "type": "string",
          "description": "HTTP health check endpoint path."
        },
        "ready_timeout_s": {
          "type": "integer",
          "description": "Max seconds to wait for readiness.",
          "minimum": 1
        }
      },
      "additionalProperties": true
    },
    "prerequisites": {
      "type": "object",
      "description": "System requirements for this configuration.",
      "properties": {
        "models": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["path"],
            "properties": {
              "path": {
                "type": "string",
                "description": "Path to required model or artifact."
              },
              "description": {
                "type": "string",
                "description": "What this artifact is."
              }
            },
            "additionalProperties": true
          }
        },
        "gpu_memory_gb": {
          "type": "number",
          "description": "Minimum GPU memory in gigabytes.",
          "exclusiveMinimum": 0
        },
        "disk_gb": {
          "type": "number",
          "description": "Minimum disk space in gigabytes.",
          "exclusiveMinimum": 0
        },
        "cuda_version": {
          "type": "string",
          "description": "CUDA version constraint."
        },
        "driver_version": {
          "type": "string",
          "description": "GPU driver version constraint."
        }
      },
      "additionalProperties": true
    },
    "notes": {
      "type": "array",
      "description": "Free-text operational notes.",
      "items": {
        "type": "string"
      }
    }
  },
  "additionalProperties": false
}
```

---

## Appendix B: Minimal Valid ServingCard

The smallest possible valid ServingCard:

```yaml
servingcard: "1.0"
model: llama4-scout
variant: baseline
hardware: nvidia-rtx4090
framework: vllm
author: janedoe
created: "2026-03-26"
method: manual

benchmark:
  single_stream:
    tok_s: 38.5
```

This is valid but minimal. A production ServingCard SHOULD include `serving`, `sampling`, `capacity`, and `readiness` sections.

---

## Appendix C: Hardware Identifier Conventions

Hardware identifiers SHOULD follow this naming pattern:

```
{vendor}-{chip}[-{memory}]
```

Examples:
| Identifier | Hardware |
|-----------|----------|
| `nvidia-rtx4090` | NVIDIA GeForce RTX 4090 (24 GB) |
| `nvidia-a100-40g` | NVIDIA A100 40 GB |
| `nvidia-a100-80g` | NVIDIA A100 80 GB |
| `nvidia-h100` | NVIDIA H100 80 GB |
| `nvidia-gb10` | NVIDIA GB10 (Grace Blackwell, 128 GB unified) |
| `nvidia-gb200` | NVIDIA GB200 |
| `amd-mi300x` | AMD Instinct MI300X |
| `apple-m4-ultra` | Apple M4 Ultra |

Memory suffixes are used only to disambiguate variants of the same chip (e.g., A100 40 GB vs 80 GB). When there is only one memory configuration for a chip, omit the suffix.

---

## Appendix D: Consuming a ServingCard

### Example: starting vLLM from a ServingCard (Python)

```python
import yaml
import subprocess

def start_from_servingcard(path: str) -> subprocess.Popen:
    with open(path) as f:
        card = yaml.safe_load(f)

    assert card["servingcard"] == "1.0"
    assert card["framework"] == "vllm"

    serving = card.get("serving", {})
    cmd = ["python", "-m", "vllm.entrypoints.openai.api_server"]
    cmd += ["--model", card["model"]]

    flag_map = {
        "tensor_parallel_size": "--tensor-parallel-size",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_model_len": "--max-model-len",
        "max_num_seqs": "--max-num-seqs",
        "quantization": "--quantization",
        "enforce_eager": "--enforce-eager",
        "enable_prefix_caching": "--enable-prefix-caching",
        "kv_cache_dtype": "--kv-cache-dtype",
    }

    for key, flag in flag_map.items():
        if key in serving:
            val = serving[key]
            if isinstance(val, bool):
                if val:
                    cmd.append(flag)
            else:
                cmd += [flag, str(val)]

    spec = serving.get("speculative_decoding", {})
    if spec:
        cmd += ["--speculative-model", spec["draft_model"]]
        cmd += ["--num-speculative-tokens", str(spec["num_speculative_tokens"])]

    return subprocess.Popen(cmd)
```

### Example: validating a ServingCard (CLI)

```bash
# Using check-jsonschema (pip install check-jsonschema)
check-jsonschema --schemafile servingcard-v1-schema.json card.yaml

# Using yq + jq
yq -o json card.yaml | check-jsonschema --schemafile servingcard-v1-schema.json /dev/stdin
```
