# servingcard

> Hardware-specific LLM serving configurations. Model cards for serving.

[![Spec Version](https://img.shields.io/badge/spec-v1.0-blue)](spec/servingcard-v1.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](packages/python/)
[![CI](https://img.shields.io/badge/CI-validate-passing)](https://github.com/zenprocess/servingcard/actions)

## Why Serving Cards?

Searching for optimal vLLM parameters returns Reddit threads, not config files.
Every time someone gets a new GPU and wants to serve a model, they spend hours
reading scattered GitHub issues, Discord messages, and blog posts to find the
right combination of tensor parallelism, quantization, speculative decoding,
and memory utilization settings. Then they run benchmarks, tweak, and repeat.
The knowledge lives in their head and their shell history.

No standard exists that combines model identity, hardware profile, framework
parameters, and verified benchmarks in a single portable file. HuggingFace
model cards tell you *what* a model is -- its architecture, training data,
evaluation scores. They never tell you *how to serve it* on your specific
hardware. Docker Compose files capture launch commands but not benchmark
results. Shell scripts work for one person but are not shareable knowledge.

A serving card is a YAML file that captures everything needed to reproduce a
serving configuration *and* the proof that it works. It is the serving
counterpart to a model card: one file per model-hardware-framework combination,
with required benchmarks so that configs without performance data are explicitly
not serving cards. The format is human-readable, machine-parseable, and designed
for community sharing.

## Key Features

- **Hardware-specific** -- same model, different configs for RTX 4090 vs A100 vs GB10
- **Benchmarks required** -- configs without benchmarks are guesses, not serving cards
- **Framework-aware** -- vLLM, TGI, SGLang, llama.cpp params in one standard
- **One-command apply** -- `servingcard launch qwen3-coder/gb10-fp8-eagle3-spec3.yaml`
- **Autoresearch-compatible** -- auto-tuning tools export directly to serving cards
- **Community registry** -- share and discover optimized configs
- **Transform documentation** -- model output quirks (think tags, float coercion) captured alongside the config
- **Readiness checks** -- warmup sequences and health endpoints in the card itself

## When NOT to Use Serving Cards

Serving cards are not the right tool for every situation. Be honest about scope:

- **Cloud-managed inference** (Fireworks, Together, Bedrock) -- these providers optimize serving for you. A serving card adds nothing when you are calling an API endpoint.
- **One-off experiments** -- if you are running a quick test and will never use the config again, writing a serving card is overkill. Just use your shell history.
- **Training configurations** -- serving cards are inference-only. Training has different concerns (learning rates, batch sizes, gradient accumulation) that deserve their own format.
- **Model evaluation** -- serving cards tell you how fast a model runs, not how good its outputs are. Use eval harnesses (lm-eval, HELM) for quality assessment.
- **Multi-model pipelines** -- a serving card describes one model on one hardware configuration. If you need to orchestrate multiple models, use a higher-level tool and reference individual serving cards.

## Quick Start

```bash
# Clone the repo (PyPI package coming soon)
git clone https://github.com/zenprocess/servingcard
cd servingcard/packages/python
pip install -e .

# Validate a config
servingcard validate registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml

# Show summary info
servingcard info registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml

# Search the registry
servingcard search --model qwen3-coder --hardware nvidia-gb10

# Launch vLLM from a serving card
servingcard launch registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml
# Expands to: vllm serve Qwen/Qwen3-Coder-480B-A35B-FP8 \
#   --tensor-parallel-size 1 --max-model-len 131072 \
#   --gpu-memory-utilization 0.90 --quantization fp8 \
#   --speculative-model aurora-spec-qwen3-coder \
#   --num-speculative-tokens 3
```

## Format Overview

A serving card is a YAML file with structured sections. Here is the production
Eagle3 config with annotations explaining each part:

```yaml
# ---- Identity ----
servingcard: "1.0"                    # Spec version (required)
model: qwen3-coder                    # Model identifier (required)
variant: fp8-eagle3-spec3             # Config variant name (required)
hardware: nvidia-gb10                 # Target hardware (required)
framework: vllm                       # Inference framework (required)
framework_version: ">=0.8.0"          # Framework version constraint

# ---- Accountability ----
author: zenprocess                    # Who created and verified this (required)
created: "2026-03-26"                 # When (required)
method: autoresearch                  # How it was tuned (required)
method_detail: "378 iterations, Karpathy pattern"

# ---- Serving Parameters ----
# Framework-specific, passed directly to the engine.
# Keys correspond to vLLM CLI flags / config options.
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

# ---- Sampling Constraints ----
# What sampling parameters this config supports and their valid ranges.
sampling:
  temperature: {min: 0, max: 2, default: 0.2}
  top_p: {min: 0, max: 1, default: 1}
  logit_bias: unsupported              # Not supported under speculative decoding
  min_p: unsupported

# ---- Capacity ----
capacity:
  context_limit: 131072                # Tested max context length
  max_concurrent: 8                    # Tested max concurrent requests
  parallel_tool_calls: {max_reliable: 3}

# ---- Benchmarks (required) ----
# At minimum, single_stream.tok_s must be present.
benchmark:
  single_stream:
    tok_s: 69.0                        # Output tokens/sec, single request
    ttft_ms: 1541                      # Time to first token
    p99_latency_ms: 2200
  parallel:
    peak_tok_s: 469                    # Aggregate tok/s at peak concurrency
    concurrency: 8
  methodology:
    tool: benchmark_serving
    prompt_distribution: coding-tasks
    num_runs: 10
    confidence_interval: 0.95

# ---- Transforms ----
# Known model output quirks and how to fix them.
transforms:
  - type: regex_strip
    pattern: "<think>.*?</think>"
    description: "Strip Qwen3-coder reasoning tags from output"
  - type: coerce_float_to_int
    scope: tool_call_arguments
    description: "Fix float tool args (42.0 -> 42)"

# ---- Readiness ----
readiness:
  warmup_requests: 3
  warmup_prompt: "Say ok."
  warmup_max_tokens: 5
  health_endpoint: /health

# ---- Prerequisites ----
prerequisites:
  models:
    - path: ~/models/aurora-spec-qwen3-coder
      description: "Eagle3 draft head for speculative decoding"
  gpu_memory_gb: 110

# ---- Notes ----
notes:
  - "CUDA graphs enabled. First 2-3 requests after restart are 3-5x slower."
  - "Draft head uses 32K/151K vocab. Non-Latin tokens fall back to baseline speed."
```

## Registry

Curated, validated configurations with reproducible benchmarks.

| Model | Hardware | Variant | tok/s | TTFT | Context | Method |
|-------|----------|---------|------:|-----:|--------:|--------|
| qwen3-coder | NVIDIA GB10 | FP8 + Eagle3 (spec=3) | **69** | 1541ms | 131K | autoresearch |
| qwen3-coder | NVIDIA GB10 | FP8 baseline | 42 | 780ms | 131K | autoresearch |
| qwen3-coder | NVIDIA GB10 | NVFP4 | 42 | 650ms | 262K | autoresearch |

Registry path convention: `registry/{model}/{hardware}-{quant}-{variant}.yaml`

*Your config here -- [contribute](#contributing)*

## Using Serving Cards Programmatically

Any tool that manages LLM inference can use serving cards to make runtime
decisions:

```python
from servingcard.schema import ServingCard

card = ServingCard.from_yaml("registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml")

# Capacity planning: don't exceed tested limits
if card.capacity:
    max_context = card.capacity.context_limit   # 131072
    max_concurrent = card.capacity.max_concurrent  # 8

# Benchmark data: estimate completion time
if card.benchmark and card.benchmark.single_stream:
    estimated_time = output_tokens / card.benchmark.single_stream.tok_s

# Transforms: apply known model quirks
for transform in (card.transforms or []):
    if transform.type == "regex_strip":
        output = re.sub(transform.pattern, "", output, flags=re.DOTALL)
```

This replaces hardcoded constants scattered across config files with a single
source of truth that is versioned, validated, and benchmarked.

## Benchmarks

### Methodology

All benchmarks in the registry follow these principles:

1. **Tool**: benchmarks use `benchmark_serving` from the vLLM project with the `sonnet` dataset, unless otherwise noted in the card's `methodology` section.

2. **Warmup**: 3 warmup requests are sent before measurement begins. CUDA graph compilation makes the first few requests 3-5x slower -- these are excluded.

3. **Single-stream baseline**: every card includes at minimum `single_stream.tok_s` measured at concurrency=1. This is the universal comparison point.

4. **Parallel throughput**: cards that include `parallel` benchmarks specify the concurrency level. Peak throughput numbers without concurrency context are meaningless.

5. **Reproducibility**: the `methodology` section documents the benchmarking tool, prompt distribution, number of runs, and confidence interval. Cards without methodology are still valid but carry less weight.

6. **Environment pinning**: driver version, CUDA version, and framework version are captured so that benchmark numbers can be reproduced on matching hardware.

### Interpreting benchmark numbers

- **tok/s** is output tokens per second. Higher is better for throughput.
- **TTFT** (time to first token) matters for interactive use. Eagle3 speculative decoding trades higher TTFT (1541ms) for higher throughput (69 tok/s vs 42 tok/s baseline).
- **p99 latency** captures worst-case behavior under normal load. Important for SLA-bound deployments.

## Documentation

- [Specification](spec/servingcard-v1.md) -- full YAML schema with JSON Schema appendix
- [Getting Started](docs/getting-started.md) -- quick intro to creating your first serving card
- [Format Overview](docs/format-overview.md) -- detailed guide to every section
- [Benchmark Guide](docs/benchmarks.md) -- how to run benchmarks and report results
- [Contributing](#contributing) -- how to submit a serving card
- [Examples](examples/) -- integration examples

## Contributing

Submit a serving card in 4 steps:

1. **Benchmark your model** on your hardware using a standard workload. We recommend
   `benchmark_serving` from vLLM with the `sonnet` dataset at concurrency=1 for the
   single-stream baseline.

2. **Write the YAML** following the [specification](spec/servingcard-v1.md). Copy an
   existing card from `registry/` as a starting point. Include at minimum the required
   fields and `benchmark.single_stream.tok_s`.

3. **Validate** your card:
   ```bash
   # Clone the repo (PyPI package coming soon)
   git clone https://github.com/zenprocess/servingcard
   cd servingcard/packages/python && pip install -e .
   servingcard validate your-config.yaml
   ```

4. **Submit a PR** placing the file at `registry/{model}/{hardware}-{quant}-{variant}.yaml`.
   CI validates automatically. Benchmark claims are spot-checked by maintainers with
   matching hardware when possible.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide, including benchmark methodology
requirements, PR template, and review criteria.

### What makes a good serving card

- **Reproducible** -- someone with the same GPU can follow it exactly
- **Benchmarked** -- real numbers from real runs, not estimates
- **Contextualized** -- notes explain tradeoffs (e.g., "2x TTFT for 64% more throughput")
- **Versioned** -- pin the engine version and driver so results can be reproduced
- **Honest** -- document limitations and failure modes, not just best-case numbers

## Design Decisions

### Why YAML?

Serving cards are written and read by humans first. YAML supports comments,
multi-line strings, and readable nested structures. JSON Schema is provided
in the specification for programmatic validation, but YAML is the source
format because people need to edit these files by hand.

### Why is hardware required?

Optimal serving parameters are hardware-dependent. `gpu_memory_utilization: 0.95`
may work on a 128 GB GB10 but OOM on a 24 GB RTX 4090.
`tensor_parallel_size: 4` is meaningless on a single-GPU system. Benchmark
numbers without hardware context are not actionable.

### Why are benchmarks required?

The gap between "this config starts without errors" and "this config performs
well" is enormous. Speculative decoding can halve throughput if misconfigured.
Quantization can degrade quality silently. Without benchmarks, a serving card
is just a guess with formatting. The `benchmark` section is the difference
between "I think this works" and "I measured this working."

### Why one file per configuration?

If your serving params are in one file and your sampling constraints in
another, they will drift apart. One file per model-hardware-framework
combination means one atomic unit of truth that can be versioned, diffed,
reviewed, and shared.

### Why is `serving` not standardized across frameworks?

Inference frameworks have genuinely different parameter surfaces. Attempting
to standardize `tensor_parallel_size` across vLLM, TGI, SGLang, llama.cpp,
and TensorRT-LLM would either miss framework-specific features or create a
leaky abstraction. Instead, the `serving` section is framework-specific by
design, while `sampling`, `capacity`, and `benchmark` use universal concepts.

### Why include transforms?

Many models have known output quirks: Qwen3-coder emits `<think>` tags, some
models return floats where integers are expected in tool calls, others produce
malformed JSON. Documenting quirks alongside the serving config means consumers
apply the right fixes without rediscovering them through trial and error.

## FAQ

**Can I use a serving card without the Python package?**

Yes. A serving card is just a YAML file. You can parse it with any YAML
library in any language. The Python package provides validation and CLI
convenience, but the format is language-agnostic.

**What if my hardware is not in the registry?**

Create a new serving card for your hardware. That is the whole point -- the
registry grows as people contribute configs for new hardware. Start by copying
an existing card and adjusting the parameters and benchmarks.

**Can a serving card describe multiple GPUs (multi-node)?**

Yes, but treat the multi-GPU setup as a single hardware configuration. Use a
hardware slug like `nvidia-h100-8x` and set `tensor_parallel_size: 8` in the
serving section. The card still describes one serving configuration.

**How do I compare two serving cards?**

Look at `benchmark.single_stream.tok_s` for throughput comparison and
`benchmark.single_stream.ttft_ms` for latency comparison. The registry
table in this README shows these side by side. For deeper comparison,
`servingcard info` shows a formatted summary.

**What about quality benchmarks (MMLU, HumanEval, etc.)?**

Serving cards focus on serving performance, not model quality. Quality
benchmarks belong in model cards. A serving card tells you how fast a model
runs on specific hardware, not how good its outputs are.

## Integrations

| Tool | Status | Description |
|------|--------|-------------|
| vLLM | `servingcard launch` | Generate vLLM CLI from a serving card |
| Multi-agent dispatchers | Compatible | Any dispatcher can read serving cards for routing and capacity |
| [auto-tuning-vllm](https://github.com/zenprocess/auto-tuning-vllm) | Planned | Export tuning results as serving cards |
| TGI | Planned | `servingcard launch --engine tgi` param mapping |
| SGLang | Planned | SGLang param mapping |
| Your tool here | -- | PRs welcome |

## Ecosystem

The long-term vision: auto-tuning tools like
[autoresearch](https://github.com/karpathy/autoresearch) run parameter sweeps
and export the best configurations as serving cards. The registry grows
organically as people share what works on their hardware. Inference frameworks
add native `--serving-card` flags to consume cards directly. The community
builds a searchable index of verified, benchmarked configurations.

Today: three configs for Qwen3-Coder on NVIDIA GB10. Tomorrow: every model,
every GPU, every framework -- with real benchmarks attached to every claim.

## Project Structure

```
servingcard/
  registry/              # Curated configs by model
    qwen3-coder/         #   One directory per model
  spec/                  # Schema specification
  packages/python/       # pip-installable CLI + library
  examples/              # Integration examples
  docs/                  # Guides and reference
  .github/               # CI workflows and issue templates
```

## Credits

Created by [zenprocess](https://github.com/zenprocess).

Inspired by [HuggingFace Model Cards](https://huggingface.co/docs/hub/model-cards),
[TOON format](https://github.com/toon-format/toon), and
[Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## License

[MIT](LICENSE)
