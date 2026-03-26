# servingcard

Hardware-specific LLM serving configurations. Model cards for serving.

**One YAML file captures the exact vLLM/TGI parameters, hardware profile, and benchmarks for a tuned model deployment — so the next person with the same GPU doesn't start from scratch.**

## The Problem

Finding optimal serving parameters for your hardware means reading dozens of Reddit threads, GitHub issues, and blog posts. Every combination of model, GPU, quantization, and speculative decoding method requires different tuning. No standard format exists for sharing "I tuned model X on hardware Y with framework Z — here are the optimal params and benchmarks."

## Quick Start

A servingcard is a YAML file that captures everything needed to reproduce a serving configuration:

```yaml
# registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml
servingcard: v1

model:
  name: Qwen/Qwen3-Coder-480B-A35B-FP8
  params: 480B
  active_params: 35B
  architecture: MoE
  quantization: fp8

hardware:
  gpu: NVIDIA GB10
  vram_gb: 128
  driver: "570.133.20"
  cuda: "12.8"

engine:
  name: vllm
  version: "0.9.1"
  args:
    tensor-parallel-size: 1
    max-model-len: 65536
    gpu-memory-utilization: 0.97
    enable-chunked-prefill: true
    max-num-batched-tokens: 2048
    speculative-model: EAGLE3-Qwen3-Coder-480B-A35B-Instruct-FP8
    num-speculative-tokens: 3
    speculative-disable-mqa-scorer: true

benchmarks:
  tool: benchmark_serving
  dataset: sonnet
  concurrency: 1
  results:
    output_tok_per_s: 69.01
    ttft_ms: 1541
    tpot_ms: 14.49
    input_tokens: 550
    output_tokens: 150

method: autoresearch        # how this config was found
author: zen                 # who submitted it
date: 2025-03-20
notes: |
  Eagle3 speculative decoding with 3 spec tokens.
  64% throughput gain over FP8 baseline at cost of 2x TTFT.
```

Validate and inspect it:

```bash
pip install servingcard
servingcard validate registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml
servingcard info registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml
```

Apply it directly to a vLLM launch:

```bash
servingcard launch registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml
# Expands to: vllm serve Qwen/Qwen3-Coder-480B-A35B-FP8 \
#   --tensor-parallel-size 1 --max-model-len 65536 ...
```

## What's in a Serving Card?

| Section | What it captures | Why it matters |
|---------|-----------------|----------------|
| `model` | Name, size, quant, architecture | Know exactly what's being served |
| `hardware` | GPU, VRAM, driver, CUDA version | Match configs to your hardware |
| `engine` | Framework, version, all launch args | Reproduce the exact setup |
| `benchmarks` | tok/s, TTFT, TPOT, test conditions | Compare configurations objectively |
| `method` | How the config was found | Trust level: manual tuning vs automated search |
| `notes` | Tradeoffs, gotchas, context | The knowledge that usually lives in Reddit comments |

## Registry

Curated, validated configurations with reproducible benchmarks.

| Model | Hardware | Variant | tok/s | TTFT | Method |
|-------|----------|---------|-------|------|--------|
| Qwen3-Coder-480B | GB10 | FP8 + Eagle3 (spec=3) | 69 | 1541ms | autoresearch |
| Qwen3-Coder-480B | GB10 | FP8 baseline | 42 | 780ms | manual |
| Qwen3-Coder-480B | GB10 | NVFP4 | 42 | 650ms | manual |

Registry path convention: `registry/{model}/{hardware}-{quant}-{variant}.yaml`

## Contributing

**Submit a config in 3 steps:**

1. **Benchmark** your setup with a standard workload (we recommend `benchmark_serving` from vLLM with the `sonnet` dataset at concurrency=1 for single-user comparison).

2. **Write** the YAML following the schema. Copy an existing card from `registry/` as a starting point.

3. **Validate and PR:**
   ```bash
   servingcard validate your-config.yaml
   # Place it in registry/{model}/{hardware}-{quant}-{variant}.yaml
   # Open a PR
   ```

Every PR is validated in CI. Benchmark claims are spot-checked by maintainers with matching hardware.

### What makes a good servingcard

- Reproducible: someone with the same GPU can follow it exactly
- Benchmarked: real numbers, not estimates
- Contextualized: notes explain tradeoffs (e.g., "2x TTFT for 64% more throughput")
- Versioned: pin the engine version and driver

## Spec

The full schema specification is at [`spec/servingcard-v1.md`](spec/servingcard-v1.md).

## Integrations

| Project | Integration | Status |
|---------|------------|--------|
| [vLLM](https://github.com/vllm-project/vllm) | `servingcard launch` generates vLLM CLI | Available |
| [Switchyard](https://github.com/zenprocess/switchyard) | Auto-selects serving config for Hermes dispatch | Available |
| [auto-tuning-vllm](https://github.com/zenprocess/auto-tuning-vllm) | Automated search outputs servingcard YAML | Planned |
| TGI | `servingcard launch --engine tgi` | Planned |

## Project Structure

```
servingcard/
  registry/          # curated configs by model
  spec/              # schema specification
  packages/python/   # pip-installable CLI + library
  examples/          # annotated examples
```

## License

MIT
