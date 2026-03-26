# Contributing to servingcard

Thank you for contributing a serving card. Every validated, benchmarked config
saves the next person hours of trial and error.

## Submitting a Serving Card

### 1. Benchmark Your Setup

Run a standard benchmark on your hardware. We recommend `benchmark_serving`
from the vLLM project:

```bash
python -m vllm.entrypoints.openai.api_server --model <your-model> <your-flags> &

# Wait for server to be ready, then:
python benchmarks/benchmark_serving.py \
  --model <your-model> \
  --dataset-name sonnet \
  --num-prompts 50 \
  --request-rate 1 \
  --endpoint /v1/completions
```

Record at minimum:
- **Output tokens/sec** (single stream, concurrency=1)
- **TTFT** (time to first token)
- **Input/output token counts** used in the benchmark
- **GPU**, **driver version**, **CUDA version**, **framework version**

### 2. Write the YAML

Copy an existing card from `registry/` as a starting point. The required fields are:

```yaml
servingcard: "1.0"           # Spec version
model: <model-name>          # HuggingFace ID or short name
variant: <config-variant>    # e.g., fp8-baseline, fp16-eagle3-spec3
hardware: <hardware-slug>    # e.g., nvidia-rtx4090, nvidia-a100-80g
framework: <framework>       # e.g., vllm, tgi, sglang
author: <your-name>          # Who created and verified this
created: "<YYYY-MM-DD>"      # When
method: <tuning-method>      # manual, autoresearch, auto-tune-vllm, upstream, community

benchmark:
  single_stream:
    tok_s: <number>          # Required: output tokens per second
```

See the [specification](spec/servingcard-v1.md) for all available sections.

### 3. Validate

```bash
pip install servingcard
servingcard validate your-config.yaml
```

Fix any errors before submitting.

### 4. Submit a Pull Request

Place the file at:
```
registry/<model>/<hardware>-<quant>-<variant>.yaml
```

Examples:
- `registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml`
- `registry/llama4-scout/a100-80g-fp16-baseline.yaml`
- `registry/deepseek-r1/h100-fp8-tp4.yaml`

Use lowercase with hyphens as separators.

## YAML Format Requirements

- **YAML 1.2** syntax
- **2-space indentation** (no tabs)
- **Quote dates** as strings: `created: "2026-03-26"` (not `created: 2026-03-26`)
- **Quote version strings**: `servingcard: "1.0"` (not `servingcard: 1.0`)
- **No trailing whitespace**
- File must end with a newline

## Benchmark Methodology Requirements

Your serving card should include enough information for someone else to
reproduce your benchmark results:

1. **Pin versions** -- framework version, driver version, CUDA version in
   the card. Benchmark numbers change across versions.

2. **Warmup first** -- exclude the first few requests from measurements.
   CUDA graph compilation makes initial requests 3-5x slower.

3. **Document the workload** -- what prompts did you use? What input/output
   sizes? The `methodology` section in `benchmark` captures this.

4. **Single-stream is the baseline** -- always include `single_stream.tok_s`
   at concurrency=1. This is the universal comparison point.

5. **Be honest about tradeoffs** -- if Eagle3 gives you 64% more throughput
   but 2x TTFT, say so in the notes.

## Review Criteria

Maintainers review PRs for:

- [ ] All required fields present and valid
- [ ] `servingcard validate` passes with no errors
- [ ] Benchmark numbers are plausible for the stated hardware
- [ ] Notes document tradeoffs and limitations
- [ ] File is placed in the correct registry path
- [ ] No duplicate of an existing card (same model + hardware + variant)

Benchmark claims are spot-checked by maintainers with matching hardware
when possible. If we cannot verify, we will note that in the card.

## Updating an Existing Card

If you have improved benchmarks or found better parameters for an existing
configuration:

1. Update the existing YAML file (do not create a duplicate)
2. Update the `created` date
3. Note what changed in the PR description
4. If benchmark numbers changed significantly, explain why

## Submitting via Issue

If you prefer not to open a PR, you can submit a serving card via
[issue template](https://github.com/zenprocess/servingcard/issues/new?template=new-servingcard.md).
Paste your YAML into the issue and a maintainer will add it to the registry.

## Code of Conduct

Be respectful and constructive. We are building a shared knowledge base.
Benchmark numbers should be accurate -- do not inflate results. If you find
an error in an existing card, open an issue or PR to fix it.

## Questions?

Open an issue or start a discussion. We are happy to help with benchmarking
methodology, YAML format questions, or anything else.
