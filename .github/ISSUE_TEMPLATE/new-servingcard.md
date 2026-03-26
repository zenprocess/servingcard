---
name: New Serving Card
about: Submit a new serving configuration for the registry
title: "[card] <model> on <hardware> — <variant>"
labels: new-card
assignees: ''
---

## Serving Card Submission

**Model**: <!-- e.g., qwen3-coder, llama4-scout, deepseek-r1 -->
**Hardware**: <!-- e.g., nvidia-rtx4090, nvidia-a100-80g, nvidia-gb10 -->
**Framework**: <!-- e.g., vllm, tgi, sglang -->
**Throughput**: <!-- e.g., 69 tok/s single stream -->

## YAML

Paste your complete serving card YAML below. Ensure it passes `servingcard validate` before submitting.

```yaml
servingcard: "1.0"
model:
variant:
hardware:
framework:
author:
created:
method:

benchmark:
  single_stream:
    tok_s:
```

## Benchmark Environment

- **GPU**: <!-- exact model, e.g., NVIDIA GB10 128GB -->
- **Driver**: <!-- e.g., 570.133.20 -->
- **CUDA**: <!-- e.g., 12.8 -->
- **Framework version**: <!-- e.g., vLLM 0.9.1 -->
- **Benchmark tool**: <!-- e.g., benchmark_serving, custom script -->
- **Number of runs**: <!-- e.g., 10 -->

## Notes

<!-- Any context about the config: tradeoffs, gotchas, how you found these params -->

## Checklist

- [ ] `servingcard validate` passes with no errors
- [ ] Benchmark numbers are from real runs (not estimates)
- [ ] Hardware, driver, CUDA, and framework versions are documented
- [ ] Notes explain any tradeoffs or limitations
