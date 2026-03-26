# Getting Started with Serving Cards

This guide walks you through creating, validating, and using your first
serving card.

## Installation

```bash
pip install servingcard
```

This installs the `servingcard` CLI and the Python library.

## Your First Serving Card

A serving card captures the exact configuration and benchmark results for
serving a model on specific hardware. Start by creating a YAML file:

```yaml
servingcard: "1.0"
model: llama4-scout
variant: fp16-baseline
hardware: nvidia-rtx4090
framework: vllm
author: your-name
created: "2026-03-26"
method: manual

serving:
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.90
  max_model_len: 32768

benchmark:
  single_stream:
    tok_s: 38.5
    ttft_ms: 420
```

Save this as `my-first-card.yaml`.

## Validate

Run the validator to check your card:

```bash
servingcard validate my-first-card.yaml
```

If everything is correct, you will see:

```
VALID: my-first-card.yaml
```

If there are errors, the validator tells you exactly what to fix:

```
INVALID: my-first-card.yaml
  - Missing required field: author
  - Missing benchmark section
```

## Inspect

View a summary of any serving card:

```bash
servingcard info my-first-card.yaml
```

Output:

```
Model:      llama4-scout
Variant:    fp16-baseline
Hardware:   nvidia-rtx4090
Framework:  vllm
Author:     your-name
Method:     manual

Benchmark:
  Single stream: 38.5 tok/s
  TTFT:          420 ms
```

## Use in Code

Load a serving card in Python:

```python
from servingcard.schema import ServingCard

card = ServingCard.from_yaml("my-first-card.yaml")

print(card.model)          # llama4-scout
print(card.hardware)       # nvidia-rtx4090

if card.benchmark and card.benchmark.single_stream:
    print(f"{card.benchmark.single_stream.tok_s} tok/s")
```

## Browse the Registry

Search for existing configs:

```bash
servingcard search --model qwen3-coder
servingcard search --hardware nvidia-gb10
```

## Next Steps

- Read the [Format Overview](format-overview.md) for details on every section
- Read the [Benchmark Guide](benchmarks.md) for how to run proper benchmarks
- Browse the [registry](../registry/) for real-world examples
- [Contribute](../CONTRIBUTING.md) your own serving card
