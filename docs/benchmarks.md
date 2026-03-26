# Benchmark Guide

Every serving card requires benchmark results. This guide explains how to
run benchmarks and what to report.

## Why Benchmarks Are Required

A serving configuration without benchmarks is a guess. Speculative decoding
can halve throughput if misconfigured. Quantization can degrade quality
silently. The gap between "this config starts without errors" and "this
config actually performs well" is enormous.

Benchmarks are the proof that a configuration works.

## Running Benchmarks

### Prerequisites

1. Your model is loaded and serving requests
2. You have a benchmark tool installed
3. The system is idle (no other GPU workloads)

### Recommended Tool: vLLM benchmark_serving

```bash
# Start your model server
vllm serve <model> <your-flags> &

# Wait for the server to be ready
curl --retry 30 --retry-delay 2 http://localhost:8000/health

# Run warmup (exclude these from measurements)
for i in 1 2 3; do
  curl -s http://localhost:8000/v1/completions \
    -d '{"model":"<model>","prompt":"Say ok.","max_tokens":5}' > /dev/null
done

# Single-stream benchmark (concurrency=1)
python benchmarks/benchmark_serving.py \
  --model <model> \
  --dataset-name sonnet \
  --num-prompts 50 \
  --request-rate 1 \
  --endpoint /v1/completions

# Parallel benchmark (find peak throughput)
for c in 2 4 8 16; do
  python benchmarks/benchmark_serving.py \
    --model <model> \
    --dataset-name sonnet \
    --num-prompts 100 \
    --request-rate $c \
    --endpoint /v1/completions
done
```

### What to Record

From the benchmark output, extract:

| Metric | Field | Description |
|--------|-------|-------------|
| Output tok/s | `single_stream.tok_s` | Tokens per second at concurrency=1 |
| TTFT | `single_stream.ttft_ms` | Time to first token in milliseconds |
| p99 latency | `single_stream.p99_latency_ms` | 99th percentile end-to-end latency |
| Input size | `single_stream.input_tokens` | Average input tokens in benchmark |
| Output size | `single_stream.output_tokens` | Average output tokens in benchmark |
| Peak parallel | `parallel.peak_tok_s` | Aggregate tok/s at best concurrency |
| Concurrency | `parallel.concurrency` | Concurrency level for peak result |

## Methodology Section

Document how you ran the benchmark so others can reproduce it:

```yaml
benchmark:
  single_stream:
    tok_s: 69.0
    ttft_ms: 1541
  methodology:
    tool: benchmark_serving          # What tool
    prompt_distribution: sonnet      # What prompts
    num_runs: 10                     # How many runs
    confidence_interval: 0.95        # Statistical confidence
    date: "2026-03-25"              # When
    notes: "Measured after warmup, CUDA graphs compiled"
```

## Common Pitfalls

### 1. Not warming up

The first 2-3 requests after server restart are 3-5x slower due to CUDA
graph compilation. Always send warmup requests before measuring.

### 2. Measuring during compilation

If you see a sudden jump in latency mid-benchmark, CUDA graphs may be
recompiling. Wait for compilation to finish before measuring.

### 3. Reporting theoretical maximums

Report what you actually measured, not what the hardware datasheet says.
A card claiming 120 tok/s on an RTX 4090 when 80 tok/s is the measured
reality hurts the community.

### 4. Ignoring TTFT

Throughput (tok/s) is not the only metric. Speculative decoding often
trades higher TTFT for higher throughput. Document both so users can
make informed tradeoffs.

### 5. Single benchmark run

Run at least 5 iterations, ideally 10. Single runs are noisy. Report
the median or mean, and note the variance if it is significant.

### 6. Benchmarking under load

Single-stream benchmarks should run at concurrency=1 with no other
requests. If you benchmark under load, report it as a parallel benchmark
with the concurrency level specified.

## Interpreting Results

- **tok/s** (tokens per second): Higher is better for throughput-bound
  workloads. This is the primary comparison metric.

- **TTFT** (time to first token): Lower is better for interactive use.
  Critical for chat applications and streaming responses.

- **p99 latency**: The worst-case latency for 99% of requests. Important
  for SLA-bound deployments. A server with great average latency but
  terrible p99 is unreliable.

- **Throughput vs latency tradeoff**: Speculative decoding (Eagle3) and
  batching improve throughput but increase TTFT. Document the tradeoff
  in the notes section.

## Example: Complete Benchmark Section

```yaml
benchmark:
  single_stream:
    tok_s: 69.0
    ttft_ms: 1541
    p99_latency_ms: 2200
    input_tokens: 4096
    output_tokens: 512
  parallel:
    peak_tok_s: 469
    concurrency: 8
  methodology:
    tool: benchmark_serving
    prompt_distribution: coding-tasks
    num_runs: 10
    confidence_interval: 0.95
    date: "2026-03-25"
    notes: >
      Measured after 3 warmup requests. CUDA graphs compiled.
      Single GB10 with 128GB unified memory. No other GPU workloads.
      Driver 570.133.20, CUDA 12.8, vLLM 0.9.1.
```
