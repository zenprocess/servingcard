"""Example: Apply a servingcard to Switchyard's serving contract system.

Shows how to load a servingcard and extract the data Switchyard needs
for its dispatch routing and capacity planning.
"""

from pathlib import Path

from servingcard.schema import ServingCard

# Load the production servingcard
card = ServingCard.from_yaml(
    Path(__file__).parent.parent / "registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml"
)

print(f"Model:    {card.model}")
print(f"Variant:  {card.variant}")
print(f"Hardware: {card.hardware}")
print()

# -- Benchmark data for routing decisions --
if card.benchmark:
    if card.benchmark.single_stream:
        print(f"Expected throughput:  {card.benchmark.single_stream.tok_s} tok/s (single stream)")
    if card.benchmark.parallel:
        print(f"Peak parallel:        {card.benchmark.parallel.peak_tok_s} tok/s")
    if card.benchmark.latency:
        print(f"TTFT:                 {card.benchmark.latency.ttft_ms} ms")
print()

# -- Capacity for dispatch planning --
if card.capacity:
    print(f"Context limit:        {card.capacity.context_limit:,} tokens")
    print(f"Max concurrent:       {card.capacity.max_concurrent}")
    print(f"GPU memory util:      {card.capacity.gpu_memory_utilization}")
print()

# -- Convert to Switchyard serving contract --
# In production, Switchyard reads these values to populate its
# HermesAgent configuration and dispatch routing tables.
switchyard_contract = {
    "performance": {
        "single_tok_s": card.benchmark.single_stream.tok_s if card.benchmark and card.benchmark.single_stream else None,
        "peak_parallel_tok_s": card.benchmark.parallel.peak_tok_s if card.benchmark and card.benchmark.parallel else None,
        "ttft_ms": card.benchmark.latency.ttft_ms if card.benchmark and card.benchmark.latency else None,
    },
    "capacity": {
        "context_limit": card.capacity.context_limit if card.capacity else None,
        "max_concurrent": card.capacity.max_concurrent if card.capacity else None,
    },
    "quantization": card.quantization.method if card.quantization else None,
    "speculative": card.speculative_decoding.method if card.speculative_decoding else None,
}

print("Switchyard contract:")
for section, values in switchyard_contract.items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for k, v in values.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {section}: {values}")
