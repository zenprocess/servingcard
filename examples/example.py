"""Load a serving card and generate vLLM launch command."""
import yaml
from pathlib import Path

# Load a serving card
card = yaml.safe_load(Path("registry/qwen3-coder/gb10-fp8-eagle3-spec3.yaml").read_text())

print(f"Model: {card['model']}")
print(f"Hardware: {card['hardware']}")
print(f"Throughput: {card['benchmark']['single_stream']['tok_s']} tok/s")

# Generate vLLM launch command
serving = card.get('serving', {})
cmd = f"vllm serve {card['model']}"
if serving.get('quantization'):
    cmd += f" --quantization {serving['quantization']}"
if serving.get('gpu_memory_utilization'):
    cmd += f" --gpu-memory-utilization {serving['gpu_memory_utilization']}"
if serving.get('max_model_len'):
    cmd += f" --max-model-len {serving['max_model_len']}"

print(f"\nLaunch: {cmd}")
