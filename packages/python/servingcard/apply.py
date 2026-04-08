"""Generate framework-specific launch commands from a servingcard."""

from __future__ import annotations

from servingcard.schema import ServingCard

REGISTRY_BASE_URL = (
    "https://raw.githubusercontent.com/zenprocess/servingcard/main/registry"
)


def resolve_source(source: str) -> str:
    """Resolve a config source to a fetchable URL or local path.

    Supports:
      - Local file path: ./my-config.yaml, /abs/path.yaml
      - Full URL: https://...
      - Registry shorthand: model/variant -> GitHub raw URL
    """
    # Full URL
    if source.startswith("http://") or source.startswith("https://"):
        return source

    # Local file (contains dot-slash, slash, or ends with .yaml/.yml)
    if "/" in source and (
        source.startswith(".")
        or source.startswith("/")
        or source.endswith(".yaml")
        or source.endswith(".yml")
    ):
        return source

    # Registry shorthand: model/variant
    if "/" in source:
        return f"{REGISTRY_BASE_URL}/{source}.yaml"

    # Bare name -- assume it is a local file
    return source


def generate_vllm_command(card: ServingCard) -> str:
    """Generate a vLLM serve command from a servingcard."""
    if not card.serving or not card.serving.engine_args:
        return f"# No engine_args in servingcard -- cannot generate vllm command\nvllm serve {card.model}"

    args = card.serving.engine_args.copy()
    model_id = args.pop("model", card.model)

    parts = [f"vllm serve {model_id}"]

    # Map engine_args keys to CLI flags
    for key, value in args.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                parts.append(f"  {flag}")
        else:
            parts.append(f"  {flag} {value}")

    # Add tensor-parallel-size if not in engine_args but inferable
    # Add enable-prefix-caching if not explicitly set
    return " \\\n".join(parts)


def generate_tgi_command(card: ServingCard) -> str:
    """Generate a TGI launch command from a servingcard."""
    if not card.serving or not card.serving.engine_args:
        return (
            "# No engine_args in servingcard -- cannot generate TGI command\n"
            f"text-generation-launcher --model-id {card.model}"
        )

    args = card.serving.engine_args.copy()
    model_id = args.pop("model", card.model)

    parts = [f"text-generation-launcher --model-id {model_id}"]

    # Map common vLLM args to TGI equivalents
    tgi_map = {
        "quantization": "quantize",
        "max_model_len": "max-input-length",
        "max_num_seqs": "max-batch-size",
    }

    for key, value in args.items():
        tgi_key = tgi_map.get(key, key.replace("_", "-"))
        # Skip speculative decoding args -- TGI doesn't support them the same way
        if key in ("speculative_model", "num_speculative_tokens"):
            parts.append(f"  # --{tgi_key} {value}  # speculative decoding: adjust for TGI")
            continue
        if key in ("gpu_memory_utilization",):
            continue  # TGI doesn't have this flag
        if isinstance(value, bool):
            if value:
                parts.append(f"  --{tgi_key}")
        else:
            parts.append(f"  --{tgi_key} {value}")

    return " \\\n".join(parts)


def generate_launch_command(card: ServingCard, engine: str | None = None) -> str:
    """Generate a launch command for the given engine.

    If engine is None, infer from card.framework.
    """
    if engine is None:
        framework = card.framework.lower()
        if "vllm" in framework:
            engine = "vllm"
        elif "tgi" in framework or "text-generation" in framework:
            engine = "tgi"
        else:
            engine = "vllm"  # default

    if engine == "vllm":
        return generate_vllm_command(card)
    elif engine == "tgi":
        return generate_tgi_command(card)
    else:
        return f"# Unsupported engine: {engine}\n# Use --engine vllm or --engine tgi"
