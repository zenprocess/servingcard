"""CLI for servingcard — validate, info, search."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from servingcard.schema import ServingCard
from servingcard.validate import validate_card

app = typer.Typer(
    name="servingcard",
    help="Hardware-specific LLM serving configurations — model cards for serving.",
    no_args_is_help=True,
)


@app.command()
def validate(
    path: Path = typer.Argument(..., help="Path to a servingcard YAML file"),
) -> None:
    """Validate a servingcard YAML file."""
    errors = validate_card(path)
    if errors:
        typer.echo(f"INVALID: {path}")
        for err in errors:
            typer.echo(f"  - {err}")
        raise typer.Exit(code=1)
    else:
        typer.echo(f"VALID: {path}")


@app.command()
def info(
    path: Path = typer.Argument(..., help="Path to a servingcard YAML file"),
) -> None:
    """Display summary info for a servingcard."""
    card = ServingCard.from_yaml(path)

    typer.echo(f"Model:      {card.model}")
    typer.echo(f"Variant:    {card.variant}")
    typer.echo(f"Hardware:   {card.hardware}")
    typer.echo(f"Framework:  {card.framework}")
    typer.echo(f"Author:     {card.author}")
    typer.echo(f"Method:     {card.method}")

    if card.quantization:
        typer.echo(f"Quant:      {card.quantization.method} ({card.quantization.bits}-bit)")

    if card.speculative_decoding:
        typer.echo(
            f"Speculative: {card.speculative_decoding.method} "
            f"({card.speculative_decoding.draft_tokens} draft tokens)"
        )

    if card.benchmark:
        typer.echo("")
        typer.echo("Benchmark:")
        if card.benchmark.single_stream:
            typer.echo(f"  Single stream: {card.benchmark.single_stream.tok_s} tok/s")
        if card.benchmark.parallel:
            typer.echo(
                f"  Parallel peak: {card.benchmark.parallel.peak_tok_s} tok/s "
                f"@ {card.benchmark.parallel.peak_concurrency} concurrent"
            )
        if card.benchmark.single_stream and card.benchmark.single_stream.ttft_ms:
            typer.echo(f"  TTFT:          {card.benchmark.single_stream.ttft_ms} ms")

    if card.capacity:
        typer.echo("")
        typer.echo("Capacity:")
        typer.echo(f"  Context:       {card.capacity.context_limit:,} tokens")
        if card.capacity.max_concurrent:
            typer.echo(f"  Max concurrent: {card.capacity.max_concurrent}")
        if card.capacity.gpu_memory_utilization:
            typer.echo(f"  GPU mem util:  {card.capacity.gpu_memory_utilization}")


@app.command()
def search(
    model: str | None = typer.Option(None, help="Filter by model name"),
    hardware: str | None = typer.Option(None, help="Filter by hardware"),
    registry: Path = typer.Option(
        Path("registry"),
        help="Path to the registry directory",
    ),
) -> None:
    """Search for servingcards in the registry."""
    if not registry.exists():
        typer.echo(f"Registry not found: {registry}")
        raise typer.Exit(code=1)

    results: list[ServingCard] = []

    for yaml_path in sorted(registry.rglob("*.yaml")):
        try:
            with yaml_path.open() as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict) or "servingcard" not in data:
                continue
            card = ServingCard.model_validate(data)
        except Exception:
            continue

        if model and model.lower() not in card.model.lower():
            continue
        if hardware and hardware.lower() not in card.hardware.lower():
            continue

        results.append(card)

    if not results:
        typer.echo("No matching servingcards found.")
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(results)} servingcard(s):\n")
    for card in results:
        throughput = ""
        if card.benchmark and card.benchmark.single_stream:
            throughput = f" | {card.benchmark.single_stream.tok_s} tok/s"

        quant = ""
        if card.quantization:
            quant = f" | {card.quantization.method}"

        typer.echo(f"  {card.model}/{card.variant} on {card.hardware}{quant}{throughput}")


if __name__ == "__main__":
    app()
