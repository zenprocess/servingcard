"""CLI for servingcard -- benchmark, apply, validate, info, search."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer
import yaml

from servingcard.apply import (
    generate_launch_command,
    resolve_source,
)
from servingcard.schema import (
    BenchmarkEntry,
    BenchmarkSection,
    CapacitySection,
    PawBenchResults,
    ServingCard,
    ServingSection,
)
from servingcard.validate import validate_card

app = typer.Typer(
    name="servingcard",
    help="Make your model faster. Benchmark, share, and apply optimized LLM serving configs.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    model: str = typer.Option(..., "--model", "-m", help="Model identifier (e.g. qwen3-coder)"),
    hardware: str = typer.Option(..., "--hardware", "-h", help="Hardware slug (e.g. nvidia-gb10)"),
    endpoint: str = typer.Option(
        "http://localhost:8000", "--endpoint", "-e", help="vLLM/TGI endpoint URL"
    ),
    variant: str = typer.Option(
        None, "--variant", "-v", help="Config variant name (auto-generated if omitted)"
    ),
    author: str = typer.Option("unknown", "--author", "-a", help="Card author"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output YAML path (auto-generated if omitted)"
    ),
    framework: str = typer.Option("vllm", "--framework", "-f", help="Serving framework"),
    suite: str = typer.Option("full", "--suite", "-s", help="PawBench suite: quick | standard | full"),
) -> None:
    """Run PawBench (or enter results manually) and produce a servingcard."""
    from servingcard.backends import PawBenchBackend, get_backend

    # Select backend
    backend = get_backend()
    backend_name = type(backend).__name__

    if isinstance(backend, PawBenchBackend):
        typer.echo(f"Running PawBench ({suite}) against {endpoint} ...")
    else:
        typer.echo("PawBench not found. Falling back to manual entry.")

    # Run benchmark
    try:
        results = backend.run(endpoint=endpoint, model=model, suite=suite)
    except RuntimeError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        typer.echo("\nAborted.")
        raise typer.Exit(code=1) from None

    # Build PawBenchResults
    pawbench = PawBenchResults(
        suite=results.get("suite", suite),
        single_stream_tok_s=results["single_stream_tok_s"],
        parallel_peak_tok_s=results.get("parallel_peak_tok_s"),
        peak_concurrency=results.get("peak_concurrency"),
        ttft_ms=results["ttft_ms"],
        quality_score=results["quality_score"],
        cacp_compliance=results["cacp_compliance"],
        useful_token_ratio=results.get("useful_token_ratio"),
        tokens_per_turn=results.get("tokens_per_turn"),
        adaptability_score=results.get("adaptability_score"),
    )

    # Build benchmark section from results
    bench = BenchmarkSection(
        single_stream=BenchmarkEntry(
            tok_s=pawbench.single_stream_tok_s,
            ttft_ms=pawbench.ttft_ms,
        ),
    )
    if pawbench.parallel_peak_tok_s:
        bench.parallel = BenchmarkEntry(
            peak_tok_s=pawbench.parallel_peak_tok_s,
            peak_concurrency=pawbench.peak_concurrency,
        )

    # Auto-generate variant if not provided
    if not variant:
        variant = f"{hardware}-{backend_name.lower()}"

    # Build servingcard
    card = ServingCard(
        servingcard="0.1",
        model=model,
        variant=variant,
        hardware=hardware,
        framework=framework,
        author=author,
        created=date.today().isoformat(),
        method=f"pawbench-{suite}" if backend_name == "PawBenchBackend" else "manual",
        benchmark=bench,
        pawbench=pawbench,
        capacity=CapacitySection(context_limit=131072),
        serving=ServingSection(engine_args={"model": model}),
    )

    # Determine output path
    if output is None:
        output = Path(f"{model}-{hardware}.yaml")

    # Write YAML
    card_yaml = card.to_yaml()
    output.write_text(card_yaml)

    # Validate
    errors = validate_card(output)
    if errors:
        typer.echo(f"\nGenerated {output} (with warnings):")
        for err in errors:
            typer.echo(f"  - {err}")
    else:
        typer.echo(f"\nGenerated servingcard: {output}")

    # Print summary
    tok_s = pawbench.single_stream_tok_s
    quality = pawbench.quality_score
    cacp = pawbench.cacp_compliance
    ttft = pawbench.ttft_ms
    typer.echo(f"  {tok_s} tok/s | TTFT {ttft}ms | quality {quality} | CACP {cacp * 100:.0f}%")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@app.command()
def apply(
    source: str = typer.Argument(..., help="Config path, registry shorthand (model/variant), or URL"),
    engine: Optional[str] = typer.Option(None, "--engine", help="Override engine: vllm | tgi"),
    execute: bool = typer.Option(False, "--execute", help="Actually run the command (with confirmation)"),
) -> None:
    """Pull a servingcard and generate the launch command."""
    resolved = resolve_source(source)

    # Fetch or read the card
    if resolved.startswith("http://") or resolved.startswith("https://"):
        card = _fetch_remote_card(resolved)
    else:
        path = Path(resolved)
        if not path.exists():
            typer.echo(f"File not found: {path}", err=True)
            raise typer.Exit(code=1)
        card = ServingCard.from_yaml(path)

    # Print card summary
    typer.echo(f"Model:    {card.model}")
    typer.echo(f"Variant:  {card.variant}")
    typer.echo(f"Hardware: {card.hardware}")
    if card.benchmark and card.benchmark.single_stream:
        typer.echo(f"Perf:     {card.benchmark.single_stream.tok_s} tok/s")
    typer.echo("")

    # Generate launch command
    cmd = generate_launch_command(card, engine=engine)
    typer.echo(cmd)

    if execute:
        typer.echo("")
        confirm = typer.confirm("Execute this command?")
        if confirm:
            import subprocess

            typer.echo("\nLaunching...\n")
            # Run the first line to get the actual command, flatten for shell
            flat_cmd = cmd.replace(" \\\n", " ")
            subprocess.run(flat_cmd, shell=True, check=False)
        else:
            typer.echo("Aborted.")


def _fetch_remote_card(url: str) -> ServingCard:
    """Fetch a servingcard from a URL."""
    try:
        from urllib.request import Request, urlopen

        req = Request(url, headers={"User-Agent": "servingcard-cli/0.1"})
        with urlopen(req, timeout=15) as resp:
            data = yaml.safe_load(resp.read().decode())
    except Exception as exc:
        typer.echo(f"Failed to fetch {url}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    return ServingCard.model_validate(data)


# ---------------------------------------------------------------------------
# validate (kept)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# info (kept)
# ---------------------------------------------------------------------------


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
        elif card.benchmark.latency and card.benchmark.latency.ttft_ms:
            typer.echo(f"  TTFT:          {card.benchmark.latency.ttft_ms} ms")

    if card.pawbench:
        typer.echo("")
        typer.echo("PawBench:")
        typer.echo(f"  Suite:         {card.pawbench.suite}")
        typer.echo(f"  Quality:       {card.pawbench.quality_score}")
        typer.echo(f"  CACP:          {card.pawbench.cacp_compliance * 100:.0f}%")

    if card.capacity:
        typer.echo("")
        typer.echo("Capacity:")
        typer.echo(f"  Context:       {card.capacity.context_limit:,} tokens")
        if card.capacity.max_concurrent:
            typer.echo(f"  Max concurrent: {card.capacity.max_concurrent}")
        if card.capacity.gpu_memory_utilization:
            typer.echo(f"  GPU mem util:  {card.capacity.gpu_memory_utilization}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: Optional[str] = typer.Argument(None, help="Search term (model name, hardware, etc.)"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model name"),
    hardware: Optional[str] = typer.Option(None, "--hardware", help="Filter by hardware"),
    registry: Path = typer.Option(
        Path("registry"),
        "--registry",
        help="Path to the local registry directory",
    ),
) -> None:
    """Search for servingcards in the registry."""
    if not registry.exists():
        typer.echo(f"Registry not found: {registry}")
        typer.echo("Hint: run from the servingcard repo root, or use --registry <path>")
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

        # Apply filters
        if model and model.lower() not in card.model.lower():
            continue
        if hardware and hardware.lower() not in card.hardware.lower():
            continue
        if query:
            q = query.lower()
            searchable = f"{card.model} {card.variant} {card.hardware} {card.framework}".lower()
            if q not in searchable:
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
