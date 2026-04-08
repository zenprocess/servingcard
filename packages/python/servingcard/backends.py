"""Pluggable benchmark backends for servingcard."""

from __future__ import annotations

import json
import shutil
import subprocess
from abc import ABC, abstractmethod


class BenchmarkBackend(ABC):
    """Interface for benchmark harnesses."""

    @abstractmethod
    def run(self, endpoint: str, model: str, **kwargs: object) -> dict:
        """Run benchmarks, return results dict.

        Returns a dict with keys:
            single_stream_tok_s, ttft_ms, quality_score, cacp_compliance,
            parallel_peak_tok_s (optional), peak_concurrency (optional),
            useful_token_ratio (optional), tokens_per_turn (optional),
            adaptability_score (optional), suite (optional).
        """
        raise NotImplementedError


class PawBenchBackend(BenchmarkBackend):
    """PawBench integration -- subprocess first, then Python import."""

    @staticmethod
    def is_available() -> bool:
        """Check if PawBench is installed."""
        if shutil.which("pawbench"):
            return True
        try:
            import importlib

            importlib.import_module("pawbench")
            return True
        except ImportError:
            return False

    def run(self, endpoint: str, model: str, **kwargs: object) -> dict:
        """Run PawBench against an endpoint."""
        # Try subprocess first (works if pawbench is a CLI tool)
        pawbench_bin = shutil.which("pawbench")
        if pawbench_bin:
            return self._run_subprocess(endpoint, model, **kwargs)
        # Fall back to Python import
        return self._run_python(endpoint, model, **kwargs)

    def _run_subprocess(self, endpoint: str, model: str, **kwargs: object) -> dict:
        """Run PawBench via subprocess."""
        cmd = [
            "pawbench",
            "run",
            "--endpoint",
            endpoint,
            "--model",
            model,
            "--output-json",
            "-",
        ]
        suite = kwargs.get("suite")
        if suite:
            cmd.extend(["--suite", str(suite)])

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"PawBench failed (exit {result.returncode}): {result.stderr}")
        return json.loads(result.stdout)  # type: ignore[no-any-return]

    def _run_python(self, endpoint: str, model: str, **kwargs: object) -> dict:
        """Run PawBench via Python API."""
        try:
            from pawbench import run_benchmark  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "PawBench not found. Install: pip install pawbench"
            ) from None

        results: dict = run_benchmark(endpoint=endpoint, model=model, **kwargs)
        return results


class ManualBackend(BenchmarkBackend):
    """Manual entry -- user provides benchmark numbers interactively."""

    def run(self, endpoint: str, model: str, **kwargs: object) -> dict:
        """Prompt the user for benchmark results."""
        print("\nPawBench not found. Enter benchmark results manually:\n")

        tok_s = self._prompt_float("  Single-stream tok/s: ")
        ttft_ms = self._prompt_float("  TTFT (ms): ")
        quality = self._prompt_float("  Quality score (0-1): ", min_val=0, max_val=1)
        cacp = self._prompt_float("  CACP compliance (0-1): ", min_val=0, max_val=1)

        parallel_tok_s_str = input("  Parallel peak tok/s (Enter to skip): ").strip()
        concurrency_str = input("  Peak concurrency (Enter to skip): ").strip()

        result: dict = {
            "single_stream_tok_s": tok_s,
            "ttft_ms": ttft_ms,
            "quality_score": quality,
            "cacp_compliance": cacp,
            "suite": "manual",
        }
        if parallel_tok_s_str:
            result["parallel_peak_tok_s"] = float(parallel_tok_s_str)
        if concurrency_str:
            result["peak_concurrency"] = int(concurrency_str)

        return result

    @staticmethod
    def _prompt_float(
        prompt: str, min_val: float | None = None, max_val: float | None = None
    ) -> float:
        """Prompt for a float value with optional range validation."""
        while True:
            try:
                val = float(input(prompt))
                if min_val is not None and val < min_val:
                    print(f"    Must be >= {min_val}")
                    continue
                if max_val is not None and val > max_val:
                    print(f"    Must be <= {max_val}")
                    continue
                return val
            except ValueError:
                print("    Enter a number.")


def get_backend() -> BenchmarkBackend:
    """Return the best available benchmark backend."""
    if PawBenchBackend.is_available():
        return PawBenchBackend()
    return ManualBackend()
