# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-03-27

### Added
- ServingCard YAML specification v1.0
- Python CLI: `servingcard benchmark`, `servingcard apply`, `servingcard validate`, `servingcard info`, `servingcard search`
- PawBench integration with manual fallback
- 3 seed configs: Qwen3-coder on NVIDIA GB10 (Eagle3, FP8 baseline, NVFP4)
- Registry at `registry/` with GitHub-based distribution
- Documentation: spec, getting-started, format-overview, benchmarks
- CI: test matrix (3.10/3.11/3.12), lint, coverage, registry validation
- PyPI publish workflow (trusted publishing on tag)
