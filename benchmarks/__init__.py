"""LCTL Benchmarking Suite v4.0.

Performance benchmarks for the LCTL event sourcing system.

Benchmarks included:
- Replay performance (100, 1000, 10000 events)
- Chain loading from JSON/YAML
- State computation at various points
- Bottleneck analysis
- Confidence timeline generation
- Chain diff comparison

Usage:
    python -m benchmarks.run_benchmarks
    python -m benchmarks.run_benchmarks --suite replay
    python -m benchmarks.run_benchmarks --output report.json
"""

from .fixtures import (
    ChainSize,
    generate_chain,
    generate_chain_with_confidence_changes,
    generate_chain_with_errors,
    generate_divergent_chains,
)

__version__ = "4.0.0"

__all__ = [
    "ChainSize",
    "generate_chain",
    "generate_chain_with_confidence_changes",
    "generate_chain_with_errors",
    "generate_divergent_chains",
]
