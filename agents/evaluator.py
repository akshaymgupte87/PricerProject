"""Compatibility exports for the canonical :mod:`pricer.evaluator` module.

New code should import from ``pricer.evaluator``. This module remains so older
agent experiments and notebooks do not break.
"""

from pricer.evaluator import (
    COLOR_MAP,
    DEFAULT_SIZE,
    GREEN,
    RED,
    RESET,
    WORKERS,
    YELLOW,
    Tester,
    aggregate_retrieval_metrics,
    evaluate,
)

__all__ = [
    "COLOR_MAP",
    "DEFAULT_SIZE",
    "GREEN",
    "RED",
    "RESET",
    "WORKERS",
    "YELLOW",
    "Tester",
    "aggregate_retrieval_metrics",
    "evaluate",
]
