"""Dependency-light data/model drift utilities for continuous evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], *, bins: int = 10
) -> float:
    """Measure distribution drift; >0.25 commonly deserves investigation."""
    if not reference or not current:
        raise ValueError("reference and current samples are required")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    combined_min = min(min(reference), min(current))
    combined_max = max(max(reference), max(current))
    if combined_min == combined_max:
        return 0.0
    width = (combined_max - combined_min) / bins

    def proportions(values: Sequence[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            index = min(int((value - combined_min) / width), bins - 1)
            counts[index] += 1
        return [max(count / len(values), 1e-6) for count in counts]

    expected = proportions(reference)
    actual = proportions(current)
    return sum((a - e) * math.log(a / e) for e, a in zip(expected, actual))


def drift_level(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "watch"
    return "drifted"
