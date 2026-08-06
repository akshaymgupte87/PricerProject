"""Deterministic safety checks for model-produced prices."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence


PRICE_PATTERN = re.compile(r"(?<![\w.])\$?\s*(\d[\d,]*(?:\.\d+)?)")


class PriceGuardrailError(ValueError):
    pass


def validate_price(value: float, *, minimum: float = 0.01, maximum: float = 1_000_000) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise PriceGuardrailError(
            f"price {result!r} is outside the allowed range ${minimum:g}-${maximum:g}"
        )
    return result


def extract_single_price(text: str) -> float:
    values = [float(match.replace(",", "")) for match in PRICE_PATTERN.findall(text or "")]
    distinct = list(dict.fromkeys(values))
    if not distinct:
        raise PriceGuardrailError(f"model returned no numeric price: {text!r}")
    if len(distinct) > 1:
        raise PriceGuardrailError(f"model returned multiple conflicting prices: {text!r}")
    return validate_price(distinct[0])


def relative_spread(values: Sequence[float]) -> float:
    checked = [validate_price(value) for value in values]
    if not checked:
        raise PriceGuardrailError("at least one price estimate is required")
    midpoint = sorted(checked)[len(checked) // 2]
    return (max(checked) - min(checked)) / midpoint
