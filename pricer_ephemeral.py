"""Price products with the local fine-tuned Qwen adapter."""

import re
from collections.abc import Callable


QUESTION = "What does this cost to the nearest dollar?"
PREFIX = "Price is $"


def build_prompt(description: str) -> str:
    """Build the exact prompt format used by the QLoRA training dataset."""
    return f"{QUESTION}\n\n{description.strip()}\n\n{PREFIX}"


def extract_price(reply: str) -> float:
    """Extract and validate the first positive price in a model completion."""
    cleaned = reply.replace("$", "").replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", cleaned)
    if not match:
        raise ValueError(f"Fine-tuned Qwen returned no numeric price: {reply!r}")
    result = float(match.group())
    if result <= 0:
        raise ValueError(f"Fine-tuned Qwen returned a non-positive price: {result}")
    return result


def price(description: str, generator: Callable[[str], str] | None = None) -> float:
    """Generate a price locally, loading the model lazily on the first call."""
    if generator is None:
        from llama import generate

        generator = generate
    return extract_price(generator(build_prompt(description)))
