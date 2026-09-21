"""Deterministic preparation, splitting, and export of product datasets."""

import random
import numpy as np

from .config import CATEGORIES


def load_categories(categories=CATEGORIES, workers=1):
    """Load one category at a time; use one worker for in-kernel notebooks."""
    from pricer.loaders import ItemLoader

    items = []
    for index, category in enumerate(categories, 1):
        print(f"[{index}/{len(categories)}] {category}", flush=True)
        items.extend(ItemLoader(category).load(workers=workers))
    return items


def deduplicate(items, seed=42):
    """Preserve the original shuffle, then title-pass, then full-text-pass order."""
    result = list(items)
    random.Random(seed).shuffle(result)
    for field in ("title", "full"):
        seen = set()
        unique = []
        for item in result:
            value = getattr(item, field)
            if value not in seen:
                seen.add(value)
                unique.append(item)
        result = unique
    return result


def weighted_sample(items, size=820_000, seed=42):
    """Original price-squared weighting and category penalties, without global RNG state."""
    if not items or size <= 0 or size > len(items):
        raise ValueError("Sample size must be positive and no larger than the item count")
    prices = np.array([item.price for item in items], dtype=float)
    if not np.isfinite(prices).all():
        raise ValueError("Prices must be finite")
    categories = np.array([item.category for item in items])
    weights = ((prices - prices.min()) / (prices.max() - prices.min() + 1e-9)) ** 2
    weights[categories == "Tools_and_Home_Improvement"] *= 0.5
    weights[categories == "Automotive"] *= 0.05
    if np.count_nonzero(weights) < size:
        raise ValueError("Not enough positive-weight items; reduce sample size or change the dataset")
    weights /= weights.sum()
    indices = np.random.RandomState(seed).choice(len(items), size=size, replace=False, p=weights)
    sample = [items[index] for index in indices]
    random.Random(seed).shuffle(sample)
    return sample


def split_items(items, sizes=(800_000, 10_000, 10_000)):
    if len(sizes) != 3 or any(size <= 0 for size in sizes) or sum(sizes) != len(items):
        raise ValueError("Train/validation/test sizes must be positive and sum to the item count")
    train_end, val_size, _ = sizes
    return items[:train_end], items[train_end:train_end + val_size], items[train_end + val_size:]


def lite_splits(splits):
    return tuple(part[:size] for part, size in zip(splits, (20_000, 1_000, 1_000)))


def summarized_copies(splits):
    """Strip raw fields on copies only, after every summary is present."""
    if any(not item.summary or not item.summary.strip() for part in splits for item in part):
        raise ValueError("Complete all summaries before exporting")
    return tuple([item.model_copy(update={"full": None, "id": None}) for item in part] for part in splits)
