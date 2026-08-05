from pricer.items import Item
import json
import re
from collections.abc import Mapping

from datasets import Dataset, Features, List, Sequence, Value
from huggingface_hub import hf_hub_download

MIN_CHARS = 600
MIN_PRICE = 0.5
MAX_PRICE = 999.49
MAX_TEXT_EACH = 3000
MAX_TEXT_TOTAL = 4000

REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]

AMAZON_REVIEWS_REPO = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_METADATA_FEATURES = Features(
    {
        "main_category": Value("string"),
        "title": Value("string"),
        "average_rating": Value("float64"),
        "rating_number": Value("int64"),
        "features": Sequence(Value("string")),
        "description": Sequence(Value("string")),
        "price": Value("string"),
        "images": List(
            {
                "hi_res": Value("string"),
                "large": Value("string"),
                "thumb": Value("string"),
                "variant": Value("string"),
            }
        ),
        "videos": List(
            {
                "title": Value("string"),
                "url": Value("string"),
                "user_id": Value("string"),
            }
        ),
        "store": Value("string"),
        "categories": Sequence(Value("string")),
        "details": Value("string"),
        "parent_asin": Value("string"),
        "bought_together": Value("string"),
        "subtitle": Value("string"),
        "author": Value("string"),
    }
)


def _json_string(value):
    if value is None or isinstance(value, str):
        return value
    return json.dumps(value)


def _amazon_metadata_rows(category: str):
    """Yield normalized rows without relying on the retired dataset script."""
    filename = f"raw/meta_categories/meta_{category}.jsonl"
    filepath = hf_hub_download(
        repo_id=AMAZON_REVIEWS_REPO,
        filename=filename,
        repo_type="dataset",
    )
    with open(filepath, encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
                yield {
                    "main_category": row.get("main_category"),
                    "title": row.get("title"),
                    "average_rating": row.get("average_rating"),
                    "rating_number": row.get("rating_number"),
                    "features": row.get("features") or [],
                    "description": row.get("description") or [],
                    "price": None if row.get("price") is None else str(row["price"]),
                    "images": [
                        {key: image.get(key) for key in ("hi_res", "large", "thumb", "variant")}
                        for image in row.get("images") or []
                    ],
                    "videos": [
                        {key: video.get(key) for key in ("title", "url", "user_id")}
                        for video in row.get("videos") or []
                    ],
                    "store": row.get("store"),
                    "categories": row.get("categories") or [],
                    "details": _json_string(row.get("details")) or "{}",
                    "parent_asin": row.get("parent_asin"),
                    "bought_together": _json_string(row.get("bought_together")),
                    "subtitle": _json_string(row.get("subtitle")),
                    "author": _json_string(row.get("author")),
                }
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                continue


def load_amazon_metadata(category: str) -> Dataset:
    """Load one Amazon metadata category using a stable, script-free schema."""
    return Dataset.from_generator(
        _amazon_metadata_rows,
        gen_kwargs={"category": category},
        features=AMAZON_METADATA_FEATURES,
    )

def simplify(text_list) -> str:
    """
    Return a simplified string without too much whitespace and limited to MAX_TEXT characters
    """
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )

def scrub(title, description, features, details) -> str:
    """
    Return a cleansed full string with product numbers and unimportant details removed
    """
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]

def get_weight(details: Mapping | None) -> float:
    """Return an item's weight in pounds, or zero for missing/malformed data."""
    if not isinstance(details, Mapping):
        return 0.0

    weight = details.get("Item Weight")
    if not isinstance(weight, (str, int, float)):
        return 0.0

    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s+"
        r"(pounds?|lbs?|ounces?|oz|grams?|g|milligrams?|mg|kilograms?|kg)\.?\s*",
        str(weight),
        flags=re.IGNORECASE,
    )
    if match:
        amount = float(match.group(1))
        unit = match.group(2).lower()
        if unit in {"pound", "pounds", "lb", "lbs"}:
            return amount
        if unit in {"ounce", "ounces", "oz"}:
            return amount / 16
        if unit in {"gram", "grams", "g"}:
            return amount / 453.592
        if unit in {"milligram", "milligrams", "mg"}:
            return amount / 453_592
        if unit in {"kilogram", "kilograms", "kg"}:
            return amount / 0.453592

    hundredths = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s+hundredths(?:\s+of)?\s+pounds?\s*",
        str(weight),
        flags=re.IGNORECASE,
    )
    if hundredths:
        return float(hundredths.group(1)) / 100

    return 0


def _parse_details(raw_details) -> dict:
    """Normalize a raw details value to a mutable dictionary."""
    if isinstance(raw_details, str):
        try:
            raw_details = json.loads(raw_details)
        except (json.JSONDecodeError, TypeError):
            return {}
    if not isinstance(raw_details, Mapping):
        return {}
    return dict(raw_details)

def parse(datapoint, category):
    try:
        price = float(datapoint["price"])
    except (TypeError, ValueError):
        return None
    if MIN_PRICE <= price <= MAX_PRICE:
        title = datapoint["title"]
        description = datapoint["description"]
        features = datapoint["features"]
        details = _parse_details(datapoint.get("details"))
        weight = get_weight(details)
        full = scrub(title, description, features, details)
        if len(full) >= MIN_CHARS:
            return Item(
                title=title,
                category=category,
                price=price,
                full=full,
                weight=weight,
            )
