"""Shared dataset names and the original experiment sizes."""

from dataclasses import dataclass

CATEGORIES = (
    "Automotive", "Electronics", "Office_Products", "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories", "Toys_and_Games", "Appliances", "Musical_Instruments",
)


@dataclass(frozen=True)
class DatasetConfig:
    owner: str = "ed-donner"
    lite: bool = False

    def repo(self, stage="summaries"):
        names = {"raw": "items_raw", "summaries": "items", "prompts": "items_prompts"}
        return f"{self.owner}/{names[stage]}_{'lite' if self.lite else 'full'}"

    @property
    def split_sizes(self):
        return (20_000, 1_000, 1_000) if self.lite else (800_000, 10_000, 10_000)
