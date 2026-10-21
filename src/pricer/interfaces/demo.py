"""Static, offline walkthrough of the deal review interface."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


DEMO_DATA = Path(__file__).resolve().parents[3] / "benchmarks" / "demo" / "opportunities.json"


def load_examples(path: Path = DEMO_DATA) -> list[dict]:
    """Load the small checked-in fixture; these values are explicitly illustrative."""
    return json.loads(path.read_text(encoding="utf-8"))


def rows(examples: list[dict]) -> list[list[str]]:
    return [
        [
            item["product_description"],
            f"${item['listing_price']:,.2f}",
            f"${item['illustrative_estimate']:,.2f}",
            f"${item['illustrative_estimate'] - item['listing_price']:,.2f}",
            f"{item['confidence']:.0%}",
            "Synthetic demo",
        ]
        for item in examples
    ]


def detail(item: dict) -> str:
    evidence = "\n".join(
        f"| {row['category']} | {row['description']} | ${row['price']:,.2f} |"
        for row in item["evidence"]
    ) or "| — | No comparable products in this example | — |"
    models = "\n".join(
        f"| {name.replace('_', ' ').title()} | ${value:,.2f} |"
        for name, value in item["model_estimates"].items()
    )
    return (
        "> **Synthetic walkthrough data.** Estimates and comparables are fabricated "
        "for demonstrating the review screen; they are not model output or evaluation results.\n\n"
        f"### {item['product_description']}\n\n"
        f"**Listing:** ${item['listing_price']:,.2f} · "
        f"**Illustrative estimate:** ${item['illustrative_estimate']:,.2f} · "
        f"**Illustrative confidence:** {item['confidence']:.0%}\n\n"
        f"**Review note:** {item['review_note']}\n\n"
        "| Example model | Illustrative estimate |\n|---|---:|\n"
        f"{models}\n\n"
        "| Category | Example comparable | Example price |\n|---|---|---:|\n"
        f"{evidence}\n\n"
        "Verify the exact product, condition, bundle, and final listing price before acting."
    )


def create_demo():
    examples = load_examples()
    with gr.Blocks(title="Pricer · Offline Demo") as app:
        gr.Markdown(
            "# Pricer · Offline demo\n"
            "A static walkthrough of deal triage and human review. No model server, "
            "product index, credentials, or network access is used. **All prices and "
            "comparables below are synthetic examples.**"
        )
        table = gr.Dataframe(
            value=rows(examples),
            headers=["Example listing", "Listing price", "Example estimate", "Difference", "Example confidence", "Data type"],
            interactive=False,
            wrap=True,
            max_height=360,
        )
        selected = gr.Markdown(detail(examples[0]))
        gr.Markdown("Select a row to inspect the illustrative model estimates and evidence.")

        def on_select(event: gr.SelectData):
            index = event.index[0] if isinstance(event.index, tuple) else event.index
            return detail(examples[index]) if 0 <= index < len(examples) else "Select a valid example."

        table.select(on_select, outputs=[selected])
        gr.Markdown(
            "The live application uses Ollama or vLLM, a Chroma product collection, "
            "the BM25 sidecar, and a trained neural checkpoint. See the README for "
            "the full setup and evidence limitations."
        )
    return app


def main() -> None:
    create_demo().launch(server_name="127.0.0.1", server_port=7860, share=False)
