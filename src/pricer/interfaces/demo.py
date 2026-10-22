"""Guided replay of a captured live deal-pricing run."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import gradio as gr


CAPTURE_DATA = Path(__file__).resolve().parents[3] / "benchmarks" / "demo" / "captured-live-run.json"


def load_capture(path: Path = CAPTURE_DATA) -> dict:
    """Load the checked-in, historical run capture; this does not call any models."""
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_opportunities(capture: dict) -> list[dict]:
    """Map saved Opportunity records to the walkthrough's presentation fields."""
    result = []
    for record in capture["opportunities"]:
        opportunity = record["opportunity"]
        deal = opportunity["deal"]
        result.append(
            {
                "product_description": deal["product_description"],
                "listing_price": deal["price"],
                "estimate": opportunity["estimate"],
                "discount": opportunity["discount"],
                "confidence": opportunity["confidence"],
                "confidence_low": opportunity.get("confidence_low"),
                "confidence_high": opportunity.get("confidence_high"),
                "model_estimates": opportunity.get("model_estimates", {}),
                "evidence": opportunity.get("evidence", []),
                "explanation": opportunity.get("explanation", ""),
                "url": deal["url"],
                "saved_status": record["status"],
                "created_at": record["created_at"],
            }
        )
    return result


def rows(opportunities: list[dict]) -> list[list[str]]:
    return [
        [
            textwrap.shorten(item["product_description"], width=94, placeholder="…"),
            f"${item['listing_price']:,.2f}",
            f"${item['estimate']:,.2f}",
            f"${item['discount']:+,.2f}",
            f"{item['confidence']:.0%} · heuristic",
            (
                f"{item['review_decision']} · correction ${item['reviewed_price']:,.2f} · replay"
                if item.get("reviewed_price") is not None
                else item.get("review_decision", "Pending · captured run")
            ),
        ]
        for item in opportunities
    ]


def status_summary(item: dict, capture: dict) -> str:
    discount = item["discount"]
    estimate = item["estimate"]
    discount_ratio = discount / estimate if estimate > 0 else 0.0
    qualifies = (
        discount > capture["decision"]["discount_threshold_usd"]
        and discount_ratio >= capture["decision"]["minimum_discount_fraction_of_estimate"]
    )
    if qualifies:
        automatic_result = "Flagged for human review"
    else:
        automatic_result = "Not flagged for an alert"
    review = item.get("review_decision")
    review_status = f"**{review}** · replay only" if review else "**Pending** · awaiting reviewer"
    comparison = (
        f"the estimate is ${abs(discount):,.2f} below asking"
        if discount < 0
        else f"the estimate is ${discount:,.2f} above asking"
    )
    return (
        "### Deal status\n\n"
        f"**Feed scan:** Complete · Qwen selected this deal from {capture['scan']['unseen_deals']} unseen listings  \n"
        f"**Price assessment:** Complete · asking ${item['listing_price']:,.2f}, estimate ${estimate:,.2f}; {comparison}  \n"
        f"**Automatic screen:** {automatic_result} · the rule requires a gap over "
        f"${capture['decision']['discount_threshold_usd']:.0f} and at least "
        f"{capture['decision']['minimum_discount_fraction_of_estimate']:.0%} of estimated value  \n"
        f"**Human review:** {review_status}"
    )


def _evidence_title(description: str) -> str:
    first_line = (description or "").splitlines()[0].strip()
    return first_line.removeprefix("Title: ") or "Comparable product"


def detail(item: dict, capture: dict) -> str:
    estimate = item["estimate"]
    models = item["model_estimates"]
    rag = models.get("frontier_rag", 0.0)
    specialist = models.get("specialist", 0.0)
    neural = models.get("neural_network", 0.0)
    evidence_rows = "\n".join(
        f"| {_evidence_title(row.get('description', ''))} | "
        f"{str(row.get('category') or '—').replace('_', ' ')} | "
        f"${float(row.get('price', 0)):,.2f} | {float(row.get('score', 0)):.3f} |"
        for row in item["evidence"]
    ) or "| No comparable products recorded | — | — | — |"
    low = item.get("confidence_low")
    high = item.get("confidence_high")
    review_status = item.get("review_decision")
    review_section = (
        f"**Status changed:** Pending → **{review_status}**  \n"
        f"**Reviewer reason:** {item.get('review_reason', 'Not provided')}  \n"
        "This update exists only in the current browser session.\n\n"
        if review_status
        else "**Review status:** Pending — this captured opportunity is waiting for a person.\n\n"
    )
    correction_section = (
        f"**Reviewer fair-price label:** ${item['reviewed_price']:,.2f} &nbsp; · &nbsp; "
        f"**Label minus asking price:** ${item['reviewed_price'] - item['listing_price']:+,.2f}\n\n"
        "The original model estimate above is preserved; this is a separate human label.\n\n"
        if item.get("reviewed_price") is not None
        else ""
    )
    estimate_interval = (
        f"**Recorded interval:** ${low:,.2f}–${high:,.2f} (heuristic)  \n"
        if low is not None and high is not None
        else ""
    )
    return (
        f"### {item['product_description']}\n\n"
        f"[Open the original DealNews listing ↗]({item['url']})\n\n"
        f"**Asking price:** ${item['listing_price']:,.2f} &nbsp; · &nbsp; "
        f"**Ensemble estimate:** ${estimate:,.2f} &nbsp; · &nbsp; "
        f"**Estimate minus asking price:** ${item['discount']:+,.2f}\n\n"
        f"**Recorded confidence signal:** {item['confidence']:.0%} (agreement/evidence heuristic, not measured accuracy)  \n"
        f"{estimate_interval}"
        f"**Run outcome:** {'This deal did not clear the alert threshold.' if item['discount'] <= 0 else 'This deal had a positive estimate gap; the run-level alert decision is shown below.'}\n\n"
        f"{review_section}{correction_section}"
        "#### Three price signals and the ensemble\n\n"
        "| Signal | What it represents | Captured price |\n|---|---|---:|\n"
        f"| Retrieved products + Qwen | Retrieval-grounded price estimate | ${rag:,.2f} |\n"
        f"| Specialist · {capture['pricing']['specialist_backend']} | Direct product-price estimate | ${specialist:,.2f} |\n"
        f"| Neural network | PyTorch model estimate | ${neural:,.2f} |\n\n"
        f"The ensemble used **80% retrieval-grounded + 10% specialist + 10% neural**: "
        f"0.8 × ${rag:,.2f} + 0.1 × ${specialist:,.2f} + 0.1 × ${neural:,.2f} = **${estimate:,.2f}**.\n\n"
        "#### The five retrieved comparables\n\n"
        "The live run combined vector search and BM25 results, fused their rankings, then "
        "reranked candidates. The score below is a reranker score, not a probability.\n\n"
        "| Retrieved product | Category | Listed price in index | Reranker score |\n|---|---|---:|---:|\n"
        f"{evidence_rows}\n\n"
        f"> **Historical live output:** captured {item['created_at'][:10]} from this project's local run. "
        "It shows what the system produced then; it is not current market data or a pricing-quality evaluation."
    )


def create_demo():
    capture = load_capture()
    opportunities = normalize_opportunities(capture)
    if not opportunities:
        raise ValueError("The captured live-run replay contains no opportunities.")
    first_index = next(
        (i for i, item in enumerate(opportunities) if "dji osmo nano" in item["product_description"].casefold()),
        0,
    )

    with gr.Blocks(title="Pricer · AI deal-pricing walkthrough") as app:
        session_opportunities = gr.State(opportunities)
        selected_index = gr.State(first_index)

        gr.Markdown(
            "# Pricer\n"
            "### A real deal scan, from RSS feed to human review\n\n"
            "**Recorded live run · October 1, 2026 · local Ollama + Qwen 3.6**  \n"
            "This is a replay of saved model outputs and retrieved evidence, not a fresh scan. "
            "The example prices and product matches below came from an actual run in this project."
        )
        gr.Markdown(
            f"**The question:** Is this listing a good deal?  \n"
            f"In this recorded run, Qwen selected {capture['scan']['selected_deals']} deals from "
            f"{capture['scan']['unseen_deals']} unseen feed listings. No deal cleared the alert rules. "
            "Start with one example, inspect the evidence, then record a reviewer decision."
        )

        status = gr.Markdown(status_summary(opportunities[first_index], capture))
        selected = gr.Markdown(detail(opportunities[first_index], capture))

        with gr.Accordion("Explore the other four deals from this scan", open=False):
            table = gr.Dataframe(
                value=rows(opportunities),
                headers=[
                    "Deal selected by Qwen",
                    "Asking price",
                    "Ensemble estimate · captured",
                    "Estimate gap · captured",
                    "Confidence signal",
                    "Review status · replay",
                ],
                interactive=False,
                wrap=True,
                max_height=360,
            )

        gr.Markdown("#### Follow the engineering stages")
        with gr.Accordion("How this result was produced", open=False):
            gr.Markdown(
                "1. **Ingest:** read DealNews electronics, computers, and smart-home RSS feeds; scrape deal pages; filter seen URLs.\n"
                "2. **Select and structure:** local Qwen 3.6 chose five of 29 unseen deals and returned structured product descriptions, prices, and URLs.\n"
                "3. **Prepare and retrieve:** Qwen rewrote each product description. MiniLM embeddings searched Chroma while SQLite FTS5 BM25 searched keywords. Reciprocal-rank fusion combined results; a MiniLM cross-encoder reranked the top candidates.\n"
                "4. **Estimate:** a retrieval-grounded Qwen estimate, a specialist Qwen estimate, and a PyTorch neural estimate were combined at 80/10/10. Disagreement and evidence count informed a heuristic confidence signal.\n"
                "5. **Decide:** the planner compared estimated value with asking price and applied its $50 and 20% thresholds. A person remains responsible for reviewing the evidence and any correction."
            )

        gr.Markdown("#### Your turn: review one captured deal")
        gr.Markdown(
            "1. Inspect the original listing, price signals, and comparable products above.\n"
            "2. Choose **Approve for follow-up** or **Reject**. Enter a reason of at least 10 characters.\n"
            "3. If you have independently checked a fair price, enter it as an optional correction.\n"
            "4. Apply the review. Watch the row move from **Pending** to your decision, and see the reason "
            "and corrected-price label appear in the deal details.\n\n"
            "This replay keeps those changes in the current browser session only. In the live dashboard, "
            "the decision and optional corrected label are persisted; a correction can later support evaluation "
            "and an explicit DPO training job."
        )
        with gr.Row():
            decision = gr.Radio(
                ["Approve for follow-up", "Reject"],
                label="Your review choice",
                value="Approve for follow-up",
            )
            corrected_price = gr.Number(label="Corrected fair price (optional)", minimum=0)
        review_reason = gr.Textbox(
            label="Reason · required (at least 10 characters)",
            placeholder="Example: Comparable bundle has fewer batteries than this listing.",
            lines=2,
        )
        review_button = gr.Button("Apply review to replay", variant="primary")
        review_result = gr.Markdown("Your replay decision will appear here.")

        def apply_review(
            decision_value: str,
            price: float | None,
            reason: str,
            index: int,
            current: list[dict],
        ):
            current = [dict(item) for item in current]
            if not 0 <= index < len(current):
                return "Select a valid captured deal.", "Select a valid captured deal.", "Select a valid captured deal.", rows(current), current, index
            item = current[index]
            if len((reason or "").strip()) < 10:
                return (
                    "Add a brief reason of at least 10 characters before applying the review.",
                    detail(item, capture), status_summary(item, capture), rows(current), current, index,
                )
            if price is not None and price <= 0:
                return (
                    "A corrected fair price must be greater than zero.",
                    detail(item, capture), status_summary(item, capture), rows(current), current, index,
                )
            item.pop("reviewed_price", None)
            item["review_decision"] = (
                "Approved · replay only"
                if decision_value == "Approve for follow-up"
                else "Rejected · replay only"
            )
            item["review_reason"] = reason.strip()
            if price is not None:
                item["reviewed_price"] = price
            message = (
                f"**Status updated in this replay:** Pending → {item['review_decision']}. "
                "The original estimate remains unchanged. "
            )
            if price is not None:
                message += (
                    f"Your separate ${price:,.2f} correction is now visible as a human label. "
                )
            message += "The saved opportunity and model are unchanged."
            return message, detail(item, capture), status_summary(item, capture), rows(current), current, index

        review_button.click(
            apply_review,
            inputs=[decision, corrected_price, review_reason, selected_index, session_opportunities],
            outputs=[review_result, selected, status, table, session_opportunities, selected_index],
        )

        def on_select(event: gr.SelectData, current: list[dict]):
            index = event.index
            if isinstance(index, (tuple, list)):
                index = index[0] if index else -1
            if isinstance(index, bool) or not isinstance(index, int):
                return "Select a valid captured deal.", "Select a valid captured deal.", 0
            return (detail(current[index], capture), status_summary(current[index], capture), index) if 0 <= index < len(current) else ("Select a valid captured deal.", "Select a valid captured deal.", 0)

        table.select(on_select, inputs=[session_opportunities], outputs=[selected, status, selected_index])

        gr.Markdown(
            "### Where fine-tuning fits\n"
            "The captured scan used the regular Ollama Qwen backend; it did not load the fine-tuned adapter. "
            "Separately, corrected reviews can be exported as preference pairs and used in an explicit QLoRA DPO job. "
            "That training step writes a separate adapter and requires evaluation before anyone switches inference to it."
        )
        with gr.Accordion("Run the current live system", open=False):
            gr.Markdown(
                "`pricer demo` replays this capture without contacting feeds or models. For a fresh RSS scan and live inference, "
                "configure the Qwen-compatible Ollama/vLLM endpoint, product index, BM25 sidecar, and neural checkpoint, "
                "then run `uv run pricer dashboard` and choose **Run deal scan now**. The dashboard exposes current logs, "
                "retrieved evidence, model disagreement, and persistent approve/reject/correct feedback."
            )
    return app


def main() -> None:
    create_demo().launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
