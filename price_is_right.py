import html
import logging
import os
import queue
import threading
import textwrap
import time
import gradio as gr
from deal_agent_framework import CATEGORIES, COLORS, DealAgentFramework
from log_utils import reformat
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv(override=True)


APP_CSS = """
.gradio-container { max-width: 1500px !important; margin: 0 auto !important; }
.hero { padding: 22px 24px; border-radius: 18px; color: white;
        background: linear-gradient(120deg, #172554, #312e81 55%, #0f766e); }
.hero h1 { margin: 0 0 6px; font-size: 30px; letter-spacing: -0.03em; color: #fbbf24; }
.hero p { margin: 0; color: #dbeafe; }
.section-card { border: 1px solid var(--border-color-primary); border-radius: 16px;
                padding: 8px; background: var(--background-fill-primary); }
.approval-card { border: 3px solid #f59e0b !important; border-radius: 18px !important;
                 padding: 18px !important; margin-top: 12px;
                 background: color-mix(in srgb, #f59e0b 8%, var(--background-fill-primary)); }
.approval-card h2 { color: #b45309; margin-top: 0; }
.approval-instructions { font-size: 16px; line-height: 1.6; }
.status-line { font-size: 14px; }
footer { display: none !important; }
"""


class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


def html_for(log_data):
    output = "<br>".join(reformat(html.escape(item)) for item in log_data[-24:])
    return f"""
    <div id="scrollContent" style="height: 400px; overflow-y: auto; border-radius: 12px; background-color: #171923; padding: 14px; font-family: ui-monospace, monospace; font-size: 12px; line-height: 1.55; color: #e5e7eb;">
    {output}
    </div>
    """


def setup_logging(log_queue):
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def table_for(opportunities):
    """Convert domain models into the stable review-table contract."""
    return [
        [
            opportunity.deal.product_description,
            f"${opportunity.deal.price:,.2f}",
            f"${opportunity.estimate:,.2f}",
            f"${opportunity.discount:,.2f}",
            (
                f"{opportunity.discount / opportunity.estimate:.0%}"
                if opportunity.estimate > 0
                else "—"
            ),
            f"{opportunity.confidence:.0%}" if opportunity.confidence else "—",
            review_risk(opportunity)[0],
            opportunity.status.title(),
        ]
        for opportunity in opportunities
    ]


def review_risk(opportunity):
    """Return a reviewer-friendly risk level and concrete reasons."""
    reasons = []
    estimates = list(opportunity.model_estimates.values())
    if len(estimates) >= 2:
        midpoint = sorted(estimates)[len(estimates) // 2]
        spread = (max(estimates) - min(estimates)) / midpoint if midpoint else float("inf")
        if spread > 0.75:
            reasons.append(f"models disagree by {spread:.0%}")
    if opportunity.confidence < 0.6:
        reasons.append(f"low confidence ({opportunity.confidence:.0%})")
    if not opportunity.evidence:
        reasons.append("no retrieved comparables")
    elif len(opportunity.evidence) < 3:
        reasons.append(f"only {len(opportunity.evidence)} comparable(s)")
    if reasons:
        return "High", reasons
    if opportunity.confidence < 0.8:
        return "Medium", ["confidence is below 80%"]
    return "Low", ["models and retrieved evidence are reasonably consistent"]


def evaluation_markdown(summary):
    """Render aggregate human-review quality metrics without hiding small samples."""
    feedback = summary["feedback_count"]
    corrected = summary["corrected_price_count"]
    if not feedback:
        return (
            "### Evaluation results\n\n"
            "No human decisions yet. Review several deals and include corrected fair "
            "prices to measure pricing error."
        )
    mae = summary["mean_absolute_error"]
    rmse = summary["root_mean_squared_error"]
    bias = summary["mean_signed_error"]
    error_text = (
        f"**MAE:** ${mae:,.2f} &nbsp; **RMSE:** ${rmse:,.2f} &nbsp; "
        f"**Bias:** {bias:+,.2f}"
        if mae is not None
        else "**Pricing error:** unavailable until a corrected fair price is supplied"
    )
    return (
        "### Evaluation results\n\n"
        f"**{feedback} decision{'s' if feedback != 1 else ''}** — "
        f"{summary['approved_count']} approved / "
        f"{summary['rejected_count']} rejected ({summary['approval_rate']:.0%} approval)  \n"
        f"**{corrected} price labels** — {error_text}\n\n"
        "Positive bias means the model tends to overvalue products; negative bias means "
        "it tends to undervalue them. Treat results as preliminary until you have at "
        "least 30 corrected-price labels."
    )


def run_framework_worker(framework, result_queue):
    """Run the pipeline and always emit a terminal event for the UI."""
    try:
        result = framework.run(
            on_progress=lambda opportunities: result_queue.put(
                {"type": "progress", "table": table_for(opportunities)}
            )
        )
        result_queue.put({"type": "complete", "table": table_for(result)})
    except Exception as error:
        logging.exception("Deal pipeline failed")
        result_queue.put(
            {
                "type": "error",
                "message": f"{type(error).__name__}: {error}",
                "table": table_for(framework.display_opportunities()),
            }
        )


class App:
    def __init__(self):
        self.agent_framework = None

    def get_agent_framework(self):
        if not self.agent_framework:
            self.agent_framework = DealAgentFramework()
        return self.agent_framework

    def run(self):
        framework = self.get_agent_framework()
        initial_table = table_for(framework.display_opportunities())

        with gr.Blocks(
            title="The Price is Right",
            fill_width=True,
        ) as ui:
            log_data = gr.State([])

            def update_output(log_data, log_queue, result_queue):
                current_table = table_for(framework.display_opportunities())
                status_text = "⏳ **Run started** — scanning for current deals."
                yield log_data, html_for(log_data), current_table, status_text

                finished = False
                while not finished:
                    changed = False
                    while True:
                        try:
                            log_data.append(log_queue.get_nowait())
                            changed = True
                        except queue.Empty:
                            break

                    try:
                        event = result_queue.get(timeout=0.1)
                    except queue.Empty:
                        event = None

                    if event:
                        changed = True
                        current_table = event["table"]
                        if event["type"] == "progress":
                            status_text = (
                                f"⚙️ **Pricing in progress** — {len(current_table)} "
                                "opportunities visible."
                            )
                        elif event["type"] == "complete":
                            status_text = (
                                f"✅ **Run complete** — showing {len(current_table)} opportunities."
                            )
                            finished = True
                        elif event["type"] == "error":
                            message = event["message"]
                            log_data.append(f"Pipeline error: {message}")
                            status_text = f"❌ **Run failed** — `{message}`"
                            finished = True

                    if changed:
                        yield log_data, html_for(log_data), current_table, status_text

            def get_initial_plot():
                fig = go.Figure()
                fig.update_layout(
                    title="Loading vector DB...",
                    height=400,
                )
                return fig

            def get_plot():
                documents, vectors, colors = DealAgentFramework.get_plot_data(max_datapoints=800)

                def wrap_hover_text(document, width=68):
                    description = " ".join(str(document).split())
                    return "<br>".join(
                        html.escape(line)
                        for line in textwrap.wrap(
                            description,
                            width=width,
                            break_long_words=False,
                            break_on_hyphens=False,
                        )
                    )

                hover_labels = [
                    wrap_hover_text(document)
                    for document in documents
                ]

                # One trace per product category gives the plot a useful legend.
                fig = go.Figure()
                category_colors = list(zip(CATEGORIES, COLORS)) + [("Other", "gray")]
                for category, color in category_colors:
                    indices = [index for index, point_color in enumerate(colors) if point_color == color]
                    if not indices:
                        continue
                    fig.add_trace(
                        go.Scatter3d(
                            name=category.replace("_", " "),
                            x=vectors[indices, 0],
                            y=vectors[indices, 1],
                            z=vectors[indices, 2],
                            text=[hover_labels[index] for index in indices],
                            mode="markers",
                            marker=dict(size=3, color=color, opacity=0.72),
                            hovertemplate=(
                                "<b>Product description</b><br>%{text}<br><br>"
                                f"<b>Category:</b> {category.replace('_', ' ')}"
                                "<extra></extra>"
                            ),
                            hoverlabel=dict(align="left", font=dict(size=12)),
                        )
                    )

                fig.update_layout(
                    title=dict(
                        text=(
                            "Product similarity map"
                            "<br><sup>Nearby points have similar descriptions. "
                            "Hover over a point to identify the product; colors show categories.</sup>"
                        ),
                        x=0.02,
                    ),
                    scene=dict(
                        xaxis_title="t-SNE dimension 1",
                        yaxis_title="t-SNE dimension 2",
                        zaxis_title="t-SNE dimension 3",
                        aspectmode="manual",
                        aspectratio=dict(x=2.2, y=2.2, z=1),
                        camera=dict(eye=dict(x=1.6, y=1.6, z=0.8)),
                    ),
                    legend=dict(title="Product category", orientation="v"),
                    autosize=True,
                    height=600,
                    hovermode="closest",
                    margin=dict(r=10, b=10, l=10, t=75),
                )

                return fig

            def run_with_logging(initial_log_data):
                log_queue = queue.Queue()
                result_queue = queue.Queue()
                handler = setup_logging(log_queue)
                thread = threading.Thread(
                    target=run_framework_worker,
                    args=(framework, result_queue),
                    daemon=True,
                    name="deal-pipeline-worker",
                )
                thread.start()
                try:
                    yield from update_output(initial_log_data, log_queue, result_queue)
                finally:
                    logging.getLogger().removeHandler(handler)

            def history_plot(url: str):
                history = framework.price_history(url) if url else []
                chronological = list(reversed(history))
                figure = go.Figure()
                if chronological:
                    figure.add_trace(go.Scatter(
                        x=[item["observed_at"] for item in chronological],
                        y=[item["observed_price"] for item in chronological],
                        name="Observed price", mode="lines+markers",
                    ))
                    figure.add_trace(go.Scatter(
                        x=[item["observed_at"] for item in chronological],
                        y=[item["estimated_value"] for item in chronological],
                        name="Estimated value", mode="lines+markers",
                    ))
                figure.update_layout(title="Price history", height=300, margin=dict(t=45, b=30))
                return figure

            def do_select(selected_index: gr.SelectData):
                opportunities = framework.display_opportunities()
                row = selected_index.index[0] if isinstance(selected_index.index, tuple) else selected_index.index
                if row >= len(opportunities):
                    return "", "Select a valid opportunity.", [], go.Figure()
                opportunity = opportunities[row]
                insights = framework.store.price_insights(opportunity.deal.url)
                risk, risk_reasons = review_risk(opportunity)
                savings_percent = (
                    opportunity.discount / opportunity.estimate
                    if opportunity.estimate > 0
                    else 0
                )
                model_rows = "\n".join(
                    f"| {name.replace('_', ' ').title()} | ${value:,.2f} |"
                    for name, value in opportunity.model_estimates.items()
                ) or "| Legacy estimate | Not recorded |"
                evidence_rows = [
                    [
                        f"${float(item.get('price', 0)):,.2f}",
                        str(item.get("category") or "—").replace("_", " "),
                        str(item.get("description") or "")[:300],
                        (
                            f"{float(item['score']):.3f}"
                            if item.get("score") is not None
                            else "—"
                        ),
                    ]
                    for item in opportunity.evidence[:5]
                ]
                detail = (
                    f"### {risk} review risk · {opportunity.status.title()}\n\n"
                    f"[Open the original deal page ↗]({opportunity.deal.url}) and verify "
                    "the exact product, condition, bundle, coupon requirements, and final price.\n\n"
                    f"**Asking price:** ${opportunity.deal.price:,.2f} &nbsp; "
                    f"**Estimated value:** ${opportunity.estimate:,.2f} &nbsp; "
                    f"**Claimed savings:** ${opportunity.discount:,.2f} ({savings_percent:.0%})\n\n"
                    f"{opportunity.explanation or 'No explanation recorded.'}\n\n"
                    f"**Interval:** ${opportunity.confidence_low or opportunity.estimate:,.2f}–"
                    f"${opportunity.confidence_high or opportunity.estimate:,.2f}  \n"
                    f"**Risk flags:** {'; '.join(risk_reasons)}\n\n"
                    "| Model | Estimate |\n|---|---:|\n"
                    f"{model_rows}\n\n"
                    f"**Price signal:** {insights.get('recommendation', 'insufficient data').replace('_', ' ').title()} "
                    f"({insights.get('samples', 0)} observation(s)). This is supporting "
                    "context, not ground truth."
                )
                return (
                    opportunity.deal.url,
                    detail,
                    evidence_rows,
                    history_plot(opportunity.deal.url),
                )

            def review(url, decision, corrected_price, reason):
                if not url:
                    return (
                        table_for(framework.display_opportunities()),
                        "⚠️ Select a deal first.",
                        evaluation_markdown(framework.store.evaluation_summary()),
                    )
                if len((reason or "").strip()) < 10:
                    return (
                        table_for(framework.display_opportunities()),
                        "⚠️ Add a brief reason (at least 10 characters) so this decision is auditable.",
                        evaluation_markdown(framework.store.evaluation_summary()),
                    )
                try:
                    framework.submit_feedback(
                        url,
                        decision,
                        corrected_price=corrected_price,
                        reason=reason,
                    )
                except Exception as error:
                    return (
                        table_for(framework.display_opportunities()),
                        f"❌ Review failed: `{error}`",
                        evaluation_markdown(framework.store.evaluation_summary()),
                    )
                message = (
                    "✅ **Approved.** The local deal alert was published."
                    if decision == "approved"
                    else "🚫 **Rejected.** The decision was saved and no alert was sent."
                )
                return (
                    table_for(framework.display_opportunities()),
                    message,
                    evaluation_markdown(framework.store.evaluation_summary()),
                )

            def save_search(name, query, category):
                if not name or not query:
                    return "⚠️ A name and query are required."
                framework.store.save_search(
                    name, query, {"category": category} if category else {}
                )
                return f"✅ Saved watch **{name}**."

            with gr.Row():
                gr.Markdown(
                    '<div class="hero"><h1>The Price is Right</h1>'
                    '<p>A local-first autonomous deal intelligence system powered by Qwen, '
                    'retrieval-augmented pricing, and durable opportunity tracking.</p></div>'
                )
            with gr.Row():
                status = gr.Markdown(
                    f"🟢 **Ready** — loaded {len(initial_table)} saved opportunities.",
                    elem_classes=["status-line"],
                )
                run_button = gr.Button("Run deal scan now", variant="primary")
            with gr.Row():
                opportunities_dataframe = gr.Dataframe(
                    value=initial_table,
                    headers=[
                        "Deals found so far", "Price", "Estimate", "Discount",
                        "Savings %", "Confidence", "Review risk", "Approval status",
                    ],
                    wrap=True,
                    column_widths=["42%", "8%", "8%", "8%", "8%", "9%", "9%", "10%"],
                    row_count=10,
                    column_count=8,
                    max_height=400,
                    interactive=False,
                    elem_classes=["section-card"],
                )
            with gr.Group(elem_classes=["approval-card"]):
                gr.Markdown(
                    "## 👤 Human approval required\n\n"
                    "**Select a deal, verify the live listing, compare independent evidence, "
                    "then record a reasoned decision. No alert is sent before approval.**",
                    elem_classes=["approval-instructions"],
                )
                with gr.Accordion("Human evaluation — step-by-step", open=True):
                    gr.Markdown(
                        "1. **Select a pending row.** Start with High risk items so weak outputs "
                        "are caught quickly.\n"
                        "2. **Open the original listing.** Confirm model/SKU, new-used-refurbished "
                        "condition, quantity, shipping, coupon or membership requirements, and "
                        "the checkout price. Reject if any extracted fact is wrong.\n"
                        "3. **Inspect model agreement.** Large differences between the three "
                        "estimates mean uncertainty; confidence is not proof.\n"
                        "4. **Check comparable evidence.** Confirm that at least three results are "
                        "the same product class and condition. Ignore superficially similar items.\n"
                        "5. **Research a fair price independently.** Prefer recent sold prices or "
                        "multiple reputable retailers. Do not use the model estimate as your label.\n"
                        "6. **Enter the corrected fair price.** This is strongly recommended because "
                        "it enables MAE, RMSE, and bias measurement.\n"
                        "7. **Write a specific reason.** Example: `Reject — refurbished unit was "
                        "compared with new retail listings.`\n"
                        "8. **Approve or reject.** Approve only when the listing is accurate and the "
                        "estimated savings remain meaningful after all costs.\n"
                        "9. **Review aggregate results.** After at least 30 corrected labels, compare "
                        "MAE, RMSE, bias, and approval rate; also segment mistakes by category."
                    )
                selected_url = gr.Textbox(
                    label="Selected deal awaiting your decision",
                    placeholder="No deal selected — click a row in the table above",
                    interactive=False,
                )
                review_status = gr.Markdown(
                    "⏸️ **Waiting for your selection.** No alert will be sent until you approve a deal."
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        review_detail = gr.Markdown(
                            "Select a row to inspect confidence, model estimates, and evidence."
                        )
                    with gr.Column(scale=1):
                        price_history_plot = gr.Plot(value=go.Figure(), show_label=False)
                comparable_evidence = gr.Dataframe(
                    headers=["Comparable price", "Category", "Description", "Retrieval score"],
                    value=[],
                    datatype=["str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                    label="Retrieved comparables — verify relevance manually",
                    column_widths=["12%", "15%", "61%", "12%"],
                )
                with gr.Row():
                    corrected_price = gr.Number(
                        label="Independent fair-price label (strongly recommended)", minimum=0.01
                    )
                    review_reason = gr.Textbox(
                        label="Decision reason (required, at least 10 characters)", max_lines=3,
                        placeholder="State what you verified and why the estimate is credible or wrong.",
                    )
                with gr.Row():
                    approve_button = gr.Button(
                        "✅ Approve and publish alert", variant="primary"
                    )
                    reject_button = gr.Button(
                        "❌ Reject — do not alert", variant="stop"
                    )
            with gr.Group(elem_classes=["section-card"]):
                evaluation_results = gr.Markdown(
                    evaluation_markdown(framework.store.evaluation_summary())
                )
                refresh_evaluation = gr.Button("Refresh evaluation results", size="sm")
            with gr.Accordion("Saved deal searches", open=False):
                with gr.Row():
                    search_name = gr.Textbox(label="Watch name")
                    search_query = gr.Textbox(label="Keywords / product")
                    search_category = gr.Dropdown(
                        choices=[(category.replace("_", " "), category) for category in CATEGORIES],
                        label="Category (optional)",
                    )
                    save_search_button = gr.Button("Save watch")
                search_status = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    logs = gr.HTML(html_for([]), elem_classes=["section-card"])
                with gr.Column(scale=2, min_width=700):
                    plot = gr.Plot(
                        value=get_initial_plot(),
                        show_label=False,
                        elem_classes=["section-card"],
                    )

            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe, status],
            )
            # Build the expensive t-SNE visualization after Gradio starts so a
            # slow or resource-constrained plot cannot block the dashboard port.
            ui.load(get_plot, outputs=[plot])

            run_button.click(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe, status],
            )

            timer = gr.Timer(value=300, active=True)
            timer.tick(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe, status],
            )

            opportunities_dataframe.select(
                do_select,
                outputs=[selected_url, review_detail, comparable_evidence, price_history_plot],
            )
            approve_button.click(
                lambda url, corrected, reason: review(url, "approved", corrected, reason),
                inputs=[selected_url, corrected_price, review_reason],
                outputs=[opportunities_dataframe, review_status, evaluation_results],
            )
            reject_button.click(
                lambda url, corrected, reason: review(url, "rejected", corrected, reason),
                inputs=[selected_url, corrected_price, review_reason],
                outputs=[opportunities_dataframe, review_status, evaluation_results],
            )
            refresh_evaluation.click(
                lambda: evaluation_markdown(framework.store.evaluation_summary()),
                outputs=[evaluation_results],
            )
            save_search_button.click(
                save_search,
                inputs=[search_name, search_query, search_category],
                outputs=[search_status],
            )

        ui.queue(default_concurrency_limit=1)
        ui.launch(
            share=False,
            inbrowser=os.getenv("PRICER_OPEN_BROWSER", "true").lower()
            in {"1", "true", "yes", "on"},
            show_error=True,
            css=APP_CSS,
            theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        )


if __name__ == "__main__":
    App().run()
