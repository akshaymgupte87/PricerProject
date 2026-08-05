import html
import logging
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
    """Convert domain models into the stable five-column UI contract."""
    return [
        [
            opportunity.deal.product_description,
            f"${opportunity.deal.price:,.2f}",
            f"${opportunity.estimate:,.2f}",
            f"${opportunity.discount:,.2f}",
            opportunity.deal.url,
        ]
        for opportunity in opportunities
    ]


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

            def do_select(selected_index: gr.SelectData):
                opportunities = framework.display_opportunities()
                row = selected_index.index[0] if isinstance(selected_index.index, tuple) else selected_index.index
                if row >= len(opportunities):
                    return
                opportunity = opportunities[row]
                framework.init_agents_as_needed()
                framework.planner.messenger.alert(opportunity)

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
                    headers=["Deals found so far", "Price", "Estimate", "Discount", "URL"],
                    wrap=True,
                    column_widths=["50%", "10%", "10%", "10%", "20%"],
                    row_count=10,
                    column_count=5,
                    max_height=400,
                    interactive=False,
                    elem_classes=["section-card"],
                )
            with gr.Row():
                with gr.Column(scale=1, min_width=360):
                    logs = gr.HTML(html_for([]), elem_classes=["section-card"])
                with gr.Column(scale=2, min_width=700):
                    plot = gr.Plot(value=get_plot(), show_label=False, elem_classes=["section-card"])

            ui.load(
                run_with_logging,
                inputs=[log_data],
                outputs=[log_data, logs, opportunities_dataframe, status],
            )

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

            opportunities_dataframe.select(do_select)

        ui.queue(default_concurrency_limit=1)
        ui.launch(
            share=False,
            inbrowser=True,
            show_error=True,
            css=APP_CSS,
            theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"),
        )


if __name__ == "__main__":
    App().run()
