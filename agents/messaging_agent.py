import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agents.deals import Deal, Opportunity
from agents.agent import Agent


class MessagingAgent(Agent):
    """Publish deal alerts locally instead of sending them to a phone.

    Alerts are printed in PyCharm/Jupyter and stored in a JSONL history plus a
    browser-friendly HTML dashboard. No API key, cloud model, or phone app is
    required.
    """

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self, output_dir: str | Path | None = None):
        self.log("Messaging Agent is initializing with local alerts")
        configured_dir = output_dir or os.getenv("PRICER_ALERT_DIR", "artifacts")
        self.output_dir = Path(configured_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "deal_alerts.jsonl"
        self.html_path = self.output_dir / "deal_alerts.html"
        self._refresh_html()
        self.log(f"Local alert page: {self.html_path.resolve()}")

        # Original phone implementation (disabled and kept for reference):
        # pushover_url = "https://api.pushover.net/1/messages.json"
        # requests.post(pushover_url, data={
        #     "user": os.getenv("PUSHOVER_USER"),
        #     "token": os.getenv("PUSHOVER_TOKEN"),
        #     "message": text,
        # })

    @staticmethod
    def craft_message(description: str, deal_price: float, estimate: float) -> str:
        """Create useful alert text locally; an LLM is unnecessary here."""
        discount = estimate - deal_price
        return (
            f"Deal: {description}\n"
            f"Price: ${deal_price:,.2f}\n"
            f"Estimated value: ${estimate:,.2f}\n"
            f"Estimated discount: ${discount:,.2f}"
        )

    def _load_alerts(self) -> list[dict]:
        if not self.jsonl_path.exists():
            return []
        alerts = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            try:
                alerts.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                # A damaged line should not prevent the other alerts loading.
                continue
        return alerts

    def _refresh_html(self) -> None:
        cards = []
        for alert in reversed(self._load_alerts()):
            url = html.escape(str(alert.get("url", "")), quote=True)
            description = html.escape(str(alert.get("description", "Deal")))
            message = html.escape(str(alert.get("message", "")))
            timestamp = html.escape(str(alert.get("timestamp", "")))
            values = []
            for label, key in (
                ("Price", "price"),
                ("Estimated value", "estimate"),
                ("Discount", "discount"),
            ):
                if alert.get(key) is not None:
                    values.append(f"{label}: ${float(alert[key]):,.2f}")
            link = f'<a href="{url}" target="_blank" rel="noopener">Open deal</a>' if url else ""
            cards.append(
                '<article class="alert">'
                f"<h2>{description}</h2>"
                f'<p class="values">{" &nbsp; | &nbsp; ".join(values)}</p>'
                f"<pre>{message}</pre>"
                f'<footer><span>{timestamp}</span>{link}</footer>'
                "</article>"
            )

        content = "\n".join(cards) or (
            '<p class="empty">No alerts yet. Run the planning agent; matching deals will appear here.</p>'
        )
        page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="30">
  <title>Pricer Local Deal Alerts</title>
  <style>
    body {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #172033;
            background: #f5f7fb; font-family: system-ui, sans-serif; }}
    h1 {{ margin-bottom: .25rem; }}
    .subtitle {{ color: #5d687a; margin-top: 0; }}
    .alert {{ margin: 1rem 0; padding: 1rem 1.25rem; background: white;
              border: 1px solid #dce2ec; border-radius: 12px; box-shadow: 0 2px 8px #17203312; }}
    .alert h2 {{ margin-top: 0; font-size: 1.1rem; }}
    .values {{ color: #075e54; font-weight: 650; }}
    pre {{ white-space: pre-wrap; font: inherit; }}
    footer {{ display: flex; gap: 1rem; justify-content: space-between; color: #687386; }}
    a {{ color: #3157c8; font-weight: 650; }}
    .empty {{ padding: 2rem; background: white; border-radius: 12px; }}
  </style>
</head>
<body>
  <h1>Pricer deal alerts</h1>
  <p class="subtitle">Local only. This page refreshes every 30 seconds.</p>
  {content}
</body>
</html>
"""
        self.html_path.write_text(page, encoding="utf-8")

    def push(self, text: str, opportunity: Opportunity | None = None) -> dict:
        """Write one alert locally and return the stored record."""
        record = {
            "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "message": text,
        }
        if opportunity is not None:
            record.update(
                {
                    "description": opportunity.deal.product_description,
                    "price": opportunity.deal.price,
                    "url": opportunity.deal.url,
                    "estimate": opportunity.estimate,
                    "discount": opportunity.discount,
                }
            )

        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._refresh_html()
        print(f"\n=== LOCAL DEAL ALERT ===\n{text}\nDashboard: {self.html_path.resolve()}\n")
        return record

    def alert(self, opportunity: Opportunity) -> dict:
        """Publish an Opportunity to the console and local dashboard."""
        text = self.craft_message(
            opportunity.deal.product_description,
            opportunity.deal.price,
            opportunity.estimate,
        )
        result = self.push(text, opportunity)
        self.log("Messaging Agent published a local alert")
        return result

    def notify(
        self,
        description: str,
        deal_price: float,
        estimated_true_value: float,
        url: str,
    ) -> dict:
        """Compatibility helper for callers without an Opportunity object."""
        opportunity = Opportunity(
            deal=Deal(product_description=description, price=deal_price, url=url),
            estimate=estimated_true_value,
            discount=estimated_true_value - deal_price,
        )
        return self.alert(opportunity)
