"""MCP server exposing PricerProject as reusable agent tools."""

from __future__ import annotations

from functools import lru_cache

from agents.deals import Deal
from deal_agent_framework import DealAgentFramework


@lru_cache(maxsize=1)
def framework() -> DealAgentFramework:
    return DealAgentFramework()


def create_server():
    try:
        from mcp.server.mcpserver import MCPServer
    except ImportError as error:
        raise RuntimeError("Install project dependencies to run the MCP server") from error

    server = MCPServer("pricer-intelligence")

    @server.tool()
    def list_opportunities(status: str = "") -> list[dict]:
        """List priced deals, optionally filtered by approval status."""
        return [
            item.model_dump()
            for item in framework().store.list_opportunities(status or None)
        ]

    @server.tool()
    def estimate_price(description: str, asking_price: float, url: str) -> dict:
        """Estimate fair value with confidence and comparable-product evidence."""
        service = framework()
        service.init_agents_as_needed()
        opportunity = service.planner.run(
            Deal(product_description=description, price=asking_price, url=url)
        )
        service.store.upsert_opportunity(opportunity)
        return opportunity.model_dump()

    @server.tool()
    def find_similar_products(
        description: str, category: str = "", limit: int = 5
    ) -> list[dict]:
        """Search dense and BM25 indexes and return cited comparable products."""
        service = framework()
        service.init_agents_as_needed()
        frontier = service.planner.ensemble.frontier
        result = frontier.retrieve(
            description,
            metadata_filter={"category": category} if category else None,
        )
        return frontier.evidence()[: max(1, min(limit, 20))]

    @server.tool()
    def submit_deal_feedback(
        url: str,
        decision: str,
        corrected_price: float | None = None,
        reason: str = "",
    ) -> dict:
        """Approve or reject an opportunity and record a correction/reason."""
        return framework().submit_feedback(
            url,
            decision,
            corrected_price=corrected_price,
            reason=reason,
        ).model_dump()

    @server.tool()
    def get_price_history(url: str) -> list[dict]:
        """Return observed and estimated prices for a product URL."""
        return framework().price_history(url)

    @server.tool()
    def save_deal_search(name: str, query: str, category: str = "") -> dict:
        """Persist a reusable user watch/search."""
        identifier = framework().store.save_search(
            name, query, {"category": category} if category else {}
        )
        return {"id": identifier, "status": "saved"}

    @server.tool()
    def run_saved_search(identifier: int) -> list[dict]:
        """Match a saved watch against the current opportunity history."""
        return [
            item.model_dump()
            for item in framework().store.saved_search_matches(identifier)
        ]

    return server


if __name__ == "__main__":
    create_server().run(transport="stdio")
