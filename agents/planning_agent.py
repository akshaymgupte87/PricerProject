import logging
from collections.abc import Callable
from typing import Optional, List
from agents.agent import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from agents.deal_seen_store import DealSeenStore


class PlanningAgent(Agent):

    name = "Planning Agent"
    color = Agent.GREEN
    DEAL_THRESHOLD = 50
    MIN_DISCOUNT_PERCENT = 0.20

    def __init__(self, collection, seen_store: DealSeenStore | None = None):
        """
        Create instances of the 3 Agents that this planner coordinates across
        """
        self.log("Planning Agent is initializing")
        self.scanner = ScannerAgent(seen_store=seen_store)
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Run the workflow for a particular deal
        :param deal: the deal, summarized from an RSS scrape
        :returns: an opportunity including the discount
        """
        self.log("Planning Agent is pricing up a potential deal")
        estimate = self.ensemble.price(deal.product_description)
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(
        self,
        memory: List[Opportunity] | None = None,
        on_opportunity: Callable[[Opportunity, List[Opportunity]], None] | None = None,
    ) -> Optional[Opportunity]:
        """
        Run the full workflow:
        1. Use the ScannerAgent to find deals from RSS feeds
        2. Use the EnsembleAgent to estimate them
        3. Use the MessagingAgent to send a notification of deals
        :param memory: a list of URLs that have been surfaced in the past
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Planning Agent is kicking off a run")
        memory = memory or []
        selection = self.scanner.scan(memory=memory)
        if selection:
            opportunities = []
            for index, deal in enumerate(selection.deals[:5], start=1):
                try:
                    opportunity = self.run(deal)
                except Exception:
                    logging.exception(
                        "Pricing failed for deal %d/%d: %s",
                        index,
                        min(5, len(selection.deals)),
                        deal.product_description[:80],
                    )
                    continue

                opportunities.append(opportunity)
                self.log(
                    f"Planning Agent evaluated deal {index}/{min(5, len(selection.deals))}"
                )
                if on_opportunity:
                    try:
                        on_opportunity(opportunity, opportunities.copy())
                    except Exception:
                        logging.exception("Opportunity progress callback failed")

            if not opportunities:
                self.log("Planning Agent could not price any selected deals")
                return None

            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            discount_percent = best.discount / best.estimate if best.estimate > 0 else 0
            qualifies = (
                best.discount > self.DEAL_THRESHOLD
                and discount_percent >= self.MIN_DISCOUNT_PERCENT
            )
            if qualifies:
                self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            return best if qualifies else None
        return None
