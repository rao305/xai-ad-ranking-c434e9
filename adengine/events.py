"""Append-only event logging: impressions and clicks.

Every served ad writes an impression; a click writes revenue. In production
these are Kafka events streamed to a warehouse — here an in-memory list with the
same append-only discipline. The log is the source of truth for all metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .domain import AuctionResult


@dataclass
class ImpressionEvent:
    user_id: str
    ad_id: str
    campaign_id: str
    rank: int
    predicted_ctr: float
    price: float          # what a click on this impression will cost
    clicked: bool = False
    revenue: float = 0.0  # price if clicked, else 0


@dataclass
class EventLog:
    """In-memory append-only log of impression/click events."""

    events: List[ImpressionEvent] = field(default_factory=list)

    def record(self, req_user_id: str, result: AuctionResult, clicked: bool) -> ImpressionEvent:
        """Log one served impression and whether it was clicked."""
        ev = ImpressionEvent(
            user_id=req_user_id,
            ad_id=result.ad.id,
            campaign_id=result.campaign.id,
            rank=result.rank,
            predicted_ctr=result.predicted_ctr,
            price=result.price,
            clicked=clicked,
            revenue=result.price if clicked else 0.0,
        )
        self.events.append(ev)
        return ev

    def impressions(self) -> int:
        return len(self.events)

    def clicks(self) -> int:
        return sum(1 for e in self.events if e.clicked)

    def revenue(self) -> float:
        return sum(e.revenue for e in self.events)
