"""Core domain types for the ad-serving engine.

Everything downstream — retrieval, ranking, the auction, logging — speaks in
terms of these small immutable records. Model the domain first; the algorithms
follow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Campaign:
    """An advertiser's budgeted line item. Bids are in dollars-per-click."""

    id: str
    advertiser: str
    bid: float            # max the advertiser will pay per click (CPC)
    daily_budget: float   # dollars/day; serving stops when spent
    category: str         # targeting key, e.g. "tech", "auto"


@dataclass(frozen=True)
class Ad:
    """A single creative belonging to a campaign."""

    id: str
    campaign_id: str
    title: str
    category: str
    base_ctr: float       # the ad's true latent click-rate (used by the simulator)


@dataclass(frozen=True)
class AdRequest:
    """One opportunity to show ads: a user viewing one feed slot."""

    user_id: str
    user_interest: str    # the category this user leans toward
    slot: int             # position in the feed (0 = top)
    num_slots: int        # how many ads this request can fill


@dataclass
class AuctionResult:
    """The outcome of ranking + clearing one AdRequest."""

    ad: Ad
    campaign: Campaign
    predicted_ctr: float
    ecpm: float           # expected revenue per mille (1000 impressions)
    price: float          # what the winner actually pays per click (GSP)
    rank: int             # 0 = top winning slot
    extras: dict = field(default_factory=dict)
