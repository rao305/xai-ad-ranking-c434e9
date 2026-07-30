"""Core domain types for the ad-serving engine.

Everything downstream — retrieval, ranking, the auction, logging — speaks in
terms of these small immutable records. Model the domain first; the algorithms
follow.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def _require_non_empty(name: str, value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must be a non-empty string")
    return text


def _require_positive(name: str, value: float) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _require_non_negative(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


@dataclass(frozen=True)
class Campaign:
    """An advertiser's budgeted line item. Bids are in dollars-per-click."""

    id: str
    advertiser: str
    bid: float            # max the advertiser will pay per click (CPC)
    daily_budget: float   # dollars/day; serving stops when spent
    category: str         # targeting key, e.g. "tech", "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty("id", self.id))
        object.__setattr__(self, "advertiser", _require_non_empty("advertiser", self.advertiser))
        object.__setattr__(self, "category", _require_non_empty("category", self.category))
        object.__setattr__(self, "bid", _require_positive("bid", self.bid))
        object.__setattr__(self, "daily_budget", _require_positive("daily_budget", self.daily_budget))


@dataclass(frozen=True)
class Ad:
    """A single creative belonging to a campaign."""

    id: str
    campaign_id: str
    title: str
    category: str
    base_ctr: float       # the ad's true latent click-rate (used by the simulator)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_non_empty("id", self.id))
        object.__setattr__(self, "campaign_id", _require_non_empty("campaign_id", self.campaign_id))
        object.__setattr__(self, "title", _require_non_empty("title", self.title))
        object.__setattr__(self, "category", _require_non_empty("category", self.category))
        if not 0.0 < self.base_ctr <= 1.0:
            raise ValueError(f"base_ctr must be in (0, 1], got {self.base_ctr}")


@dataclass(frozen=True)
class AdRequest:
    """One opportunity to show ads: a user viewing one feed slot."""

    user_id: str
    user_interest: str    # the category this user leans toward
    slot: int             # position in the feed (0 = top)
    num_slots: int        # how many ads this request can fill

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_id", _require_non_empty("user_id", self.user_id))
        object.__setattr__(
            self, "user_interest", _require_non_empty("user_interest", self.user_interest)
        )
        if self.slot < 0:
            raise ValueError(f"slot must be >= 0, got {self.slot}")
        if self.num_slots < 1:
            raise ValueError(f"num_slots must be >= 1, got {self.num_slots}")


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

    def __post_init__(self) -> None:
        if self.predicted_ctr < 0.0 or self.predicted_ctr > 1.0:
            raise ValueError(f"predicted_ctr must be in [0, 1], got {self.predicted_ctr}")
        _require_non_negative("ecpm", self.ecpm)
        _require_non_negative("price", self.price)
        if self.rank < 0:
            raise ValueError(f"rank must be >= 0, got {self.rank}")
