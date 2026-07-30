"""Candidate retrieval: AdRequest -> a small set of eligible ads to rank.

A real system has millions of ads and can't score them all per request, so a
cheap retrieval stage narrows inventory to a candidate set *before* the
expensive model runs. We model that filter here.
"""
from __future__ import annotations

from typing import Dict, List

from .domain import Ad, AdRequest, Campaign


class AdIndex:
    """In-memory inventory: all ads + their campaigns, with spend tracking."""

    def __init__(self, ads: List[Ad], campaigns: List[Campaign]) -> None:
        if not campaigns:
            raise ValueError("campaigns must not be empty")
        if not ads:
            raise ValueError("ads must not be empty")

        campaign_ids = [c.id for c in campaigns]
        if len(set(campaign_ids)) != len(campaign_ids):
            raise ValueError("campaign ids must be unique")

        ad_ids = [a.id for a in ads]
        if len(set(ad_ids)) != len(ad_ids):
            raise ValueError("ad ids must be unique")

        self.campaigns: Dict[str, Campaign] = {c.id: c for c in campaigns}
        for ad in ads:
            if ad.campaign_id not in self.campaigns:
                raise ValueError(f"ad {ad.id} references unknown campaign {ad.campaign_id}")

        self.ads = list(ads)
        self.spent: Dict[str, float] = {c.id: 0.0 for c in campaigns}

    def campaign_for(self, ad: Ad) -> Campaign:
        return self.campaigns[ad.campaign_id]

    def remaining_budget(self, campaign: Campaign) -> float:
        return max(0.0, campaign.daily_budget - self.spent[campaign.id])

    def has_budget(self, campaign: Campaign) -> bool:
        """A campaign is eligible only while it has budget left today."""
        return self.remaining_budget(campaign) > 0.0

    def charge(self, campaign: Campaign, amount: float) -> float:
        """Record spend, capped so a single click cannot blow past the daily budget."""
        if amount < 0:
            raise ValueError(f"charge amount must be >= 0, got {amount}")
        allowed = min(amount, self.remaining_budget(campaign))
        self.spent[campaign.id] += allowed
        return allowed

    def candidates(self, req: AdRequest) -> List[Ad]:
        """Eligible ads for this request: matching category OR the user's
        interest, and whose campaign still has budget."""
        out: List[Ad] = []
        for ad in self.ads:
            campaign = self.campaign_for(ad)
            if not self.has_budget(campaign):
                continue
            # Keep ads that speak the user's language — category match is the cheap filter.
            if ad.category == req.user_interest or campaign.category == req.user_interest:
                out.append(ad)
        return out
