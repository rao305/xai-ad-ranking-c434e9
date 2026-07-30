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
        self.ads = ads
        self.campaigns: Dict[str, Campaign] = {c.id: c for c in campaigns}
        self.spent: Dict[str, float] = {c.id: 0.0 for c in campaigns}

    def campaign_for(self, ad: Ad) -> Campaign:
        return self.campaigns[ad.campaign_id]

    def has_budget(self, campaign: Campaign) -> bool:
        """A campaign is eligible only while it has budget left today."""
        return self.spent[campaign.id] < campaign.daily_budget

    def charge(self, campaign: Campaign, amount: float) -> None:
        """Record spend against a campaign's daily budget."""
        self.spent[campaign.id] += amount

    def candidates(self, req: AdRequest) -> List[Ad]:
        """Eligible ads for this request: matching category OR the user's
        interest, and whose campaign still has budget."""
        out: List[Ad] = []
        for ad in self.ads:
            campaign = self.campaign_for(ad)
            if not self.has_budget(campaign):
                continue
            if ad.category == req.user_interest or campaign.category == req.user_interest:
                out.append(ad)
        return out
