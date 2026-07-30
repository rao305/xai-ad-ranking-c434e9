"""Shared fixtures and tiny builders for adengine tests."""
from __future__ import annotations

from adengine.domain import Ad, AdRequest, Campaign
from adengine.retrieval import AdIndex


def make_campaign(
    cid: str = "camp-1",
    advertiser: str = "adv-1",
    bid: float = 1.0,
    daily_budget: float = 10.0,
    category: str = "tech",
) -> Campaign:
    return Campaign(cid, advertiser, bid, daily_budget, category)


def make_ad(
    ad_id: str = "ad-1",
    campaign_id: str = "camp-1",
    title: str = "tech offer",
    category: str = "tech",
    base_ctr: float = 0.2,
) -> Ad:
    return Ad(ad_id, campaign_id, title, category, base_ctr)


def make_request(
    user_id: str = "user-1",
    interest: str = "tech",
    slot: int = 0,
    num_slots: int = 2,
) -> AdRequest:
    return AdRequest(user_id, interest, slot, num_slots)


def make_index() -> AdIndex:
    campaigns = [
        make_campaign("camp-tech", "adv-tech", bid=1.2, category="tech"),
        make_campaign("camp-auto", "adv-auto", bid=0.8, category="auto"),
        make_campaign("camp-finance", "adv-finance", bid=1.5, category="finance", daily_budget=5.0),
    ]
    ads = [
        make_ad("ad-tech-1", "camp-tech", "tech one", "tech", 0.25),
        make_ad("ad-tech-2", "camp-tech", "tech two", "tech", 0.15),
        make_ad("ad-auto-1", "camp-auto", "auto one", "auto", 0.2),
        make_ad("ad-finance-1", "camp-finance", "finance one", "finance", 0.18),
    ]
    return AdIndex(ads, campaigns)
