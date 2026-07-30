"""Ranking: score each candidate and order them by expected revenue.

The platform's revenue per impression is pCTR * bid (a click only pays when it
happens). Ranking by that quantity — eCPM — aligns the auction with what the
platform actually earns, not just who bids highest.
"""
from __future__ import annotations

from typing import List, NamedTuple

from .domain import Ad, AdRequest, Campaign
from .features import vectorize
from .model import CTRModel
from .retrieval import AdIndex


class Scored(NamedTuple):
    """A candidate after scoring, ready for the auction."""

    ad: Ad
    campaign: Campaign
    predicted_ctr: float
    ecpm: float


def ecpm(predicted_ctr: float, bid: float) -> float:
    """Expected revenue per 1000 impressions: pCTR * bid * 1000."""
    return predicted_ctr * bid * 1000.0


def rank(req: AdRequest, index: AdIndex, model: CTRModel) -> List[Scored]:
    """Score every candidate and return them sorted by eCPM, highest first."""
    scored: List[Scored] = []
    for ad in index.candidates(req):
        campaign = index.campaign_for(ad)
        p = model.predict(vectorize(req, ad, campaign))
        scored.append(Scored(ad, campaign, p, ecpm(p, campaign.bid)))
    scored.sort(key=lambda s: s.ecpm, reverse=True)
    return scored
