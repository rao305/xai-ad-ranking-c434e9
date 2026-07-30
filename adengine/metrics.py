"""Platform metrics + rule-based optimization suggestions.

Reads the append-only EventLog and derives the KPIs an ad platform lives by,
then applies simple thresholds to surface concrete "do this next" advice — the
optimization suggestions the dashboard shows.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from .events import EventLog


@dataclass
class Metrics:
    impressions: int
    clicks: int
    ctr: float            # clicks / impressions
    revenue: float
    rpm: float            # revenue per 1000 impressions
    calibration_error: float  # mean |predicted_ctr - click outcome|


@dataclass
class CampaignMetrics:
    campaign_id: str
    impressions: int
    clicks: int
    ctr: float
    revenue: float


def compute(log: EventLog) -> Metrics:
    """Fold the event log into platform KPIs."""
    imps = log.impressions()
    clicks = log.clicks()
    ctr = clicks / imps if imps else 0.0
    revenue = log.revenue()
    rpm = (revenue / imps * 1000.0) if imps else 0.0

    # Compare each prediction to the actual 0/1 click — not the global CTR.
    cal = (
        sum(abs(e.predicted_ctr - (1.0 if e.clicked else 0.0)) for e in log.events) / imps
        if imps
        else 0.0
    )
    return Metrics(imps, clicks, ctr, revenue, rpm, cal)


def by_campaign(log: EventLog) -> List[CampaignMetrics]:
    """Break platform metrics down per campaign for pacing / quality checks."""
    buckets: Dict[str, List] = defaultdict(list)
    for ev in log.events:
        buckets[ev.campaign_id].append(ev)

    out: List[CampaignMetrics] = []
    for campaign_id, events in sorted(buckets.items()):
        imps = len(events)
        clicks = sum(1 for e in events if e.clicked)
        revenue = sum(e.revenue for e in events)
        out.append(
            CampaignMetrics(
                campaign_id=campaign_id,
                impressions=imps,
                clicks=clicks,
                ctr=(clicks / imps) if imps else 0.0,
                revenue=revenue,
            )
        )
    return out


def suggestions(m: Metrics) -> List[str]:
    """Turn KPIs into concrete optimization advice."""
    out: List[str] = []
    if m.impressions == 0:
        return ["No impressions served — widen retrieval targeting."]
    if m.ctr < 0.05:
        out.append("Low CTR: tighten interest-match targeting in retrieval.")
    if m.calibration_error > 0.35:
        out.append("Model poorly calibrated: retrain the CTR model on recent logs.")
    if m.rpm < 1.0:
        out.append("Low RPM: raise floor prices or admit higher-bid campaigns.")
    if not out:
        out.append("Healthy: CTR, calibration, and RPM are within target.")
    return out
