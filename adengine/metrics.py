"""Platform metrics + rule-based optimization suggestions.

Reads the append-only EventLog and derives the KPIs an ad platform lives by,
then applies simple thresholds to surface concrete "do this next" advice — the
optimization suggestions the dashboard shows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .events import EventLog


@dataclass
class Metrics:
    impressions: int
    clicks: int
    ctr: float            # clicks / impressions
    revenue: float
    rpm: float            # revenue per 1000 impressions
    calibration_error: float  # mean |predicted_ctr - actual_ctr|


def compute(log: EventLog) -> Metrics:
    """Fold the event log into platform KPIs."""
    imps = log.impressions()
    clicks = log.clicks()
    ctr = clicks / imps if imps else 0.0
    revenue = log.revenue()
    rpm = (revenue / imps * 1000.0) if imps else 0.0

    # Calibration: average gap between what we predicted and what happened.
    actual = ctr
    cal = (
        sum(abs(e.predicted_ctr - actual) for e in log.events) / imps
        if imps
        else 0.0
    )
    return Metrics(imps, clicks, ctr, revenue, rpm, cal)


def suggestions(m: Metrics) -> List[str]:
    """Turn KPIs into concrete optimization advice."""
    out: List[str] = []
    if m.impressions == 0:
        return ["No impressions served — widen retrieval targeting."]
    if m.ctr < 0.05:
        out.append("Low CTR: tighten interest-match targeting in retrieval.")
    if m.calibration_error > 0.1:
        out.append("Model poorly calibrated: retrain the CTR model on recent logs.")
    if m.rpm < 1.0:
        out.append("Low RPM: raise floor prices or admit higher-bid campaigns.")
    if not out:
        out.append("Healthy: CTR, calibration, and RPM are within target.")
    return out
