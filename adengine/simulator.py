"""Feed simulator: generate AdRequests, run the serve loop, resolve clicks.

This is the engine's test harness and data generator. For each simulated feed
view it retrieves → ranks → clears the auction → logs the impressions, then
flips a biased coin against the ad's TRUE ctr to decide whether each was
clicked. That click feeds the log (and the next training round).
"""
from __future__ import annotations

import random
from typing import List

from .auction import clear
from .domain import Ad, AdRequest
from .events import EventLog
from .model import CTRModel
from .ranking import rank
from .retrieval import AdIndex

# Top slots get more attention; this multiplier decays the true CTR by position.
POSITION_BIAS = [1.0, 0.7, 0.5, 0.35, 0.25]


def position_bias(rank_i: int) -> float:
    return POSITION_BIAS[rank_i] if rank_i < len(POSITION_BIAS) else 0.2


def realize_click(ad: Ad, rank_i: int, rng: random.Random) -> bool:
    """Flip a biased coin: the ad's true CTR, decayed by slot position."""
    p_true = ad.base_ctr * position_bias(rank_i)
    return rng.random() < p_true


def run_session(
    req: AdRequest,
    index: AdIndex,
    model: CTRModel,
    log: EventLog,
    rng: random.Random,
) -> None:
    """Serve and resolve one feed view."""
    ranked = rank(req, index, model)
    results = clear(req, ranked)
    for result in results:
        clicked = realize_click(result.ad, result.rank, rng)
        log.record(req.user_id, result, clicked)
        if clicked:
            index.charge(result.campaign, result.price)


def simulate(
    requests: List[AdRequest],
    index: AdIndex,
    model: CTRModel,
    log: EventLog,
    seed: int = 7,
) -> None:
    """Run a batch of feed views through the full serve loop."""
    rng = random.Random(seed)
    for req in requests:
        run_session(req, index, model, log, rng)
