"""CLI entry point: seed inventory, simulate a feed, train, re-simulate, report.

    python -m adengine.run

Shows the flywheel end to end — a cold model, a training pass on its own logs,
and the lift in revenue when we serve again with the trained model.
"""
from __future__ import annotations

import random
from typing import List

from .domain import Ad, AdRequest, Campaign
from .events import EventLog
from .features import vectorize
from .metrics import compute, suggestions
from .model import CTRModel
from .retrieval import AdIndex
from .simulator import realize_click, simulate

CATEGORIES = ["tech", "auto", "finance", "travel"]


def seed_inventory() -> AdIndex:
    """A small, fixed catalog of campaigns and ads."""
    campaigns: List[Campaign] = []
    ads: List[Ad] = []
    rng = random.Random(1)
    for i, cat in enumerate(CATEGORIES):
        for j in range(3):
            cid = f"camp-{cat}-{j}"
            campaigns.append(
                Campaign(cid, f"adv-{cat}-{j}", bid=round(0.5 + rng.random(), 2),
                         daily_budget=50.0, category=cat)
            )
            ads.append(
                Ad(f"ad-{cat}-{j}", cid, f"{cat} offer {j}", cat,
                   base_ctr=round(0.05 + rng.random() * 0.25, 3))
            )
    return AdIndex(ads, campaigns)


def seed_requests(n: int) -> List[AdRequest]:
    rng = random.Random(2)
    return [
        AdRequest(f"user-{i}", rng.choice(CATEGORIES), slot=0, num_slots=3)
        for i in range(n)
    ]


def training_pass(index: AdIndex, model: CTRModel) -> None:
    """Generate labeled examples by serving once, then SGD-train on them."""
    rng = random.Random(99)
    log = EventLog()
    simulate(seed_requests(2000), index, model, log, seed=99)
    data = []
    for ev in log.events:
        ad = next(a for a in index.ads if a.id == ev.ad_id)
        req = AdRequest(ev.user_id, ad.category, ev.rank, 3)
        data.append((vectorize(req, ad, index.campaign_for(ad)), int(ev.clicked)))
    loss = model.train(data, epochs=5)
    print(f"trained on {len(data)} examples, final log loss {loss:.4f}")


def report(label: str, log: EventLog) -> None:
    m = compute(log)
    print(f"\n=== {label} ===")
    print(f"impressions={m.impressions} clicks={m.clicks} "
          f"ctr={m.ctr:.3f} revenue=${m.revenue:.2f} rpm=${m.rpm:.2f} "
          f"calib={m.calibration_error:.3f}")
    for s in suggestions(m):
        print(f"  - {s}")


def main() -> None:
    # Cold model: weights all zero, every ad scores 0.5.
    cold = CTRModel()
    cold_log = EventLog()
    simulate(seed_requests(1000), seed_inventory(), cold, cold_log, seed=7)
    report("COLD model", cold_log)

    # Train on logged behavior, then serve again with fresh inventory.
    trained = CTRModel()
    training_pass(seed_inventory(), trained)
    warm_log = EventLog()
    simulate(seed_requests(1000), seed_inventory(), trained, warm_log, seed=7)
    report("TRAINED model", warm_log)


if __name__ == "__main__":
    main()
