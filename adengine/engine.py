"""Orchestration layer: inventory seeding, serving, training, and reporting.

The CLI and tests should not re-implement the cold→train→warm flywheel. AdEngine
owns that workflow and keeps serve-time features identical to train-time features.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .config import EngineConfig, RunReport
from .domain import Ad, AdRequest, Campaign
from .events import EventLog
from .features import vectorize
from .metrics import by_campaign, compute, suggestions
from .model import CTRModel, Example
from .retrieval import AdIndex
from .simulator import simulate


class AdEngine:
    """Configured retrieve → rank → auction → log → train loop."""

    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()

    def seed_inventory(self, seed: Optional[int] = None) -> AdIndex:
        """Build a small fixed catalog of campaigns and ads."""
        cfg = self.config
        rng = random.Random(cfg.inventory_seed if seed is None else seed)
        campaigns: List[Campaign] = []
        ads: List[Ad] = []
        for cat in cfg.categories:
            for j in range(cfg.ads_per_category):
                cid = f"camp-{cat}-{j}"
                campaigns.append(
                    Campaign(
                        id=cid,
                        advertiser=f"adv-{cat}-{j}",
                        bid=round(0.5 + rng.random(), 2),
                        daily_budget=cfg.daily_budget,
                        category=cat,
                    )
                )
                ads.append(
                    Ad(
                        id=f"ad-{cat}-{j}",
                        campaign_id=cid,
                        title=f"{cat} offer {j}",
                        category=cat,
                        base_ctr=round(0.05 + rng.random() * 0.25, 3),
                    )
                )
        return AdIndex(ads, campaigns)

    def seed_requests(self, n: int, seed: Optional[int] = None) -> List[AdRequest]:
        """Generate n feed requests with varying interest and request slots."""
        cfg = self.config
        rng = random.Random(cfg.request_seed if seed is None else seed)
        return [
            AdRequest(
                user_id=f"user-{i}",
                user_interest=rng.choice(list(cfg.categories)),
                # Vary request slots so the slot feature actually moves.
                slot=rng.randint(0, cfg.max_request_slot),
                num_slots=cfg.num_slots,
            )
            for i in range(n)
        ]

    def new_model(self) -> CTRModel:
        return CTRModel(lr=self.config.learning_rate)

    def examples_from_log(self, log: EventLog, index: AdIndex) -> List[Example]:
        """Rebuild training examples from the exact logged request context.

        Using the original user_interest and request_slot avoids the classic
        leakage bug where training always sees interest_match=1.
        """
        ads_by_id = {ad.id: ad for ad in index.ads}
        data: List[Example] = []
        for ev in log.events:
            ad = ads_by_id[ev.ad_id]
            campaign = index.campaign_for(ad)
            req = AdRequest(
                user_id=ev.user_id,
                user_interest=ev.user_interest,
                slot=ev.request_slot,
                num_slots=self.config.num_slots,
            )
            data.append((vectorize(req, ad, campaign), int(ev.clicked)))
        return data

    def train_from_log(self, model: CTRModel, log: EventLog, index: AdIndex) -> float:
        data = self.examples_from_log(log, index)
        return model.train(data, epochs=self.config.train_epochs)

    def simulate(
        self,
        requests: List[AdRequest],
        index: AdIndex,
        model: CTRModel,
        seed: int,
        log: Optional[EventLog] = None,
    ) -> EventLog:
        event_log = log or EventLog()
        simulate(
            requests,
            index,
            model,
            event_log,
            seed=seed,
            reserve_price=self.config.reserve_price,
        )
        return event_log

    def report(self, label: str, log: EventLog) -> RunReport:
        m = compute(log)
        return RunReport(
            label=label,
            impressions=m.impressions,
            clicks=m.clicks,
            ctr=m.ctr,
            revenue=m.revenue,
            rpm=m.rpm,
            calibration_error=m.calibration_error,
            suggestions=suggestions(m),
        )

    def format_report(self, report: RunReport, log: Optional[EventLog] = None) -> str:
        lines = [
            f"\n=== {report.label} ===",
            (
                f"impressions={report.impressions} clicks={report.clicks} "
                f"ctr={report.ctr:.3f} revenue=${report.revenue:.2f} "
                f"rpm=${report.rpm:.2f} calib={report.calibration_error:.3f}"
            ),
        ]
        for tip in report.suggestions:
            lines.append(f"  - {tip}")
        if log is not None:
            campaign_rows = by_campaign(log)
            if campaign_rows:
                lines.append("  campaigns:")
                for row in campaign_rows[:8]:
                    lines.append(
                        f"    {row.campaign_id}: imps={row.impressions} "
                        f"clicks={row.clicks} ctr={row.ctr:.3f} "
                        f"rev=${row.revenue:.2f}"
                    )
                if len(campaign_rows) > 8:
                    lines.append(f"    ... {len(campaign_rows) - 8} more campaigns")
        return "\n".join(lines)

    def run_demo(self) -> Tuple[RunReport, RunReport, float]:
        """Cold serve → train on honest logs → warm serve."""
        cfg = self.config

        cold_model = self.new_model()
        cold_log = self.simulate(
            self.seed_requests(cfg.cold_requests, seed=cfg.request_seed),
            self.seed_inventory(seed=cfg.inventory_seed),
            cold_model,
            seed=cfg.cold_seed,
        )
        cold_report = self.report("COLD model", cold_log)

        train_index = self.seed_inventory(seed=cfg.inventory_seed)
        train_model = self.new_model()
        train_log = self.simulate(
            self.seed_requests(cfg.train_requests, seed=cfg.request_seed + 1),
            train_index,
            train_model,
            seed=cfg.train_seed,
        )
        loss = self.train_from_log(train_model, train_log, train_index)

        warm_log = self.simulate(
            self.seed_requests(cfg.warm_requests, seed=cfg.request_seed),
            self.seed_inventory(seed=cfg.inventory_seed),
            train_model,
            seed=cfg.warm_seed,
        )
        warm_report = self.report("TRAINED model", warm_log)
        return cold_report, warm_report, loss
