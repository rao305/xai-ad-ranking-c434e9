"""Validated configuration knobs for simulation, training, and auction clearing.

Keeping settings in one place makes the CLI, tests, and AdEngine share the same
defaults without scattering magic numbers across modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


DEFAULT_CATEGORIES: Tuple[str, ...] = ("tech", "auto", "finance", "travel")


@dataclass(frozen=True)
class EngineConfig:
    """Tunable parameters for a local cold→train→warm simulation run."""

    categories: Tuple[str, ...] = DEFAULT_CATEGORIES
    ads_per_category: int = 3
    daily_budget: float = 50.0
    num_slots: int = 3
    max_request_slot: int = 2
    cold_requests: int = 1000
    train_requests: int = 2000
    warm_requests: int = 1000
    train_epochs: int = 5
    learning_rate: float = 0.1
    reserve_price: float = 0.01
    inventory_seed: int = 1
    request_seed: int = 2
    cold_seed: int = 7
    train_seed: int = 99
    warm_seed: int = 7

    def __post_init__(self) -> None:
        if not self.categories:
            raise ValueError("categories must not be empty")
        if any(not c.strip() for c in self.categories):
            raise ValueError("categories must be non-empty strings")
        for name in (
            "ads_per_category",
            "num_slots",
            "cold_requests",
            "train_requests",
            "warm_requests",
            "train_epochs",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_request_slot < 0:
            raise ValueError("max_request_slot must be >= 0")
        if self.daily_budget <= 0:
            raise ValueError("daily_budget must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.reserve_price < 0:
            raise ValueError("reserve_price must be >= 0")


@dataclass
class RunReport:
    """Printed and returned summary for one simulation phase."""

    label: str
    impressions: int
    clicks: int
    ctr: float
    revenue: float
    rpm: float
    calibration_error: float
    suggestions: List[str] = field(default_factory=list)
