"""CLI entry point: seed inventory, simulate a feed, train, re-simulate, report.

    python -m adengine
    python -m adengine --cold-requests 500 --train-epochs 3

Shows the flywheel end to end — a cold model, a training pass on its own logs,
and how metrics change when we serve again with the trained model.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import EngineConfig
from .engine import AdEngine


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adengine",
        description="Local AI-powered ad ranking and GSP auction simulator.",
    )
    p.add_argument("--cold-requests", type=int, default=1000, help="Requests for cold serve")
    p.add_argument("--train-requests", type=int, default=2000, help="Requests used to train")
    p.add_argument("--warm-requests", type=int, default=1000, help="Requests for warm serve")
    p.add_argument("--train-epochs", type=int, default=5, help="SGD epochs on logged examples")
    p.add_argument("--learning-rate", type=float, default=0.1, help="SGD learning rate")
    p.add_argument("--reserve-price", type=float, default=0.01, help="Floor CPC in GSP auction")
    p.add_argument("--daily-budget", type=float, default=50.0, help="Campaign daily budget")
    p.add_argument("--num-slots", type=int, default=3, help="Ads filled per request")
    p.add_argument("--seed", type=int, default=7, help="Base seed for cold/warm serve")
    p.add_argument(
        "--show-campaigns",
        action="store_true",
        help="Include per-campaign metric breakdowns in the report",
    )
    return p


def config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        cold_requests=args.cold_requests,
        train_requests=args.train_requests,
        warm_requests=args.warm_requests,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        reserve_price=args.reserve_price,
        daily_budget=args.daily_budget,
        num_slots=args.num_slots,
        cold_seed=args.seed,
        warm_seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = AdEngine(config_from_args(args))

    # Cold model: weights all zero, every ad scores ~0.5.
    cold_model = engine.new_model()
    cold_index = engine.seed_inventory()
    cold_log = engine.simulate(
        engine.seed_requests(engine.config.cold_requests),
        cold_index,
        cold_model,
        seed=engine.config.cold_seed,
    )
    cold_report = engine.report("COLD model", cold_log)
    print(
        engine.format_report(cold_report, cold_log if args.show_campaigns else None)
    )

    # Train on logged behavior using the exact request context, then serve again.
    train_index = engine.seed_inventory()
    trained = engine.new_model()
    train_log = engine.simulate(
        engine.seed_requests(engine.config.train_requests, seed=engine.config.request_seed + 1),
        train_index,
        trained,
        seed=engine.config.train_seed,
    )
    loss = engine.train_from_log(trained, train_log, train_index)
    print(f"\ntrained on {len(train_log.events)} examples, final log loss {loss:.4f}")

    warm_index = engine.seed_inventory()
    warm_log = engine.simulate(
        engine.seed_requests(engine.config.warm_requests),
        warm_index,
        trained,
        seed=engine.config.warm_seed,
    )
    warm_report = engine.report("TRAINED model", warm_log)
    print(
        engine.format_report(warm_report, warm_log if args.show_campaigns else None)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
