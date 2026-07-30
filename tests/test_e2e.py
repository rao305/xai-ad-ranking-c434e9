from __future__ import annotations

from adengine.config import EngineConfig
from adengine.engine import AdEngine
from adengine.run import main


def test_end_to_end_cold_train_warm_is_consistent_and_deterministic():
    cfg = EngineConfig(
        cold_requests=120,
        train_requests=200,
        warm_requests=120,
        train_epochs=3,
        reserve_price=0.01,
        daily_budget=50.0,
    )
    engine = AdEngine(cfg)

    cold1, warm1, loss1 = engine.run_demo()
    cold2, warm2, loss2 = engine.run_demo()

    assert loss1 >= 0.0
    assert cold1.impressions == cold2.impressions
    assert warm1.impressions == warm2.impressions
    assert abs(cold1.revenue - cold2.revenue) < 1e-9
    assert abs(warm1.revenue - warm2.revenue) < 1e-9
    assert cold1.impressions > 0
    assert warm1.impressions > 0
    assert 0.0 <= cold1.ctr <= 1.0
    assert 0.0 <= warm1.ctr <= 1.0


def test_cli_tiny_simulation(capsys):
    code = main(
        [
            "--cold-requests",
            "40",
            "--train-requests",
            "60",
            "--warm-requests",
            "40",
            "--train-epochs",
            "2",
            "--seed",
            "7",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "COLD model" in out
    assert "TRAINED model" in out
    assert "trained on" in out
