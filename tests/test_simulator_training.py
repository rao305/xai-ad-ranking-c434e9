from __future__ import annotations

from adengine.config import EngineConfig
from adengine.engine import AdEngine
from adengine.events import EventLog
from adengine.model import CTRModel
from adengine.simulator import position_bias, simulate
from helpers import make_index, make_request


def test_position_bias_decays():
    assert position_bias(0) == 1.0
    assert position_bias(1) == 0.7
    assert position_bias(99) == 0.2


def test_simulate_is_deterministic():
    cfg_requests = [make_request(user_id=f"u{i}", interest="tech") for i in range(20)]

    def run_once():
        log = EventLog()
        simulate(cfg_requests, make_index(), CTRModel(), log, seed=123, reserve_price=0.01)
        return [(e.ad_id, e.clicked, e.revenue) for e in log.events]

    assert run_once() == run_once()


def test_training_examples_preserve_request_context():
    from adengine.domain import AdRequest
    from adengine.features import vectorize

    engine = AdEngine(EngineConfig(cold_requests=5, train_requests=5, warm_requests=5, train_epochs=1))
    index = engine.seed_inventory()
    log = engine.simulate(engine.seed_requests(30), index, engine.new_model(), seed=11)
    examples = engine.examples_from_log(log, index)
    assert len(examples) == len(log.events)
    for ev, (features, clicked) in zip(log.events, examples):
        assert clicked == int(ev.clicked)
        # Rebuild independently and compare vectors for fidelity.
        ad = next(a for a in index.ads if a.id == ev.ad_id)
        req = AdRequest(ev.user_id, ev.user_interest, ev.request_slot, engine.config.num_slots)
        assert features == vectorize(req, ad, index.campaign_for(ad))
    assert any(ev.request_slot != 0 for ev in log.events)
