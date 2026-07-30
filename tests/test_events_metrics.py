from __future__ import annotations

from adengine.domain import AuctionResult
from adengine.events import EventLog
from adengine.metrics import by_campaign, compute, suggestions
from helpers import make_ad, make_campaign, make_request


def _result(price: float = 0.4, pctr: float = 0.2, campaign_id: str = "camp-1"):
    return AuctionResult(
        ad=make_ad(campaign_id=campaign_id),
        campaign=make_campaign(cid=campaign_id),
        predicted_ctr=pctr,
        ecpm=pctr * 1.0 * 1000,
        price=price,
        rank=0,
    )


def test_event_log_keeps_request_context():
    log = EventLog()
    req = make_request(interest="auto", slot=2)
    ev = log.record(req, _result(), clicked=True, charged=0.3)
    assert ev.user_interest == "auto"
    assert ev.request_slot == 2
    assert ev.clicked is True
    assert ev.revenue == 0.3


def test_metrics_empty_log():
    m = compute(EventLog())
    assert m.impressions == 0
    assert m.ctr == 0.0
    assert suggestions(m)[0].startswith("No impressions")


def test_metrics_and_campaign_breakdown():
    log = EventLog()
    log.record(make_request(), _result(price=0.5, pctr=0.8, campaign_id="camp-1"), True)
    log.record(make_request(), _result(price=0.5, pctr=0.1, campaign_id="camp-2"), False)
    m = compute(log)
    assert m.impressions == 2
    assert m.clicks == 1
    assert abs(m.ctr - 0.5) < 1e-9
    assert abs(m.revenue - 0.5) < 1e-9
    # |0.8-1| + |0.1-0| / 2 = 0.15
    assert abs(m.calibration_error - 0.15) < 1e-9
    rows = by_campaign(log)
    assert {r.campaign_id for r in rows} == {"camp-1", "camp-2"}
