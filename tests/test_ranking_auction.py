from __future__ import annotations

from adengine.auction import clear
from adengine.domain import AdRequest
from adengine.model import CTRModel
from adengine.ranking import Scored, ecpm, rank
from adengine.retrieval import AdIndex
from helpers import make_ad, make_campaign, make_request


def test_ecpm_formula():
    assert ecpm(0.1, 2.0) == 200.0


def test_rank_orders_by_ecpm_with_stable_ties():
    # Force equal predictions with a cold model and equal bids.
    camp_a = make_campaign("camp-a", bid=1.0, category="tech")
    camp_b = make_campaign("camp-b", bid=1.0, category="tech")
    ads = [
        make_ad("ad-b", "camp-b", "b", "tech"),
        make_ad("ad-a", "camp-a", "a", "tech"),
    ]
    index = AdIndex(ads, [camp_a, camp_b])
    scored = rank(make_request(), index, CTRModel())
    # Equal eCPM → reverse lexicographic on ad id because sort is reverse=True.
    assert [s.ad.id for s in scored] == ["ad-b", "ad-a"]


def test_gsp_charges_runner_up_and_respects_reserve():
    winner = Scored(
        make_ad("ad-1"),
        make_campaign("camp-1", bid=2.0),
        predicted_ctr=0.2,
        ecpm=400.0,
    )
    runner = Scored(
        make_ad("ad-2", "camp-2"),
        make_campaign("camp-2", bid=1.0),
        predicted_ctr=0.1,
        ecpm=100.0,
    )
    req = AdRequest("u1", "tech", 0, 1)
    results = clear(req, [winner, runner], reserve_price=0.01)
    assert len(results) == 1
    # pay = next_ecpm / (pCTR * 1000) = 100 / 200 = 0.5
    assert results[0].price == 0.5


def test_last_winner_pays_at_least_reserve():
    winner = Scored(
        make_ad("ad-1"),
        make_campaign(bid=1.0),
        predicted_ctr=0.5,
        ecpm=500.0,
    )
    req = AdRequest("u1", "tech", 0, 1)
    results = clear(req, [winner], reserve_price=0.05)
    assert results[0].price == 0.05
