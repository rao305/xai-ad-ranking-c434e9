from __future__ import annotations

import pytest

from adengine.retrieval import AdIndex
from helpers import make_ad, make_campaign, make_index, make_request


def test_candidates_match_user_interest():
    index = make_index()
    req = make_request(interest="tech")
    ids = {ad.id for ad in index.candidates(req)}
    assert "ad-tech-1" in ids
    assert "ad-tech-2" in ids
    assert "ad-auto-1" not in ids


def test_unknown_campaign_rejected():
    with pytest.raises(ValueError, match="unknown campaign"):
        AdIndex([make_ad(campaign_id="missing")], [make_campaign()])


def test_charge_caps_at_remaining_budget():
    camp = make_campaign(daily_budget=1.0)
    ad = make_ad()
    index = AdIndex([ad], [camp])
    charged = index.charge(camp, 5.0)
    assert charged == 1.0
    assert index.spent[camp.id] == 1.0
    assert not index.has_budget(camp)
    assert index.candidates(make_request()) == []
