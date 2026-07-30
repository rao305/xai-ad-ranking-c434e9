from __future__ import annotations

from adengine.domain import Ad, AdRequest, Campaign
from adengine.features import NUM_FEATURES, raw_features, vectorize


def test_raw_features_include_interest_match_and_slot():
    req = AdRequest("u1", "tech", 2, 3)
    ad = Ad("a1", "c1", "offer", "auto", 0.2)
    camp = Campaign("c1", "adv", 1.0, 10.0, "auto")
    feats = raw_features(req, ad, camp)
    assert "interest_match=0" in feats
    assert "slot=2" in feats
    assert "bias" in feats


def test_vectorize_is_fixed_width_and_deterministic():
    req = AdRequest("u1", "tech", 0, 3)
    ad = Ad("a1", "c1", "offer", "tech", 0.2)
    camp = Campaign("c1", "adv", 1.0, 10.0, "tech")
    v1 = vectorize(req, ad, camp)
    v2 = vectorize(req, ad, camp)
    assert len(v1) == NUM_FEATURES
    assert v1 == v2
    assert sum(v1) >= 1.0
