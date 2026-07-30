from __future__ import annotations

import pytest

from adengine.config import EngineConfig
from adengine.domain import Ad, AdRequest, Campaign


def test_campaign_rejects_non_positive_bid():
    with pytest.raises(ValueError, match="bid"):
        Campaign("c1", "adv", 0.0, 10.0, "tech")


def test_ad_rejects_invalid_base_ctr():
    with pytest.raises(ValueError, match="base_ctr"):
        Ad("a1", "c1", "title", "tech", 0.0)


def test_request_rejects_zero_slots():
    with pytest.raises(ValueError, match="num_slots"):
        AdRequest("u1", "tech", 0, 0)


def test_engine_config_rejects_empty_categories():
    with pytest.raises(ValueError, match="categories"):
        EngineConfig(categories=())


def test_engine_config_defaults_are_valid():
    cfg = EngineConfig()
    assert cfg.num_slots >= 1
    assert cfg.reserve_price >= 0
