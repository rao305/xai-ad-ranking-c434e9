"""Feature extraction: (request, ad, campaign) -> fixed-width float vector.

A CTR model can't consume raw strings. We turn each opportunity into a list of
named string features, then *hash* those into a fixed-width vector so the model
size never depends on how many categories or ad ids exist.
"""
from __future__ import annotations

import hashlib
from typing import List

from .domain import Ad, AdRequest, Campaign

# Width of the hashed feature vector. Fixed and known at model-build time.
NUM_FEATURES = 256


def raw_features(req: AdRequest, ad: Ad, campaign: Campaign) -> List[str]:
    """Named, human-readable features for one (request, ad) opportunity."""
    return [
        "bias",                                   # always-on intercept term
        f"ad_cat={ad.category}",
        f"user_interest={req.user_interest}",
        f"slot={req.slot}",
        f"advertiser={campaign.advertiser}",
        # Cross feature: does this user's interest match the ad's category?
        f"interest_match={int(req.user_interest == ad.category)}",
        # Cross of slot position with category — top slots behave differently.
        f"slot_x_cat={req.slot}:{ad.category}",
    ]


def _hash(token: str) -> int:
    """Stable hash of a feature string into [0, NUM_FEATURES)."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % NUM_FEATURES


def vectorize(req: AdRequest, ad: Ad, campaign: Campaign) -> List[float]:
    """Hash the named features into a fixed-width 0/1 vector."""
    vec = [0.0] * NUM_FEATURES
    for token in raw_features(req, ad, campaign):
        vec[_hash(token)] = 1.0
    return vec
