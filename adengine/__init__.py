"""AI-Powered Ad Ranking & Auction Engine.

A local educational decision core: retrieve → predict CTR → rank by eCPM →
clear a GSP auction → log impressions/clicks → retrain → measure.
"""

from .auction import clear
from .config import EngineConfig
from .domain import Ad, AdRequest, AuctionResult, Campaign
from .engine import AdEngine
from .events import EventLog, ImpressionEvent
from .model import CTRModel
from .retrieval import AdIndex

__all__ = [
    "Ad",
    "AdEngine",
    "AdIndex",
    "AdRequest",
    "AuctionResult",
    "CTRModel",
    "Campaign",
    "EngineConfig",
    "EventLog",
    "ImpressionEvent",
    "clear",
]

__version__ = "1.0.0"
