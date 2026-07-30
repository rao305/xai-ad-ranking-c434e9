"""The generalized second-price (GSP) auction.

Winners are the top-ranked candidates by eCPM. Each winner pays the *minimum*
price it would have needed to keep its slot — derived from the eCPM of the
candidate ranked just below it. This is the rule Google/Meta-style ad auctions
use; it discourages bid-gaming far better than a first-price auction.
"""
from __future__ import annotations

from typing import List

from .domain import AdRequest, AuctionResult
from .ranking import Scored


def clear(
    req: AdRequest,
    ranked: List[Scored],
    reserve_price: float = 0.01,
) -> List[AuctionResult]:
    """Run a GSP auction over an eCPM-sorted candidate list.

    reserve_price is the floor CPC when there is no meaningful runner-up eCPM.
    Without it the last winner would pay $0, which is unrealistic and wastes
    inventory.
    """
    if reserve_price < 0:
        raise ValueError(f"reserve_price must be >= 0, got {reserve_price}")

    num_winners = min(req.num_slots, len(ranked))
    results: List[AuctionResult] = []

    for rank_i in range(num_winners):
        winner = ranked[rank_i]
        # Competing eCPM comes from the next candidate, else reserve eCPM.
        if rank_i + 1 < len(ranked):
            next_ecpm = ranked[rank_i + 1].ecpm
        else:
            next_ecpm = reserve_price * winner.predicted_ctr * 1000.0

        # Price-per-click that exactly meets the runner-up's eCPM:
        #   pay * pCTR * 1000 = next_ecpm  ->  pay = next_ecpm / (pCTR * 1000)
        if winner.predicted_ctr > 0:
            price = next_ecpm / (winner.predicted_ctr * 1000.0)
        else:
            price = 0.0

        # Never pay more than the bid, and never below the reserve floor.
        price = min(price, winner.campaign.bid)
        price = max(price, min(reserve_price, winner.campaign.bid))

        results.append(
            AuctionResult(
                ad=winner.ad,
                campaign=winner.campaign,
                predicted_ctr=winner.predicted_ctr,
                ecpm=winner.ecpm,
                price=round(price, 4),
                rank=rank_i,
                extras={
                    "next_ecpm": round(next_ecpm, 4),
                    "reserve_price": reserve_price,
                },
            )
        )
    return results
