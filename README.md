# AI-Powered Ad Ranking & Auction Engine

So I built this project to understand what happens after an ad request reaches a real delivery system. Not the full production stack with Kafka and Redis — the decision core. The result is a working local Python engine: model ads and campaigns, extract user×ad features, predict click-through rate with a hand-rolled logistic regression model, rank candidates by expected revenue (eCPM), clear winners through a generalized second-price (GSP) auction, log impressions/clicks/revenue, retrain on the exact served context, then compute platform metrics and emit optimization suggestions.

This is an educational simulator of the ranking → auction → measurement loop that powers large advertising systems. It is intentionally dependency-light and deterministic so you can read every line and rerun the same experiment.

## What I learned

- Retrieval and ranking are different jobs. A cheap category/budget filter has to shrink inventory before the model scores anything.
- Ranking by bid alone is wrong for the platform. Ranking by eCPM (`pCTR × bid × 1000`) aligns selection with expected revenue.
- GSP pricing is a product rule, not a math trick: winners pay the minimum needed to keep their slot, floored by a reserve so the last winner does not get free inventory.
- Training data must preserve the original request. If you rebuild labels with `user_interest = winning_ad.category`, `interest_match` becomes always-on leakage and the model lies to itself.
- Budgets need hard caps at charge time. Checking `spent < budget` before a click is not enough — one expensive click can still overshoot.
- Better calibration does not guarantee more revenue. A trained model can raise CTR while GSP prices and budget pacing move total revenue down. Measure both.

## Architecture

```text
AdRequest
  → AdIndex.candidates (category match + remaining budget)
  → feature hash (256-d)
  → CTRModel.predict (logistic regression)
  → rank by eCPM
  → GSP clear (reserve floor)
  → EventLog (impression / click / charged revenue)
        ↘ train on exact logged request context
        ↘ metrics + suggestions
```

| Module | Role |
|---|---|
| `adengine/domain.py` | Campaign, Ad, AdRequest, AuctionResult |
| `adengine/retrieval.py` | In-memory inventory, targeting, budget-safe charging |
| `adengine/features.py` | Named features → hashed vector |
| `adengine/model.py` | From-scratch logistic CTR model (SGD) |
| `adengine/ranking.py` | eCPM scoring + deterministic tie-break |
| `adengine/auction.py` | Generalized second-price clearing |
| `adengine/events.py` | Append-only impression/click log with request context |
| `adengine/simulator.py` | Feed simulator + position-biased click realization |
| `adengine/metrics.py` | Platform/campaign KPIs + suggestions |
| `adengine/config.py` | Validated run knobs |
| `adengine/engine.py` | Cold → train → warm orchestration |
| `adengine/run.py` | CLI |

## Ranking and auction math

```text
eCPM = predicted_ctr × bid × 1000

GSP price for winner i:
  next_ecpm = eCPM(i+1)   or   reserve × pCTR(i) × 1000 if no runner-up
  price     = next_ecpm / (pCTR(i) × 1000)
  price     = min(price, bid)
  price     = max(price, min(reserve, bid))
```

Clicks are realized from each ad's latent `base_ctr`, decayed by auction position bias. Campaign spend is incremented only on clicks and never exceeds `daily_budget`.

## Measured result

Deterministic local run with defaults and `--seed 7`:

```text
=== COLD model ===
impressions=2976 clicks=370 ctr=0.124 revenue=$237.38 rpm=$79.76 calib=0.500
  - Model poorly calibrated: retrain the CTR model on recent logs.

trained on 5377 examples, final log loss 0.3526

=== TRAINED model ===
impressions=3000 clicks=406 ctr=0.135 revenue=$197.12 rpm=$65.71 calib=0.294
  - Healthy: CTR, calibration, and RPM are within target.
```

Training improved CTR and calibration. Revenue fell in this seed because the warm ranker and GSP prices allocate inventory differently than a cold model that scores every ad near 0.5. That tradeoff is the point of the exercise.


Coverage includes domain/config validation, feature hashing, SGD learning, targeting and budget caps, deterministic ranking/ties, GSP + reserve pricing, event/metric accuracy, train/serve feature fidelity, and an end-to-end cold→train→warm workflow.

## Notes 

- This is a mock decision engine, not a distributed ad platform. There is no Redis, Kafka, PostgreSQL, Docker, or HTTP serving path here. (Probably be adding it in the future IDK)
- Inventory, bids, and click probabilities are synthetic.
- The CTR model is plain logistic regression over hashed features — no deep model, no online learning service.
- Suggestions are rule-based thresholds, not an auto-optimizer that mutates bids or targeting.

Stack: Python, logistic regression, eCPM ranking, GSP auctions, event logging, simulation.
