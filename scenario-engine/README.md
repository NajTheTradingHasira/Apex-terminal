# Apex scenario integration

The engine, rules and 18-template catalogue are copied unchanged from the user's
**Build SPY scenario state engine (2)** project (`Vercel 9/src`). The calendar
comes from that project's `recovered/nexus-dev/src/liveSession.js`. Original
engine, scenario, provisional and partial-input tests accompany the copy.

`apex-adapter.js` is the Apex-specific integration. It aggregates five complete
one-minute candles into each five-minute bar, preserves first receipt times,
and supplies an explicitly labelled bar-VWAP estimate and within-session volume
ratio. Its volume ratio compares one five-minute block to the prior four blocks;
it is not a historical same-time relative-volume baseline. A separate daily
history read supplies the previous completed session's high, low, close and a
simple 14-day true-range average. Today's partial daily row is excluded.

All 18 templates participate in the original phase, priority, acceptance,
invalidation and successor logic. The library exposes individual missing and
failed predicates. The three neutral templates remain neutral observations or
no-trade states rather than manufactured bullish/bearish signals.

The adapter does **not** have value-area/volume-profile levels, breadth history,
positioning or verified event coverage. Those inputs remain missing. The
original provisional-analysis option permits price scenario context while
event coverage is unknown, but cannot grant trade permission. Confirming the
old tape-input checkbox does not synthesize these sources. Source receipt
times mean opening the panel mid-session cannot retroactively qualify levels
as though the client observed them before opening.

By default the library is live context alongside the existing 1m entry gate.
**Require scenario approval for new entries** explicitly opts into its stricter
permission requirements. With current missing event coverage that switch
blocks entries. It can never promote an existing NO-GO or CAUTION. It resets
on reload. The same scenario context is included in AI requests.

The source calendar covers published 2026–2028 holidays and early closes and
fails unavailable for other years. It does not cover unscheduled closures.
This calendar applies to the scenario layer; the pre-existing one-minute gate
still has its previously documented weekday/hour checks.

Run `node --test scenario-engine/test/*.test.js`, followed by Apex's candle and
setup regression fixtures. These are unbacktested classification rules, not
claims of predictive accuracy.

## Scenario data, checkpoint history and 0DTE screening (adapter 1.1.0)

- `/api/scenario/calendars` transports fixed official New York Fed and Federal Reserve documents. Browser parsers retain server receipt times, require complete recognized layouts, and reject stale/partial calendars. Coverage is limited to the NY Fed indicator calendar plus verified FOMC decision/conference times. It excludes minutes, speeches, unscheduled news and headlines. Known upcoming events survive temporary feed failure.
- QQQ/IWM completed minute candles provide cross-asset returns. They do not represent constituent breadth. Intraday breadth, value profile and positioning remain unavailable; the existing daily breadth endpoint is not relabeled intraday.
- IndexedDB retains up to 500 checkpoint records, at most one per minute while the panel receives candles. Each includes the exact dataset, pre-evaluation engine state, resulting card, entry status and contract screen. Verify replay reproduces those decisions; it is not an options backtest. Export downloads the saved inputs. Storage failure is displayed, and no arrival times are backdated after reload.
- `/api/scenario/spy-contracts` uses the existing Polygon account for same-day SPY option snapshots within $10 of the whole-dollar spot centre. Access depends on the account's options entitlement. No subscription is purchased. Missing/delayed quotes remain unavailable.
- Screening requires a real-time quote no older than 30 seconds, standard 100-share contracts, matching direction/expiry, positive two-sided sizes, spread <=10% of midpoint, delta magnitude 0.35–0.65, and volume/prior-day OI >=100. These are research defaults, not performance-calibrated thresholds. Quotes never auto-fill confirmation or grant entry permission. Screening runs from 9:45 ET to 15 minutes before scheduled close.
- API reference: https://massive.com/docs/rest/options/snapshots/option-chain-snapshot

Validation: `node --test scenario-engine/test/*.test.js`; existing candle/setup and original Apex fixture scripts. Backend: `python -m pytest api/test_scenario_feeds.py data/test_intraday_cache.py utils/test_cache.py -q`.

Live verification on 2026-09-10: deployed official calendar transport parsed 45 entries and passed the scoped coverage check. Browser saved checkpoints and replay reproduced 3/3. The deployed options snapshot endpoint returned unavailable; live contract qualification could not be validated with the current provider response. Automated fixtures verify quote-age, delay, spread, liquidity, direction and expiry rejection. Full intraday breadth/value-profile/positioning integration still requires suitable source data.


## UW correction (adapter 1.2.0)

The contract endpoint now uses the existing UNUSUAL_WHALES_API_KEY and UW `/api/stock/SPY/option-contracts`, replacing the Polygon path described above. It requests today’s expiry with up to two 500-row pages and filters strikes within $10 of supplied spot. It never substitutes another expiry or provider.

UW bid/ask, delta, volume and OI support a REVIEW shortlist. Last tape time is trade activity, not a quote timestamp. Quote timestamps, sizes and deliverables are absent in the observed UW response, so the adapter leaves them null. REVIEW contracts require broker confirmation and never become fully verified candidates or grant entry permission. The UI displays provider status even after hours.

## UW positioning (adapter 1.3.0)

The existing `/api/greeks/SPY/spot-strike?date=YYYY-MM-DD` route supplies the current Eastern date's all-expiry profile. No new subscription or backend credential is needed. The adapter refreshes once per minute independently of candles. It requires source timestamps within ten minutes and receipt within two minutes, rejects malformed/duplicate rows and potentially truncated 500-row profiles, and excludes entirely zero-exposure rows from freshness. Nonzero call/put contributions that offset still require fresh timestamps.

The explicit model sums `callGammaOi - abs(putGammaOi)` and supplies its sign plus the five largest absolute net strike concentrations. This assumes positive call and negative put contributions; it is not observed dealer inventory. These are positioning references, not a calibrated gamma flip or standalone entry signals. Existing price, session, candle freshness and entry gates still apply. Checkpoints include the evidence and its actual receipt time.

Live source verification on 2026-09-10 returned 483 strike rows. After-hours observations correctly remain stale and contribute no evidence. Value-profile levels and intraday breadth remain unavailable. Earlier sections above describe historical rollout states; scoped calendar coverage and UW options are now connected.

## Manual SPY value profile (adapter 1.4.0)

The scenario panel accepts chart-transcribed VAL, POC and VAH from the previous completed SPY regular trading session, with a 70% value area. Enter both profile and target session dates, chart source and row settings. POC maps to the engine's VPOC. These inputs are explicitly manual and not independently verified; they do not replace candle or entry confirmation.

Only the immediately previous trading session is accepted, using the exchange holiday/early-close calendar. Require positive VAL ≤ POC ≤ VAH and distinct boundaries. Source close must precede receipt; receipt must precede evaluation. Levels expire at target session close. Tomorrow's profile can be saved after today's close; it remains inactive until the target date. Local storage holds one profile, and reload rebases its receipt to the current time. Clearing removes it from subsequent evaluations. Exact evidence is retained in checkpoint datasets for replay. Form drafts and focus survive live panel refreshes.

UW's stock-volume-price-levels endpoint documents Nasdaq-operated exchange volume and FINRA off-exchange volume, not a complete consolidated regular-session profile. It is therefore not substituted for chart VAH/VAL/POC: https://api.unusualwhales.com/docs/operations/PublicApi.TickerController.stock_volume_price_level
