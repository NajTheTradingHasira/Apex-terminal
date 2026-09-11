# Apex-terminal

## Scenario library — 18 templates

SPY Logic now includes the original five-minute scenario engine from the other
project, with bullish/bearish templates plus neutral pin/chop/event-risk states.
Expand **Scenario library** to see phase eligibility, individual checks,
expected path, invalidation and successor. The live layer receives aggregated
SPY candles, estimated VWAP, a labelled within-session volume ratio, prior-day
OHLC and daily ATR. It does not invent value-profile, breadth, positioning or
event-coverage inputs.

The library is live scenario context by default. **Require scenario approval
for new entries** makes it an additional downward-only gate; missing verified
event coverage currently prevents that stricter gate from approving entries.
The existing one-minute entry detector remains in place. See
[integration details](scenario-engine/README.md) and run
`node --test scenario-engine/test/*.test.js` for source and adapter checks.

## SPY automatic candle detection

In **AI Analysis → SPY Logic**, Automatic mode polls the existing
`/api/stock/SPY/intraday?interval=1m` endpoint every 30 seconds while the
panel is open during regular hours. Choose the VWAP or opening-range playbook.
The detector fills SPY trigger, invalidation, target and opening-range levels,
and derives VWAP state, the 5/10/20 EMA ribbon and the retest input. Switch to
Manual mode to use chart-entered levels instead.

Automatic confirmation requires a breakout/reclaim close, a separate retest
within five bars, then a later candle CLOSE beyond the retest extreme. Both
long and short directions require EMA alignment, a higher low/lower high,
confirmation volume at least 1.2× the preceding 20 bars, at least 1.5R remaining,
and no more than 0.5R chase. A later stop breach or close back across the level
invalidates the candidate. Candidates expire three minutes after confirmation.
The target is a labelled 2R projection, capped at the pre-breakout session
extreme where relevant; it is not an option profit forecast.

Only completed one-minute bars (plus five seconds publication grace) count.
The first 25 session bars are warmup. Missing session bars, conflicting
duplicates, bad OHLCV, prior-day data or a latest candle close over two minutes
old block detection. Four or more VWAP crosses in ten bars suppress setups.
VWAP is estimated from volume-weighted typical candle prices; EMA is seeded
from the session's first close. No future or forming bar contributes.

Contract bid, ask, expiry and market internals still require confirmation.
The selected contract must expire today and its spread must be at most 10%
of midpoint. Confirmation expires after five minutes; edits, a changed
candidate or feed failure clear it. Automatic detection never promotes an
existing NO-GO/CAUTION bias or gamma gate, and it must agree with that bias.
Underlying R:R does not predict option returns. These rules are unbacktested
heuristics, not probabilities of success or order execution.

The feed is Yahoo via Nexus; plan/provider delays can prevent current signals.
The backend companion change gives intraday data its own 30-second cache
instead of the daily five-minute cache. Frontend freshness checks work with
either backend version and always use candle timestamps. Weekday/ET window
checks are not an exchange holiday/early-close calendar. In manual mode,
freshness still uses the summary API timestamp, not an exchange quote timestamp.

Run `node sl-candle-fixtures.mjs`, `node sl-intraday-fixtures.mjs`,
`node sl-pivot-fixtures.mjs`, `node gex-regime-fixtures.mjs`,
`node uw-status-fixtures.mjs`, and `node wl-stage-fixtures.mjs`.
