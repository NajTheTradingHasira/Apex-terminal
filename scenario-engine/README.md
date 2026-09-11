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
