# Apex-terminal

## SPY intraday setup checks

In **AI Analysis → SPY Logic**, select a VWAP reclaim/rejection or a
15-minute opening-range break/retest. Set the tape inputs and enter SPY
trigger, invalidation and target levels, plus the selected option's bid,
ask and expiration. ORB also needs the completed 09:30–09:45 ET range.
Confirm the completed retest candle and current inputs on your live chart.

The setup layer can restrict the existing bias/gamma gate, never promote it.
WAIT prevents new-entry eligibility; CAUTION preserves existing restrictions;
READY means the entered evidence passes the checks, not an order or a
probability of success. Editing tape inputs or setup fields clears confirmation;
confirmation expires in five minutes and is not persisted across reloads.

Reviewable, unbacktested defaults: at least 1.5R remaining at the worse of
trigger/current price, no more than 0.5R chase, and option spread no wider than
10% of midpoint. These are underlying-price calculations, not option-return
projections. Existing premium stops, structure and gamma controls still apply.

The existing API supplies snapshots, not candle or volume history or contract
quotes. Retests, opening range and contract quotes therefore require manual
confirmation. The freshness check uses the backend timestamp (maximum two
minutes); it does not independently establish the exchange quote time or data
delay. Session checks use weekday/ET hours, not an exchange holiday/early-close
calendar. Verify market hours and feed latency before use.

Run regression checks with `node sl-intraday-fixtures.mjs`,
`node sl-pivot-fixtures.mjs`, `node gex-regime-fixtures.mjs`,
`node uw-status-fixtures.mjs`, and `node wl-stage-fixtures.mjs`.
