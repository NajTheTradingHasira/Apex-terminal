# Delayed breadth context

The Market Breadth panel uses the deployed `/api/scenario/breadth` endpoint, sourced from Massive minute aggregates and State Street SPY equity holdings. No subscription or credentials are changed by this frontend integration.

`delayed-breadth.js` requires complete provider status, 98% coverage of a 490–510 holding universe, consistent counts, numeric metrics and timezone-qualified source/receipt timestamps. A same-session source age of 15–25 minutes permits background AI context only. Older observations remain visibly historical and have their metrics withheld from current AI context. Missing denominators remain unavailable, not zero or infinity. Partial coverage is withheld. Fetch failure clears the client payload.

No observations are passed to the scenario engine, BD stays unavailable, and entry gates are unchanged. Index/ETF analysis receives a separate delayedBreadthContext with an explicit prohibition on treating it as live confirmation. The panel refreshes once per minute while breadth or AI is open; freshness is reevaluated when AI context is built. No reconstructed history or synthetic chart is provided.

The provider computes equal-weight A/D versus previous close, regular-session volume assigned by advancing/declining stock, and above-VWAP percentage from provider minute bars. These are SPY holdings proxy measurements, not exchange-wide breadth or uptick/downtick volume. Coverage represents the backend's accepted symbols, not an independent audit of every underlying bar.

Verification: delayed adapter fixtures, 129 scenario tests, 62 candle and 49 entry checks. Browser on 2026-09-13 returned 502/502 coverage for September 11 close and correctly labeled it historical with no current AI metrics. No browser errors. Real-time market-session behavior remains to be observed.
