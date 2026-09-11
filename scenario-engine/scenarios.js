/** Serializable rule catalogue. No UI state or option premiums participate in rules. */
const definition = (id, name, regime, priority, windows, level, conditions, path, next = 'inside-value-rotation') => ({
  id, scenario: name, regime, priority, windows, level,
  direction: 'both', preconditions: conditions, trigger: ['acceptance'],
  confirmation: ['independent-evidence', 'no-global-veto'], expected_path: path,
  target_ladder: ['nearest-eligible-SPY-levels'], invalidation: ['opposite-acceptance'],
  time_expiry: { minutes: 45, phaseEnd: true }, confidence_grade: 'evidence-graded-A-B-C-D',
  next_state_if_invalidated: next
});
export const scenarioDefinitions = [
  definition('gap-and-go', 'Gap-and-go', 'Continuation', 70, ['opening', 'confirmation'], 'gap-edge', ['gap-aligned', 'outside-value', 'vwap-aligned'], ['Hold the accepted gap edge', 'Extend toward overnight and weekly structure']),
  definition('gap-rejection-fill', 'Gap rejection / gap fill', 'Reversal', 85, ['opening', 'confirmation', 'morning'], 'OPEN', ['gap-opposed', 'prior-gap-extension'], ['Reject the opening gap', 'Rotate toward previous close; continuation beyond the fill needs new acceptance']),
  definition('gap-reclaim-rejection', 'Gap-down reclaim / gap-up rejection', 'Reversal', 88, ['opening', 'confirmation', 'morning'], 'PDC', ['gap-opposed'], ['Accept back through previous close', 'Test the opposite value boundary']),
  definition('inside-value-rotation', 'Inside-value rotation', 'Balance / mean reversion', 30, ['opening', 'confirmation', 'morning', 'midday', 'afternoon', 'power-hour'], 'value-edge', ['inside-value', 'sweep'], ['Hold the reclaimed value boundary', 'Rotate to VPOC, then the opposite value edge'], 'no-trade-chop'),
  definition('open-drive', 'Open drive', 'Continuation', 65, ['opening', 'confirmation'], 'OPEN', ['drive', 'vwap-aligned'], ['Maintain directional opening pressure', 'Test the next SPY structural level']),
  definition('open-test-drive', 'Open-test-drive', 'Continuation', 75, ['opening', 'confirmation'], 'OPEN', ['opening-test', 'drive', 'vwap-aligned'], ['Hold the opening test', 'Resume the drive toward the next SPY level']),
  definition('opening-range-breakout', 'Opening-range breakout', 'Continuation', 78, ['confirmation', 'morning', 'afternoon'], 'or-edge', ['vwap-aligned'], ['Hold acceptance beyond the completed opening range', 'Expand toward IB and external structure'], 'failed-opening-range-breakout'),
  definition('failed-opening-range-breakout', 'Failed opening-range breakout', 'Reversal', 96, ['confirmation', 'morning', 'midday', 'afternoon', 'power-hour'], 'failed-or-edge', ['failed-or'], ['Accept back inside the opening range', 'Rotate to VWAP, then the opposite opening-range boundary']),
  definition('trend-pullback-continuation', 'Trend-pullback continuation', 'Continuation', 74, ['morning', 'afternoon'], 'pullback-level', ['prior-trend', 'pullback', 'vwap-aligned'], ['Hold the tested support or resistance', 'Resume toward the session extreme and next structural level']),
  definition('liquidity-sweep-reversal', 'Liquidity-sweep reversal', 'Reversal', 94, ['opening', 'confirmation', 'morning', 'midday', 'afternoon', 'power-hour'], 'swept-major', ['sweep'], ['Accept back through the swept level', 'Rotate toward VWAP and the next opposing liquidity level']),
  definition('midday-compression-breakout', 'Midday compression breakout', 'Continuation', 72, ['midday', 'afternoon'], 'compression-edge', ['compression', 'vwap-aligned', 'strong-volume'], ['Accept outside the prior compression box', 'Expand only while participation persists']),
  definition('breadth-divergence-reversal', 'Breadth-divergence reversal', 'Reversal', 92, ['morning', 'midday', 'afternoon', 'power-hour'], 'swept-major', ['sweep', 'breadth-divergence'], ['Price rejects an extreme as breadth diverges', 'Accept back toward VWAP before an opposing trend can develop']),
  definition('event-repricing', 'Event repricing', 'Continuation', 98, ['confirmation', 'morning', 'midday', 'afternoon', 'power-hour'], 'event-edge', ['resolved-event', 'post-event-acceptance', 'strong-volume'], ['Accept beyond the pre-event reference range', 'Reprice toward the next SPY level; reassess macro response']),
  definition('power-hour-continuation', 'Power-hour continuation', 'Continuation', 82, ['power-hour'], 'trend-edge', ['prior-trend', 'vwap-aligned', 'strong-volume'], ['Hold afternoon expansion', 'Continue toward session and weekly objectives before the close']),
  definition('late-day-unwind', 'Late-day unwind / reversal', 'Reversal', 91, ['power-hour'], 'late-reference', ['late-unwind'], ['Lose the established afternoon reference', 'Unwind toward VWAP and opposing structure']),
  { ...definition('late-day-strike-pin', 'Late-day strike pin', 'Balance / mean reversion', 40, ['power-hour'], 'nearest-strike', ['pin'], ['Rotate around the supplied strike while price remains balanced'], 'no-trade-chop'), direction: 'neutral', trigger: ['pin'], invalidation: ['range-expansion'], time_expiry: { minutes: 30, phaseEnd: true } },
  { ...definition('no-trade-chop', 'No-trade chop', 'Event risk / no trade', 0, ['premarket', 'opening', 'confirmation', 'morning', 'midday', 'afternoon', 'power-hour', 'closed'], null, [], ['Wait for coherent internals and fresh SPY acceptance'], 'no-trade-chop'), direction: 'neutral', trigger: [], invalidation: ['quality-restored'] },
  { ...definition('event-risk', 'Event risk / no trade', 'Event risk / no trade', 100, ['premarket', 'opening', 'confirmation', 'morning', 'midday', 'afternoon', 'power-hour'], null, [], ['Wait for the event window to clear and fresh post-event acceptance'], 'no-trade-chop'), direction: 'neutral', trigger: [], invalidation: ['event-resolved'] }
];

export function catalogue(config) {
  const ids = new Set(scenarioDefinitions.map(s => s.id));
  for (const id of Object.keys(config.scenarios)) if (!ids.has(id)) throw new Error(`Unknown scenario override: ${id}`);
  const allowed = new Set(['enabled', 'priority', 'windows', 'level', 'preconditions', 'trigger', 'expected_path', 'time_expiry', 'next_state_if_invalidated']);
  const predicates = new Set(['acceptance', 'gap-aligned', 'gap-opposed', 'outside-value', 'inside-value', 'vwap-aligned', 'prior-gap-extension', 'drive', 'opening-test', 'sweep', 'prior-trend', 'pullback', 'failed-or', 'compression', 'strong-volume', 'breadth-divergence', 'resolved-event', 'post-event-acceptance', 'late-unwind', 'pin']);
  for (const [id, patch] of Object.entries(config.scenarios)) {
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new Error(`Invalid override for ${id}`);
    for (const key of Object.keys(patch)) if (!allowed.has(key)) throw new Error(`Unsupported scenario override ${id}.${key}`);
  }
  const rules = scenarioDefinitions.map(s => ({ ...structuredClone(s), ...config.scenarios[s.id] }));
  for (const rule of rules) {
    if (!ids.has(rule.next_state_if_invalidated)) throw new Error(`Unknown next state for ${rule.id}`);
    if (!Number.isFinite(rule.priority) || !Array.isArray(rule.preconditions) || !Array.isArray(rule.windows) || !Array.isArray(rule.trigger)) throw new Error(`Invalid rule: ${rule.id}`);
    if (!['both', 'neutral'].includes(rule.direction)) throw new Error(`Invalid direction: ${rule.id}`);
    if (rule.enabled !== undefined && typeof rule.enabled !== 'boolean') throw new Error(`Invalid enabled flag: ${rule.id}`);
    if (!Number.isFinite(rule.time_expiry?.minutes) || rule.time_expiry.minutes <= 0 || typeof rule.time_expiry.phaseEnd !== 'boolean') throw new Error(`Invalid expiry: ${rule.id}`);
    if (rule.windows.some(w => !['premarket', 'opening', 'confirmation', 'morning', 'midday', 'afternoon', 'power-hour', 'closed'].includes(w))) throw new Error(`Invalid time window: ${rule.id}`);
    for (const name of [...rule.preconditions, ...rule.trigger]) if (!predicates.has(name)) throw new Error(`Unknown rule predicate: ${name}`);
  }
  return rules;
}
