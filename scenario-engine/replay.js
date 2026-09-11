import { ScenarioEngine } from './engine.js';
import { ms, iso } from './time.js';
import { validateDataset } from './model.js';
import { sign } from './acceptance.js';

/** Replay every five-minute checkpoint, including empty intervals to reveal outages. */
export function replay(dataset, options = {}) {
  validateDataset(dataset);
  const checkpoints = options.checkpoints ?? (() => {
    const times = [iso(ms(dataset.session.open) - 300000)];
    for (let t = ms(dataset.session.open); t <= ms(dataset.session.close); t += 300000) times.push(iso(t));
    return times;
  })();
  const engine = new ScenarioEngine(options.config);
  const cards = checkpoints.map(at => engine.evaluate(dataset, at));
  return { session: dataset.session.id, data_label: dataset.label ?? 'User-supplied data', config: engine.config, cards,
    summary: { checkpoints: cards.length, permitted: cards.filter(c => c.trade_permitted).length,
      transitions: engine.history.filter(t => t.from !== t.to || t.fromDirection !== t.toDirection),
      grade_counts: Object.fromEntries(['A', 'B', 'C', 'D'].map(g => [g, cards.filter(c => c.confidence_grade === g).length])) } };
}

/** Forward outcomes are computed AFTER classification. They never feed the engine.
 * One sample per permitted scenario episode, entry at next bar open; not option P&L.
 */
export function backtest(dataset, replayResult, horizonMinutes = 30) {
  if (!Number.isFinite(horizonMinutes) || horizonMinutes <= 0) throw new Error('Positive outcome horizon required');
  const bars = [...dataset.bars].sort((a, b) => ms(a.start) - ms(b.start));
  const samples = [];
  let previousKey = null;
  for (const card of replayResult.cards) {
    const key = card.trade_permitted ? `${card.scenario_id}:${card.directional_bias}` : null;
    if (!key || key === previousKey) { previousKey = key; continue; }
    previousKey = key;
    const entry = bars.find(b => ms(b.start) >= ms(card.timestamp) && ms(b.start) < ms(dataset.session.close));
    if (!entry || ms(entry.start) - ms(card.timestamp) > 300000) continue;
    const end = Math.min(ms(entry.start) + horizonMinutes * 60000, ms(dataset.session.close));
    const forward = bars.filter(b => ms(b.start) >= ms(entry.start) && ms(b.end) <= end);
    const complete = forward.length === (end - ms(entry.start)) / 300000 && forward.every((b, i) => ms(b.start) === ms(entry.start) + i * 300000);
    const d = sign(card.directional_bias), target = card.target_ladder[0]?.price;
    const targetReachable = target !== undefined && d * (target - entry.open) > 0;
    const targetBar = targetReachable ? forward.find(b => d === 1 ? b.high >= target : b.low <= target) : null;
    const structuralLevel = card.invalidation.find(i => i.kind === 'opposite-acceptance')?.price;
    const adverseBar = forward.find(b => structuralLevel !== undefined && (d === 1 ? b.low <= structuralLevel : b.high >= structuralLevel));
    const moves = forward.flatMap(b => [d * (b.high - entry.open), d * (b.low - entry.open)]);
    samples.push({ signal_at: card.timestamp, entry_at: entry.start, scenario: card.scenario_id, direction: card.directional_bias,
      grade: card.confidence_grade, spy_entry: entry.open, horizon_minutes: horizonMinutes, complete,
      target_already_passed_at_entry: !targetReachable,
      mfe_spy: complete ? Math.max(0, ...moves) : null, mae_spy: complete ? Math.min(0, ...moves) : null,
      end_change_spy: complete ? d * (forward.at(-1).close - entry.open) : null,
      first_target_touch_at: complete ? targetBar?.end ?? null : null,
      reference_touch_at: complete ? adverseBar?.end ?? null : null,
      ambiguous_same_bar: !!(targetBar && adverseBar && targetBar.end === adverseBar.end) });
  }
  return { methodology: 'Descriptive SPY forward excursions, next-bar entry, fixed horizon. Reference touches are not mechanical invalidations. Overlapping episodes may correlate. No fills, slippage, commissions, option premiums, win rates, or calibrated probabilities.', samples };
}
