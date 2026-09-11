import test from 'node:test';
import assert from 'node:assert/strict';
import { ScenarioEngine, acceptance, sweep, normalize, configure, catalogue, replay, backtest, validateDataset } from '../index.js';
import { phase, eastern } from '../time.js';
import { evidence, vwapCrossings } from '../evidence.js';
import { at, bar, level, observation, dataset, refreshed } from './fixtures.js';

const cfg = configure();
const view = (data, time) => normalize(validateDataset(data), time, cfg);
const only = id => ({ scenarios: Object.fromEntries(catalogue(cfg).filter(r => r.level && r.id !== id).map(r => [r.id, { enabled: false }])) });

test('two consecutive closed 5m bars accept a level in either direction', () => {
  for (const [direction, closes] of [['bullish', [100.2, 100.3]], ['bearish', [99.8, 99.7]]]) {
    const result = acceptance(view(dataset(closes), at(10)), level('TEST', 100), direction, cfg);
    assert.equal(result.status, 'accepted'); assert.equal(result.method, 'two-5m-closes');
  }
});
test('one close is pending; wick is a sweep, not acceptance', () => {
  const data = dataset([99.9]); data.bars[0].high = 100.5;
  assert.equal(acceptance(view(data, at(5)), level('TEST', 100), 'bullish', cfg).status, 'sweep');
  assert.equal(acceptance(view(dataset([100.2]), at(5)), level('TEST', 100), 'bullish', cfg).status, 'pending');
});
test('15m close followed by retest qualifies only after the retest closes', () => {
  const data = dataset([99.9, 99.9, 100.2, 100.005, 100.1]);
  data.bars[3].low = 99.98; data.bars[4].low = 99.99;
  assert.notEqual(acceptance(view(data, at(20)), level('TEST', 100), 'bullish', cfg).status, 'accepted');
  const result = acceptance(view(data, at(25)), level('TEST', 100), 'bullish', cfg);
  assert.equal(result.status, 'accepted'); assert.equal(result.method, '15m-close-and-retest');
});
test('misaligned 15m aggregation and nonconsecutive bars cannot accept', () => {
  const data = dataset([100.2, 100.3, 100.4]); data.bars.splice(1, 1);
  assert.notEqual(acceptance(view(data, at(15)), level('TEST', 100), 'bullish', cfg).status, 'accepted');
});
test('unclosed, delayed, and future-level data cannot confirm acceptance', () => {
  const data = dataset([100.2, 100.3]); data.bars[1].available_at = at(20);
  assert.notEqual(acceptance(view(data, at(10)), level('TEST', 100), 'bullish', cfg).status, 'accepted');
  const result = acceptance(view(dataset([100.2, 100.3]), at(10)), level('TEST', 100, { available_at: at(5) }), 'bullish', cfg);
  assert.equal(result.status, 'pending');
});
test('sweep reversal requires a close back through the swept price', () => {
  const data = dataset([100.2, 100.3]); data.bars[0].open = 100.1; data.bars[0].low = 99.5;
  assert.equal(sweep(view(data, at(10)), level('TEST', 100), 'bullish', cfg).extreme, 99.5);
  data.bars[0].close = 99.8;
  assert.equal(sweep(view(data, at(10)), level('TEST', 100), 'bullish', cfg), null);
});
test('OR15, OR30 and initial balance are unavailable until complete', () => {
  const data = dataset(Array.from({ length: 12 }, (_, i) => 100 + i * 0.1));
  assert.equal(view(data, at(10)).getLevel('OR15H'), undefined);
  assert.equal(view(data, at(15)).getLevel('OR15H').value, 100.3);
  assert.equal(view(data, at(25)).getLevel('OR30H'), undefined);
  assert.ok(view(data, at(30)).getLevel('OR30H'));
  assert.equal(view(data, at(55)).getLevel('IBH'), undefined);
  assert.ok(view(data, at(60)).getLevel('IBH'));
  data.bars.splice(1, 1);
  assert.equal(view(data, at(60)).getLevel('IBH'), undefined);
});
test('opening range acceptance cannot use bars that formed the range', () => {
  const data = dataset([100, 100.2, 100.4, 100.8, 101]);
  const s = view(data, at(25));
  assert.equal(acceptance(s, s.getLevel('OR15H'), 'bullish', cfg).status, 'accepted');
  assert.notEqual(acceptance(view(data, at(15)), s.getLevel('OR15H'), 'bullish', cfg).status, 'accepted');
});
test('a confirmed breakout invalidates into a failed breakout, not a generic opposite alert', () => {
  const data = refreshed(dataset([100, 100.2, 100.4, 100.8, 101, 100.3, 100.2]));
  // Negative direction evidence must agree once price returns inside the OR.
  const change = [observation('adRatio', 0.5, 35), observation('sp500AboveVwapPct', 30, 35), observation('upDownVolumeRatio', 0.5, 35), observation('qqqReturnPct', -0.5, 35), observation('iwmReturnPct', -0.4, 35), observation('vixChangePct', 1, 35)];
  data.observations.push(...change);
  data.levels.push(level('VWAP', 100.6, { as_of: at(35), available_at: at(35) }));
  const engine = new ScenarioEngine({ scenarios: Object.fromEntries(catalogue(cfg).filter(r => r.level && !['opening-range-breakout', 'failed-opening-range-breakout'].includes(r.id)).map(r => [r.id, { enabled: false }])) });
  const first = engine.evaluate(data, at(25));
  assert.equal(first.scenario_id, 'opening-range-breakout'); assert.equal(first.directional_bias, 'bullish');
  const pending = engine.evaluate(data, at(30));
  assert.equal(pending.scenario_id, 'opening-range-breakout'); assert.equal(pending.trade_permitted, false); assert.equal(pending.trigger_status, 'waiting');
  const second = engine.evaluate(data, at(35));
  assert.equal(second.scenario_id, 'failed-opening-range-breakout'); assert.equal(second.directional_bias, 'bearish');
  assert.equal(second.explanation.transition.type, 'invalidation');
});
test('invalidation without qualified successor goes to explicit no-trade', () => {
  const data = refreshed(dataset([100, 100.2, 100.4, 100.8, 101, 100.3, 100.2]));
  const engine = new ScenarioEngine(only('opening-range-breakout'));
  assert.equal(engine.evaluate(data, at(25)).scenario_id, 'opening-range-breakout');
  const card = engine.evaluate(data, at(35));
  assert.equal(card.scenario_id, 'no-trade-chop'); assert.equal(card.trade_permitted, false);
});
test('conflicting internals veto an otherwise accepted SPY breakout', () => {
  const data = refreshed(dataset()); data.observations.push(observation('qqqReturnPct', -0.8, 25));
  const card = new ScenarioEngine().evaluate(data, at(25));
  assert.equal(card.trade_permitted, false); assert.ok(card.decision_reasons.includes('Conflicting market internals'));
});
test('breadth measures count as one category and cannot alone earn grade A', () => {
  const data = dataset(); data.observations = ['adRatio', 'sp500AboveVwapPct', 'nyseTickMean', 'nasdaqTickMean'].map((k, i) => observation(k, [2, 80, 400, 400][i], 25));
  const e = evidence(view(data, at(25)), 'bullish', cfg);
  assert.equal(e.independentCount, 1); assert.equal(e.highEligible, false);
});
test('weak relative volume yields explicit no-trade', () => {
  const data = refreshed(dataset()); data.observations.push(observation('relativeVolume', 0.5, 25));
  assert.ok(new ScenarioEngine().evaluate(data, at(25)).decision_reasons.includes('Weak SPY relative volume'));
});
test('repeated VWAP crossings use contemporaneous VWAP versions', () => {
  const data = dataset([100.2, 99.8, 100.2, 99.8, 100.2]);
  data.levels = data.levels.filter(l => l.kind !== 'VWAP');
  data.levels.push(...data.bars.map(b => level('VWAP', 100, { as_of: b.end, available_at: b.end })));
  assert.equal(vwapCrossings(view(data, at(25)), cfg).crossings, 4);
  assert.equal(new ScenarioEngine().evaluate(data, at(25)).trade_permitted, false);
});
test('events, uncertain headlines, and missing coverage fail closed', () => {
  const data = refreshed(dataset());
  data.events.push({ id: 'ISM', type: 'ISM', at: at(30), available_at: at(-60), source: 'synthetic:calendar', status: 'scheduled' });
  const eventCard = new ScenarioEngine().evaluate(data, at(25));
  assert.equal(eventCard.scenario_id, 'event-risk'); assert.deepEqual(eventCard.conflicting_evidence, []);
  data.events = []; data.eventCoverage = [];
  assert.equal(new ScenarioEngine().evaluate(data, at(25)).scenario_id, 'event-risk');
});
test('stale price and missing mandatory fields never permit a trade', () => {
  assert.equal(new ScenarioEngine().evaluate(dataset(), at(60)).trade_permitted, false);
  const data = refreshed(dataset()); data.levels = data.levels.filter(l => l.kind !== 'VWAP');
  assert.ok(new ScenarioEngine().evaluate(data, at(25)).decision_reasons.includes('Session VWAP unavailable'));
});
test('future input changes cannot alter an earlier decision', () => {
  const data = refreshed(dataset());
  const card = new ScenarioEngine().evaluate(data, at(20));
  data.bars[4] = bar(20, 120, { open: 100.8 });
  data.observations.push(observation('adRatio', 0.01, 25));
  data.levels.push(level('VAH', 200, { as_of: at(25), available_at: at(25) }));
  assert.deepEqual(new ScenarioEngine().evaluate(data, at(20)), card);
});
test('replay is deterministic and prefix invariant', () => {
  const data = refreshed(dataset([100, 100.2, 100.4, 100.8, 101, 101.2]));
  const checkpoints = [at(10), at(15), at(20), at(25), at(30)];
  const full = replay(data, { checkpoints });
  assert.deepEqual(full, replay(data, { checkpoints }));
  assert.deepEqual(full.cards.slice(0, 3), replay(data, { checkpoints: checkpoints.slice(0, 3) }).cards);
});
test('expired scenarios leave the old state even if its price level still holds', () => {
  const data = refreshed(dataset(Array.from({ length: 12 }, (_, i) => 100 + i * 0.2)));
  const config = only('opening-range-breakout'); config.scenarios['opening-range-breakout'] = { time_expiry: { minutes: 5, phaseEnd: true } };
  const engine = new ScenarioEngine(config);
  assert.equal(engine.evaluate(data, at(25)).scenario_id, 'opening-range-breakout');
  assert.equal(engine.evaluate(data, at(30)).scenario_id, 'no-trade-chop');
});
test('same timestamp is idempotent; backward timestamps rejected', () => {
  const data = refreshed(dataset()); const engine = new ScenarioEngine();
  const first = engine.evaluate(data, at(25));
  assert.deepEqual(engine.evaluate(data, at(25)), first); assert.equal(engine.history.length, 1);
  assert.throws(() => engine.evaluate(data, at(20)), /chronological/);
});
test('positioning cannot override unconfirmed price or create probability fields', () => {
  const data = refreshed(dataset([100])); data.levels.push(level('GAMMA_FLIP', 99), level('STRIKE', 105));
  data.observations.push(observation('dealerGammaSign', -1, 5));
  const card = new ScenarioEngine().evaluate(data, at(5));
  assert.equal(card.trade_permitted, false); assert.equal('probability' in card, false);
});
test('directional target ladder contains only SPY prices ahead of price', () => {
  const card = new ScenarioEngine(only('opening-range-breakout')).evaluate(refreshed(dataset()), at(25));
  assert.equal(card.trade_permitted, true);
  assert.ok(card.target_ladder.length > 0);
  assert.ok(card.target_ladder.every((t, i, arr) => t.price > 101 && (!i || t.price >= arr[i - 1].price)));
});
test('no external target means no trade', () => {
  const data = refreshed(dataset()); data.levels = data.levels.filter(l => l.value < 101 || l.kind === 'ATR');
  const card = new ScenarioEngine(only('opening-range-breakout')).evaluate(data, at(25));
  assert.equal(card.trade_permitted, false); assert.ok(card.decision_reasons.includes('No supplied SPY target ahead of current price'));
});
test('strict transport validation catches malformed bars and missing provenance', () => {
  const data = dataset(); data.bars[0].low = 500;
  assert.throws(() => validateDataset(data), /OHLCV/);
  const d2 = dataset(); delete d2.levels[0].method;
  assert.throws(() => validateDataset(d2), /methodology/);
  const d3 = dataset(); d3.bars.push(d3.bars[0]);
  assert.throws(() => validateDataset(d3), /Duplicate/);
});
test('ambiguous singleton levels and externally redefined opening levels are rejected', () => {
  const data = dataset(); data.levels.push(level('PDC', 105, { id: 'conflicting-pdc' }));
  assert.throws(() => view(data, at(25)), /Ambiguous PDC/);
  const other = dataset(); other.levels.push(level('OPEN', 110));
  assert.throws(() => view(other, at(25)), /engine-derived/);
});
test('DST, checkpoints, early-close and nontrading sessions use explicit calendar', () => {
  assert.equal(eastern('2026-01-05T14:30:00Z').minute, 570);
  assert.equal(eastern('2026-09-04T13:30:00Z').minute, 570);
  const s = dataset().session;
  assert.deepEqual([0, 15, 60, 150, 270, 330, 390].map(t => phase(at(t), s)), ['opening', 'confirmation', 'morning', 'midday', 'afternoon', 'power-hour', 'closed']);
  assert.equal(phase(at(210), { ...s, close: at(210) }), 'closed');
  assert.equal(phase(at(30), { ...s, isTradingDay: false }), 'closed');
});
test('all directional scenarios are bilateral and all successors exist', () => {
  const rules = catalogue(cfg); assert.equal(rules.length, 18);
  for (const r of rules) {
    assert.ok(rules.some(other => other.id === r.next_state_if_invalidated));
    for (const key of ['scenario', 'direction', 'preconditions', 'trigger', 'confirmation', 'expected_path', 'target_ladder', 'invalidation', 'time_expiry', 'confidence_grade', 'next_state_if_invalidated']) assert.ok(key in r);
    if (!['no-trade-chop', 'event-risk', 'late-day-strike-pin'].includes(r.id)) assert.equal(r.direction, 'both');
  }
});
test('backtest starts at next bar open and flags incomplete horizons', () => {
  const data = refreshed(dataset([100, 100.2, 100.4, 100.8, 101, 101.2, 101.4]));
  const run = replay(data, { checkpoints: [at(25)], config: only('opening-range-breakout') });
  const report = backtest(data, run);
  assert.equal(report.samples[0].entry_at, at(25));
  assert.equal(report.samples[0].spy_entry, 101);
  assert.equal(report.samples[0].complete, false); assert.equal(report.samples[0].mfe_spy, null);
});
test('config rejects typo keys and invalid values', () => {
  assert.throws(() => configure({ accepted: true }), /Unknown/);
  assert.throws(() => configure({ acceptance: { buffer: -1 } }), /Invalid/);
  assert.throws(() => new ScenarioEngine({ scenarios: { invented: {} } }), /Unknown/);
  assert.throws(() => new ScenarioEngine({ scenarios: { 'open-drive': { trigger: ['invented'] } } }), /Unknown rule predicate/);
  assert.throws(() => configure({ evidence: { highConfidenceCategories: 1 } }), /multiple categories/);
});
