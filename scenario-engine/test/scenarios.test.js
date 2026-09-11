import test from 'node:test';
import assert from 'node:assert/strict';
import { configure, catalogue } from '../index.js';
import { evaluateRule } from '../rules.js';
import { at, bar, level, observation } from './fixtures.js';

const config = configure();
function state(closes, levels = [], extra = {}) {
  const bars = closes.map((v, i) => bar(i * 5, v, { open: i ? closes[i - 1] : v - 0.1, low: Math.min(v, i ? closes[i - 1] : v - 0.1) - 0.03, high: Math.max(v, i ? closes[i - 1] : v - 0.1) + 0.03 }));
  const s = { at: at(bars.length * 5), bars, current: bars.at(-1), first: bars[0], levels, metrics: { relativeVolume: 1.5 },
    session: { open: at(0) }, atr: 5, recentEvents: [], blockingEvents: [], allObservations: [], ...extra };
  s.getLevel = kind => s.levels.find(l => l.kind === kind);
  return s;
}
function mirror(s) {
  const mapped = { PDH: 'PDL', PDL: 'PDH', PMH: 'PML', PML: 'PMH', ONH: 'ONL', ONL: 'ONH', VAH: 'VAL', VAL: 'VAH', OR15H: 'OR15L', OR15L: 'OR15H', OR30H: 'OR30L', OR30L: 'OR30H', IBH: 'IBL', IBL: 'IBH' };
  const bars = s.bars.map(b => ({ ...b, open: 200 - b.open, high: 200 - b.low, low: 200 - b.high, close: 200 - b.close }));
  const m = { ...s, bars, first: bars[0], current: bars.at(-1), levels: s.levels.map(l => ({ ...l, kind: mapped[l.kind] ?? l.kind, id: mapped[l.id] ?? l.id, value: 200 - l.value })),
    allObservations: s.allObservations.map(o => ({ ...o, value: o.metric === 'adRatio' ? 3 - o.value : 100 - o.value })) };
  m.getLevel = kind => m.levels.find(l => l.kind === kind);
  return m;
}
const vwap = level('VWAP', 99);
const cases = [
  ['gap-and-go', () => state([100.2, 100.4, 100.6], [level('PDC', 98), level('PDH', 100), level('VAH', 99.5), level('VAL', 97), vwap])],
  ['gap-rejection-fill', () => {
    const s = state([98.8, 98.7, 99.2, 99.3], [level('PDC', 100), level('OPEN', 99)]);
    s.first.open = 99; return s;
  }],
  ['gap-reclaim-rejection', () => {
    const s = state([98.8, 99.8, 100.2, 100.3], [level('PDC', 100)]); s.first.open = 99; return s;
  }],
  ['inside-value-rotation', () => {
    const s = state([100.1, 100.2], [level('VAL', 100), level('VAH', 102)]); s.bars[0].open = 100.1; s.bars[0].low = 99.8; s.bars[1].low = 100.08; return s;
  }],
  ['open-drive', () => state([100.2, 100.4, 100.6], [level('OPEN', 100), vwap])],
  ['open-test-drive', () => {
    const s = state([100.2, 100.4, 100.6], [level('OPEN', 100), vwap]); s.first.open = 100; s.first.low = 99.8; return s;
  }],
  ['opening-range-breakout', () => state([99.8, 100, 100.2, 100.5, 100.6], [level('OR15H', 100.3, { available_at: at(15) }), vwap])],
  ['failed-opening-range-breakout', () => state([99.8, 99.7, 99.5, 99.6, 99.3, 99.2, 99.7, 99.8], [level('OR15L', 99.5, { available_at: at(15) })])],
  ['trend-pullback-continuation', () => state([99, 99.4, 99.8, 100.2, 100.6, 100.4, 100.5, 100.6], [level('AVWAP', 100.35), vwap])],
  ['liquidity-sweep-reversal', () => {
    const s = state([100.1, 100.2], [level('PDL', 100)]); s.first.open = 100.1; s.first.low = 99.8; s.bars[1].low = 100.08; return s;
  }],
  ['midday-compression-breakout', () => state([99.8, 99.85, 99.8, 99.85, 99.8, 99.85, 100.2, 100.3], [vwap])],
  ['breadth-divergence-reversal', () => {
    const s = state([100.1, 100.2, 100.1, 100.2, 100.1, 100.2], [level('PDL', 100)]);
    s.bars[4].low = 99.8; s.bars[5].low = 100.08;
    s.allObservations = [observation('adRatio', 1, 5), observation('adRatio', 1.5, 25), observation('sp500AboveVwapPct', 45, 5), observation('sp500AboveVwapPct', 55, 25)]; return s;
  }],
  ['event-repricing', () => state([99.8, 99.9, 100, 100.4, 100.5, 100.6], [], { recentEvents: [{ id: 'macro', status: 'resolved', at: at(15) }] })],
  ['power-hour-continuation', () => state([99, 99.2, 99.4, 99.6, 100.2, 100.4], [level('IBH', 100), vwap])],
  ['late-day-unwind', () => {
    const s = state([101.9, 101.5, 101, 100, 100.6, 100.7], [level('IBL', 100.5)]); s.first.open = 102; return s;
  }]
];
for (const [id, build] of cases) {
  for (const direction of ['bullish', 'bearish']) test(`${id}: ${direction} rule can qualify from price and required inputs`, () => {
    const s = direction === 'bullish' ? build() : mirror(build());
    const result = evaluateRule(catalogue(config).find(r => r.id === id), s, direction, config);
    assert.equal(result.eligible, true, JSON.stringify(result.checks.filter(c => !c.pass)));
  });
}
test('a fresh sweep cannot borrow pre-sweep closes for acceptance', () => {
  const s = state([100.1, 100.2], [level('PDL', 100)]); s.bars[1].low = 99.8;
  const result = evaluateRule(catalogue(config).find(r => r.id === 'liquidity-sweep-reversal'), s, 'bullish', config);
  assert.equal(result.eligible, false); assert.equal(result.acceptance.status, 'pending');
});
test('an ordinary level crossing is not mislabeled as a liquidity sweep', () => {
  const s = state([99.9, 100.2, 100.3], [level('PDL', 100)]);
  const result = evaluateRule(catalogue(config).find(r => r.id === 'liquidity-sweep-reversal'), s, 'bullish', config);
  assert.equal(result.eligible, false);
});
test('strike pin requires both observed compression and supplied positioning', () => {
  const s = state([100, 100.02, 100.01, 100.03, 100.02, 100.01], [level('STRIKE', 100)]);
  s.metrics.dealerGammaSign = 1;
  const rule = catalogue(config).find(r => r.id === 'late-day-strike-pin');
  assert.equal(evaluateRule(rule, s, 'neutral', config).eligible, true);
  s.bars[3].high = 102;
  assert.equal(evaluateRule(rule, s, 'neutral', config).eligible, false);
});
