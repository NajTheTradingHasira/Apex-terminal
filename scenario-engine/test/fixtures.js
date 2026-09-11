const base = Date.parse('2026-09-04T13:30:00Z');
export const at = minute => new Date(base + minute * 60000).toISOString();
export const level = (kind, value, extra = {}) => ({ id: kind, kind, value, as_of: at(-60), available_at: at(-60), source: 'synthetic:test', method: 'Synthetic fixture level; not a market measurement', ...extra });
export const observation = (metric, value, minute = 0) => ({ metric, value, as_of: at(minute), available_at: at(minute), source: 'synthetic:test', method: 'Synthetic fixture observation' });
export function bar(minute, close, extra = {}) {
  const open = extra.open ?? close - 0.1;
  return { start: at(minute), end: at(minute + 5), available_at: at(minute + 5), timeframe: 5,
    open, high: Math.max(open, close) + 0.1, low: Math.min(open, close) - 0.1,
    close, volume: 100000, source: 'synthetic:test', ...extra };
}
export function dataset(closes = [100, 100.2, 100.4, 100.8, 101]) {
  const bars = closes.map((close, i) => bar(i * 5, close, { open: i ? closes[i - 1] : 99.9 }));
  return { symbol: 'SPY', label: 'SYNTHETIC TEST DATA — NOT LIVE',
    session: { id: '2026-09-04', open: at(0), close: at(390), isTradingDay: true, source: 'synthetic:test-calendar', available_at: at(-120) },
    bars,
    levels: [level('PDC', 99.5), level('PDH', 101.5), level('PDL', 98), level('VAH', 101.3), level('VAL', 98.5), level('VPOC', 99.4), level('ATR', 5), level('WEEKLY', 105), level('EM_HIGH', 104), level('EM_LOW', 96), level('VWAP', 99.8, { as_of: at(0), available_at: at(0), valid_until: at(390) })],
    observations: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60].flatMap(t => [observation('adRatio', 2, t), observation('sp500AboveVwapPct', 70, t), observation('relativeVolume', 1.3, t), observation('upDownVolumeRatio', 2, t), observation('qqqReturnPct', 0.5, t), observation('iwmReturnPct', 0.4, t), observation('vixChangePct', -1, t)]),
    events: [], eventCoverage: [{ from: at(-90), through: at(390), available_at: at(-90), source: 'synthetic:test-calendar' }] };
}
export function refreshed(data) {
  data.levels = data.levels.filter(l => l.kind !== 'VWAP').concat(data.bars.map(b => level('VWAP', 99.8, { as_of: b.end, available_at: b.end })));
  return data;
}
