import { ms, iso, minutesBetween, eastern, phase } from './time.js';

const assert = (condition, message) => { if (!condition) throw new Error(message); };
const timestamp = (v, label) => assert(typeof v === 'string' && /(?:Z|[+-]\d{2}:\d{2})$/.test(v) && Number.isFinite(ms(v)), `${label} must be an ISO timestamp with timezone`);
const finite = (v, label) => assert(typeof v === 'number' && Number.isFinite(v), `${label} must be finite`);
function provenance(item, label) {
  assert(typeof item.source === 'string' && item.source.trim(), `${label}.source required`);
  timestamp(item.available_at, `${label}.available_at`);
}

/** Validate the transport contract; nulls stay missing and no provider is assumed. */
export function validateDataset(data) {
  assert(data?.symbol === 'SPY', 'Only SPY underlying prices are supported');
  const s = data.session;
  assert(s && typeof s.id === 'string' && typeof s.isTradingDay === 'boolean', 'Explicit session id and isTradingDay required');
  timestamp(s.open, 'session.open'); timestamp(s.close, 'session.close');
  assert(ms(s.close) > ms(s.open), 'Session close must follow open');
  provenance(s, 'session');
  assert(eastern(s.open).date === s.id && eastern(s.close).date === s.id, 'Session id must equal Eastern session date');
  assert(Array.isArray(data.bars), 'bars must be an array');
  const seen = new Set();
  for (const b of data.bars) {
    timestamp(b.start, 'bar.start'); timestamp(b.end, 'bar.end'); provenance(b, 'bar');
    assert(b.timeframe === 5 && minutesBetween(b.end, b.start) === 5, 'Supply finalized five-minute bars only');
    assert(ms(b.available_at) >= ms(b.end), 'Bar cannot be available before its close');
    for (const key of ['open', 'high', 'low', 'close', 'volume']) finite(b[key], `bar.${key}`);
    assert(b.low > 0 && b.low <= Math.min(b.open, b.close) && b.high >= Math.max(b.open, b.close) && b.volume >= 0, 'Invalid OHLCV');
    assert(!seen.has(b.start), 'Duplicate bars require an explicit revision adapter'); seen.add(b.start);
  }
  for (const l of data.levels ?? []) {
    assert(typeof l.id === 'string' && l.id && typeof l.kind === 'string', 'Level id and kind required');
    finite(l.value, 'level.value'); assert(l.value > 0, 'Level value must be positive');
    provenance(l, 'level'); timestamp(l.as_of, 'level.as_of');
    assert(ms(l.as_of) <= ms(l.available_at), 'Level as_of cannot follow available_at');
    assert(typeof l.method === 'string' && l.method.trim(), 'Every level requires its methodology');
    if (l.valid_until) timestamp(l.valid_until, 'level.valid_until');
  }
  for (const o of data.observations ?? []) {
    provenance(o, 'observation'); timestamp(o.as_of, 'observation.as_of');
    assert(ms(o.as_of) <= ms(o.available_at), 'Observation as_of cannot follow available_at');
    assert(typeof o.metric === 'string' && o.metric, 'Observation metric required');
    finite(o.value, 'observation.value');
    assert(typeof o.method === 'string' && o.method.trim(), 'Observation method required');
    if (['adRatio', 'upDownVolumeRatio', 'relativeVolume', 'vix', 'vix1d', 'spyPremarketPrice'].includes(o.metric)) assert(o.value >= 0, `${o.metric} cannot be negative`);
    if (['sp500AboveVwapPct', 'overnightAbovePdcPct'].includes(o.metric)) assert(o.value >= 0 && o.value <= 100, `${o.metric} must be 0..100`);
    if (['sectorAlignment', 'macroAlignment', 'dealerGammaSign'].includes(o.metric)) assert(o.value >= -1 && o.value <= 1, `${o.metric} must be -1..1`);
  }
  for (const e of data.events ?? []) {
    provenance(e, 'event'); timestamp(e.at, 'event.at');
    assert(typeof e.id === 'string' && e.id && typeof e.type === 'string', 'Event id and type required');
    assert(['scheduled', 'uncertain', 'resolved'].includes(e.status), 'Unknown event status');
  }
  for (const c of data.eventCoverage ?? []) {
    provenance(c, 'eventCoverage'); timestamp(c.from, 'eventCoverage.from'); timestamp(c.through, 'eventCoverage.through');
    assert(ms(c.through) >= ms(c.from), 'Invalid event coverage interval');
  }
  return data;
}

function latestBy(rows, key) {
  const result = Object.create(null);
  for (const row of [...rows].sort((a, b) => ms(a.available_at) - ms(b.available_at))) result[row[key]] = row;
  return result;
}

/** Only rows actually available by the evaluation timestamp enter this view. */
export function normalize(data, at, config) {
  timestamp(at, 'checkpoint');
  assert(eastern(at).date === data.session.id, 'Checkpoint must belong to the Eastern session date');
  assert(ms(data.session.available_at) <= ms(at), 'Session schedule not yet available');
  const visible = row => ms(row.available_at) <= ms(at);
  const bars = data.bars.filter(b => visible(b) && ms(b.end) <= ms(at) && ms(b.start) >= ms(data.session.open) && ms(b.end) <= ms(data.session.close))
    .sort((a, b) => ms(a.start) - ms(b.start));
  const issues = [];
  const levelRows = Object.values(latestBy((data.levels ?? []).filter(visible), 'id'));
  const levels = levelRows.filter(l => {
    const expired = l.valid_until && ms(l.valid_until) < ms(at);
    const dynamic = ['VWAP', 'AVWAP'].includes(l.kind);
    if (expired || dynamic && minutesBetween(at, l.as_of) > config.freshness.levelsMinutes) { issues.push(`Stale level: ${l.id}`); return false; }
    return true;
  });
  const allObservations = (data.observations ?? []).filter(visible);
  const observations = latestBy(allObservations, 'metric');
  const metrics = {};
  for (const [key, o] of Object.entries(observations)) {
    const ttl = key.startsWith('dealer') ? config.freshness.positioningMinutes : config.freshness.metricsMinutes;
    if (minutesBetween(at, o.as_of) <= ttl) metrics[key] = o.value;
    else issues.push(`Stale input: ${key}`);
  }
  const getLevel = kind => levels.find(l => l.kind === kind);
  const complete = (start, end) => {
    const span = bars.filter(b => ms(b.start) >= start && ms(b.end) <= end);
    return span.length === (end - start) / 300000 && span.every((b, i) => ms(b.start) === start + i * 300000) ? span : null;
  };
  const addRange = (count, prefix) => {
    const start = ms(data.session.open), end = start + count * 60000;
    if (ms(at) < end) return;
    const span = complete(start, end);
    if (!span) { issues.push(`Unavailable ${prefix}: incomplete opening bars`); return; }
    for (const [suffix, value] of [['H', Math.max(...span.map(b => b.high))], ['L', Math.min(...span.map(b => b.low))]]) {
      levels.push({ id: `${prefix}${suffix}`, kind: `${prefix}${suffix}`, value, as_of: iso(end), available_at: iso(Math.max(...span.map(b => ms(b.available_at)))), source: 'engine:SPY-bars', method: `Extrema of complete first ${count} RTH minutes` });
    }
  };
  // OR and IB are derived only from complete RTH intervals; external duplicates are rejected.
  assert(!levels.some(l => l.kind === 'OPEN' || /^(OR15|OR30|IB)[HL]$/.test(l.kind)), 'OPEN, opening range and IB are engine-derived; do not supply duplicate definitions');
  const singletonKinds = ['PDH', 'PDL', 'PDC', 'PMH', 'PML', 'ONH', 'ONL', 'VAH', 'VAL', 'VPOC', 'VWAP', 'ATR', 'EM_HIGH', 'EM_LOW', 'GAMMA_FLIP'];
  for (const kind of singletonKinds) assert(levels.filter(l => l.kind === kind).length <= 1, `Ambiguous ${kind}: use one stable series id, retaining timestamped versions`);
  addRange(15, 'OR15'); addRange(30, 'OR30'); addRange(60, 'IB');
  const current = bars.at(-1);
  const eventRows = Object.values(latestBy((data.events ?? []).filter(visible), 'id'));
  const eventCoverage = (data.eventCoverage ?? []).some(c => visible(c) && ms(c.from) <= ms(at) && ms(c.through) >= ms(at));
  const recentEvents = eventRows.filter(e => minutesBetween(at, e.at) >= 0 && minutesBetween(at, e.at) <= config.events.repricingMinutes);
  const blockingEvents = eventRows.filter(e => {
    const delta = minutesBetween(at, e.at);
    return e.status === 'uncertain' && delta >= -config.events.beforeMinutes || delta >= -config.events.beforeMinutes && delta <= config.events.afterMinutes;
  });
  const lastEnd = current ? ms(current.end) : ms(data.session.open);
  const gaps = bars.some((b, i) => ms(b.start) !== (i ? ms(bars[i - 1].end) : ms(data.session.open)));
  const dataStale = !current || (ms(at) - lastEnd) / 60000 > config.freshness.barsMinutes || gaps;
  if (gaps) issues.push('Incomplete RTH bar history');
  const first = bars.find(b => b.start === data.session.open || ms(b.start) === ms(data.session.open));
  if (first) levels.push({ id: 'OPEN', kind: 'OPEN', value: first.open, as_of: first.start, available_at: first.available_at, source: first.source, method: 'First RTH five-minute bar open' });
  const atr = getLevel('ATR')?.value;
  const range = bars.length ? Math.max(...bars.map(b => b.high)) - Math.min(...bars.map(b => b.low)) : null;
  return { symbol: 'SPY', at, session: data.session, phase: phase(at, data.session), bars, levels, metrics, observations,
    allObservations, levelHistory: (data.levels ?? []).filter(visible), current, first, atr, range,
    rangeConsumed: atr && range !== null ? range / atr : null,
    events: eventRows, recentEvents, blockingEvents, eventCoverage, dataStale, issues,
    getLevel: kind => levels.find(l => l.kind === kind) };
}
