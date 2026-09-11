import { ms, minutesBetween, iso } from './time.js';

export const sign = direction => direction === 'bullish' ? 1 : -1;
export const beyond = (price, level, direction, buffer = 0) => sign(direction) * (price - level) > buffer;
const contiguous = bars => bars.every((b, i) => !i || ms(b.start) === ms(bars[i - 1].end));

/** A level cannot qualify using bars that began before it was known. */
export function acceptance(state, level, direction, config) {
  if (!level) return { status: 'unavailable', reason: 'Required level unavailable', level: null };
  const cfg = config.acceptance;
  const eligible = state.bars.filter(b => ms(b.start) >= ms(level.available_at) && minutesBetween(state.at, b.end) <= cfg.maxAgeMinutes);
  const last = eligible.at(-1);
  const result = { status: 'unconfirmed', level: { id: level.id, value: level.value, source: level.source, method: level.method, as_of: level.as_of, available_at: level.available_at }, direction, reason: 'Waiting for closed-bar acceptance', confirmed_at: null };
  if (!last || minutesBetween(state.at, last.end) > config.freshness.barsMinutes) return result;
  const pass = b => beyond(b.close, level.value, direction, cfg.buffer);
  const pair = eligible.slice(-2);
  if (pair.length === 2 && contiguous(pair) && pair.every(pass)) {
    return { ...result, status: 'accepted', method: 'two-5m-closes', confirmed_at: pair[1].end, reason: `Two consecutive 5m closes ${direction === 'bullish' ? 'above' : 'below'} ${level.id} with $${cfg.buffer} buffer` };
  }
  // Construct true clock-aligned 15m bars from complete 5m triples, relative to RTH open.
  for (let i = eligible.length - 4; i >= 0; i--) {
    const triple = eligible.slice(i, i + 3);
    if (!contiguous(triple) || (ms(triple[0].start) - ms(state.session.open)) % 900000 !== 0 || !pass(triple[2])) continue;
    const after = eligible.slice(i + 3);
    if (!after.length || after.length > cfg.retestMaxBars || !contiguous([triple[2], ...after])) continue;
    const retest = after.find(b => {
      const touch = direction === 'bullish' ? b.low <= level.value + cfg.retestTolerance : b.high >= level.value - cfg.retestTolerance;
      return touch && pass(b);
    });
    // A close back through the level voids the setup; a wick through it may be a retest.
    if (retest && after.every(b => sign(direction) * (b.close - level.value) >= -cfg.buffer) && pass(last)) {
      return { ...result, status: 'accepted', method: '15m-close-and-retest', confirmed_at: retest.end, reason: `Closed 15m beyond ${level.id}; subsequent 5m retest held` };
    }
  }
  const wick = direction === 'bullish' ? last.open <= level.value && last.high > level.value + cfg.buffer && !pass(last) : last.open >= level.value && last.low < level.value - cfg.buffer && !pass(last);
  return { ...result, status: wick ? 'sweep' : pass(last) ? 'pending' : 'unconfirmed', reason: wick ? 'Wick beyond level without close acceptance' : pass(last) ? 'One 5m close beyond level; awaiting confirmation' : result.reason };
}

export function sweep(state, level, direction, config) {
  if (!level) return null;
  const side = sign(direction);
  const bars = state.bars.filter(b => ms(b.start) >= ms(level.available_at)).slice(-config.structure.sweepLookback);
  // Bullish sweep takes downside liquidity and closes back above; bearish is symmetric.
  const hit = [...bars].reverse().find(b => side === 1
    ? b.open > level.value + config.acceptance.buffer && b.low < level.value - config.acceptance.buffer && b.close > level.value + config.acceptance.buffer
    : b.open < level.value - config.acceptance.buffer && b.high > level.value + config.acceptance.buffer && b.close < level.value - config.acceptance.buffer);
  if (!hit) return null;
  const following = bars.filter(b => ms(b.start) >= ms(hit.end));
  if (following.some(b => side === 1 ? b.close < hit.low : b.close > hit.high)) return null;
  return { at: hit.end, extreme: side === 1 ? hit.low : hit.high, level: level.id };
}

