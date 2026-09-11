import { acceptance, beyond, sign, sweep } from './acceptance.js';
import { ms, iso } from './time.js';

const majorKinds = ['PDH', 'PDL', 'ONH', 'ONL', 'PMH', 'PML', 'WEEKLY', 'VAH', 'VAL', 'AVWAP'];
const efficiency = bars => {
  if (bars.length < 2) return 0;
  const travel = bars.reduce((n, b, i) => n + Math.abs(b.close - (i ? bars[i - 1].close : b.open)), 0);
  return travel ? Math.abs(bars.at(-1).close - bars[0].open) / travel : 0;
};
function derived(id, value, bars, method) {
  return { id, kind: id, value, source: 'engine:SPY-bars', method, as_of: bars.at(-1).end, available_at: iso(Math.max(...bars.map(b => ms(b.available_at)))) };
}
const bound = (bars, direction) => direction === 'bullish' ? Math.max(...bars.map(b => b.high)) : Math.min(...bars.map(b => b.low));

function triggerAcceptance(ctx) {
  const { state, level, direction, config, rule } = ctx;
  const hit = rule?.preconditions.includes('sweep') ? sweep(state, level, direction, config) : null;
  const afterSweep = hit ? { ...state, bars: state.bars.filter(b => ms(b.start) >= ms(hit.at) - 300000) } : state;
  return acceptance(afterSweep, level, direction, config);
}

export function candidateLevels(selector, state, direction, config, previous) {
  const up = direction === 'bullish', level = kind => state.getLevel(kind);
  const pick = kinds => kinds.map(level).filter(Boolean);
  const preSignal = state.bars.slice(0, -2);
  switch (selector) {
    case null: return [];
    case 'gap-edge': return pick(up ? ['PDH', 'PMH', 'ONH'] : ['PDL', 'PML', 'ONL']);
    case 'value-edge': return pick(up ? ['VAL'] : ['VAH']);
    case 'or-edge': return pick(up ? ['OR30H', 'OR15H'] : ['OR30L', 'OR15L']);
    case 'failed-or-edge': return previous?.id === 'opening-range-breakout' && previous.direction !== direction ? [previous.level] : pick(up ? ['OR30L', 'OR15L'] : ['OR30H', 'OR15H']);
    case 'swept-major': return state.levels.filter(l => majorKinds.includes(l.kind) && sweep(state, l, direction, config));
    case 'pullback-level': return state.levels.filter(l => ['OR15H', 'OR15L', 'OR30H', 'OR30L', 'IBH', 'IBL', 'AVWAP'].includes(l.kind));
    case 'compression-edge': {
      const box = preSignal.slice(-config.structure.compressionBars);
      return box.length === config.structure.compressionBars ? [derived('COMPRESSION', bound(box, direction), box, 'Extrema of N five-minute bars before the latest two signal bars')] : [];
    }
    case 'event-edge': {
      const event = state.recentEvents.filter(e => e.status === 'resolved').at(-1);
      const bars = event ? state.bars.filter(b => ms(b.end) <= ms(event.at)).slice(-3) : [];
      return bars.length === 3 ? [derived('PRE_EVENT', bound(bars, direction), bars, 'Extrema of last three completed pre-event bars')] : [];
    }
    case 'trend-edge': return pick(up ? ['IBH', 'OR30H'] : ['IBL', 'OR30L']);
    case 'late-reference': return pick(up ? ['IBL', 'OR30L'] : ['IBH', 'OR30H']);
    case 'nearest-strike': return state.levels.filter(l => l.kind === 'STRIKE').sort((a, b) => Math.abs(a.value - state.current.close) - Math.abs(b.value - state.current.close)).slice(0, 1);
    default: return pick([selector]);
  }
}

export function predicate(name, ctx) {
  const { state: s, direction: dir, level, config: c } = ctx;
  const d = sign(dir), price = s.current?.close, open = s.first?.open, pdc = s.getLevel('PDC')?.value;
  const gap = open && pdc ? (open - pdc) / pdc * 100 : null;
  const recent = s.bars.slice(-6), before = s.bars.slice(0, -2).slice(-6);
  const vwap = s.getLevel('VWAP')?.value;
  const vah = s.getLevel('VAH')?.value, val = s.getLevel('VAL')?.value;
  const missing = reason => ({ pass: false, unavailable: true, reason });
  const result = (pass, reason) => ({ pass: !!pass, unavailable: false, reason });
  switch (name) {
    case 'acceptance': return result(triggerAcceptance(ctx).status === 'accepted', triggerAcceptance(ctx).reason);
    case 'gap-aligned': case 'gap-opposed': return gap === null ? missing('Previous close or first RTH open unavailable') : result(Math.abs(gap) >= c.structure.minGapPct && (name === 'gap-aligned' ? gap * d > 0 : gap * d < 0), `SPY opening gap ${gap.toFixed(3)}%`);
    case 'outside-value': return vah === undefined || val === undefined ? missing('External VAH/VAL unavailable') : result(d === 1 ? open > vah : open < val, 'Opening price must be outside prior value in the gap direction');
    case 'inside-value': return vah === undefined || val === undefined ? missing('External VAH/VAL unavailable') : result(price >= val && price <= vah, 'SPY is within supplied value area');
    case 'vwap-aligned': return vwap === undefined ? missing('Session VWAP unavailable') : result(d * (price - vwap) > c.acceptance.buffer, 'SPY price is on the directional side of session VWAP');
    case 'prior-gap-extension': return result(s.bars.slice(0, -2).some(b => d === 1 ? b.low < open - c.acceptance.buffer : b.high > open + c.acceptance.buffer), 'Gap extended away from previous close before rejection');
    case 'drive': return result(s.bars.length >= 2 && efficiency(s.bars) >= c.structure.driveEfficiency && d * (price - open) > c.acceptance.buffer, 'Opening path efficiency and displacement meet drive thresholds');
    case 'opening-test': return result(s.bars.slice(0, 3).some(b => d === 1 ? b.low < open - c.acceptance.buffer && b.close > open : b.high > open + c.acceptance.buffer && b.close < open), 'First 15 minutes tested the opposite side of the open and recovered');
    case 'sweep': return result(!!sweep(s, level, dir, c), 'A recent wick swept the level and closed back through it');
    case 'prior-trend': return result(before.length >= 4 && efficiency(before) >= c.structure.trendEfficiency && d * (before.at(-1).close - before[0].open) > c.acceptance.buffer, 'Pre-signal bars show efficient directional trend');
    case 'pullback': {
      const test = recent.slice(0, -1).some(b => d === 1 ? b.low <= level.value + c.acceptance.retestTolerance && b.close >= level.value : b.high >= level.value - c.acceptance.retestTolerance && b.close <= level.value);
      const retreat = before.length >= 2 && before.some((b, i) => i && d * (b.close - before[i - 1].close) < 0);
      return result(test && retreat, 'A countertrend retracement tested the selected reference and held');
    }
    case 'failed-or': {
      const prior = s.bars.filter(b => ms(b.start) >= ms(level.available_at)).slice(0, -2);
      const priorAccepted = prior.some((b, i) => i && ms(b.start) === ms(prior[i - 1].end) && beyond(b.close, level.value, dir === 'bullish' ? 'bearish' : 'bullish', c.acceptance.buffer) && beyond(prior[i - 1].close, level.value, dir === 'bullish' ? 'bearish' : 'bullish', c.acceptance.buffer));
      const priorWick = prior.some(b => d === 1 ? b.low < level.value - c.acceptance.buffer : b.high > level.value + c.acceptance.buffer);
      return result(priorAccepted || priorWick, 'Earlier OR break or sweep followed by opposite acceptance back inside');
    }
    case 'compression': {
      const box = s.bars.slice(0, -2).slice(-c.structure.compressionBars);
      if (!s.atr) return missing('Daily ATR unavailable');
      return result(box.length === c.structure.compressionBars && Math.max(...box.map(b => b.high)) - Math.min(...box.map(b => b.low)) <= s.atr * c.structure.compressionAtrFraction, 'Pre-signal box width is within configured ATR fraction');
    }
    case 'strong-volume': return s.metrics.relativeVolume === undefined ? missing('Relative volume unavailable') : result(s.metrics.relativeVolume >= c.evidence.relativeVolumeConfirm, 'Relative volume meets participation threshold');
    case 'breadth-divergence': {
      if (!c.evidence.breadthEnabled) return missing('Breadth rules disabled');
      const hit = sweep(s, level, dir, c);
      if (!hit) return result(false, 'No qualifying price extreme');
      const metricKeys = ['adRatio', 'sp500AboveVwapPct'];
      const deltas = metricKeys.map((metric, i) => {
        const rows = s.allObservations.filter(o => o.metric === metric && ms(o.as_of) <= ms(hit.at) && ms(o.available_at) <= ms(hit.at)).sort((a, b) => ms(a.as_of) - ms(b.as_of));
        const latest = rows.at(-1), earlier = rows.filter(o => latest && ms(latest.as_of) - ms(o.as_of) >= 900000 && ms(latest.as_of) - ms(o.as_of) <= 3600000).at(-1);
        return latest && earlier && ms(hit.at) - ms(latest.as_of) <= c.freshness.metricsMinutes * 60000 ? d * (latest.value - earlier.value) >= (i ? c.evidence.divergenceAboveVwapDelta : c.evidence.divergenceAdDelta) : null;
      });
      return deltas.some(x => x === null) ? missing('Point-in-time breadth comparison unavailable') : result(deltas.every(Boolean), 'AD ratio and S&P above-VWAP percentage diverged into the swept price extreme');
    }
    case 'resolved-event': return result(s.recentEvents.some(e => e.status === 'resolved') && !s.blockingEvents.length, 'Recent event resolved and blackout elapsed');
    case 'post-event-acceptance': {
      const event = s.recentEvents.filter(e => e.status === 'resolved').at(-1);
      return result(event && s.bars.slice(-2).length === 2 && s.bars.slice(-2).every(b => ms(b.start) >= ms(event.at)), 'Acceptance bars must start after the event');
    }
    case 'late-unwind': return result(before.length >= 4 && d * (before.at(-1).close - open) < -c.acceptance.buffer && d * (price - before.at(-1).close) > c.acceptance.buffer, 'Late price reverses the earlier session displacement through structure');
    case 'pin': {
      if (!s.atr || s.metrics.dealerGammaSign === undefined) return missing('ATR or externally modeled dealer gamma sign unavailable');
      const width = recent.length ? Math.max(...recent.map(b => b.high)) - Math.min(...recent.map(b => b.low)) : Infinity;
      return result(recent.length === 6 && width <= s.atr * c.structure.pinAtrFraction && Math.abs(price - level.value) <= s.atr * c.structure.pinAtrFraction / 2 && s.metrics.dealerGammaSign > 0, 'Observed 30-minute compression near supplied strike with positive modeled gamma');
    }
    default: throw new Error(`Unknown rule predicate: ${name}`);
  }
}

export function evaluateRule(rule, state, direction, config, previous) {
  const levels = candidateLevels(rule.level, state, direction, config, previous);
  if (!levels.length) return { rule, direction, eligible: false, checks: [{ name: 'level', pass: false, unavailable: true, reason: `Required reference unavailable: ${rule.level}` }] };
  const candidates = levels.map(level => {
    const ctx = { state, direction, config, level, previous, rule };
    const checks = [...rule.preconditions, ...rule.trigger].map(name => ({ name, ...predicate(name, ctx) }));
    return { rule, direction, level, checks, acceptance: direction === 'neutral' ? { status: 'not-applicable', level } : triggerAcceptance(ctx), eligible: checks.every(x => x.pass) };
  });
  return candidates.find(x => x.eligible) ?? candidates.sort((a, b) => b.checks.filter(x => x.pass).length - a.checks.filter(x => x.pass).length)[0];
}
