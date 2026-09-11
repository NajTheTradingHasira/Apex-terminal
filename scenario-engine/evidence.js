import { sign } from './acceptance.js';
import { ms } from './time.js';

export const metricNames = ['adRatio', 'upDownVolumeRatio', 'sp500AboveVwapPct', 'nyseTickMean', 'nasdaqTickMean', 'relativeVolume', 'vix', 'vix1d', 'vixChangePct', 'vix1dChangePct', 'qqqReturnPct', 'iwmReturnPct', 'sectorAlignment', 'yieldChangeBps', 'dollarChangePct', 'macroAlignment', 'dealerGammaSign', 'spyPremarketPrice', 'overnightAbovePdcPct'];
const category = (name, items) => {
  const positives = items.filter(x => x.vote > 0), negatives = items.filter(x => x.vote < 0);
  return { category: name, vote: positives.length && negatives.length ? 0 : positives.length ? 1 : negatives.length ? -1 : 0,
    conflicted: !!(positives.length && negatives.length), items };
};
export function evidence(state, direction, config) {
  const d = sign(direction), t = config.evidence, m = state.metrics;
  const item = (metric, rawVote) => ({ metric, value: m[metric], vote: rawVote * d,
    source: state.observations[metric]?.source, as_of: state.observations[metric]?.as_of, method: state.observations[metric]?.method });
  const threshold = (metric, bull, bear) => m[metric] === undefined ? [] : [item(metric, m[metric] >= bull ? 1 : m[metric] <= bear ? -1 : 0)];
  const categories = [];
  if (t.breadthEnabled) categories.push(category('breadth', [
    ...threshold('adRatio', t.adRatioBull, t.adRatioBear),
    ...threshold('sp500AboveVwapPct', t.aboveVwapBull, t.aboveVwapBear),
    ...threshold('nyseTickMean', t.tickBull, t.tickBear),
    ...threshold('nasdaqTickMean', t.tickBull, t.tickBear)
  ]));
  const volume = threshold('upDownVolumeRatio', t.upDownVolumeBull, t.upDownVolumeBear);
  if (m.relativeVolume !== undefined) volume.push({ ...item('relativeVolume', 0), vote: m.relativeVolume >= t.relativeVolumeConfirm ? (volume[0]?.vote ?? 1) : 0 });
  categories.push(category('participation', volume));
  categories.push(category('cross-asset', [
    ...threshold('qqqReturnPct', t.alignmentReturnPct, -t.alignmentReturnPct),
    ...threshold('iwmReturnPct', t.alignmentReturnPct, -t.alignmentReturnPct),
    ...threshold('sectorAlignment', 0.5, -0.5)
  ]));
  const volatility = ['vixChangePct', 'vix1dChangePct'].flatMap(metric => m[metric] === undefined ? [] : [item(metric, m[metric] <= -t.vixChangeConfirm ? 1 : m[metric] >= t.vixChangeConfirm ? -1 : 0)]);
  categories.push(category('volatility', volatility));
  // Rates/USD have no universal bullish sign. A documented external macro mapping is mandatory.
  if (state.events.some(e => e.macroSensitive && ms(e.at) <= ms(state.at))) categories.push(category('macro', threshold('macroAlignment', 0.5, -0.5)));
  const vwap = state.getLevel('VWAP');
  categories.push(category('price-location', vwap && state.current ? [{ metric: 'SPY vs session VWAP', value: state.current.close - vwap.value, vote: Math.sign(state.current.close - vwap.value) * d, source: vwap.source, as_of: vwap.as_of, method: vwap.method }] : []));
  const confirming = categories.filter(c => c.vote > 0 && !c.conflicted);
  const conflicting = categories.filter(c => c.vote < 0 || c.conflicted);
  return { categories, confirming, conflicting,
    missing: metricNames.filter(k => m[k] === undefined),
    independentCount: confirming.length,
    highEligible: confirming.length >= t.highConfidenceCategories && !conflicting.length && confirming.some(c => c.category === 'breadth') && confirming.some(c => c.category === 'participation') };
}

export function vwapCrossings(state, config) {
  const bars = state.bars.slice(-config.structure.vwapCrossLookback);
  const sides = bars.map(b => {
    const level = state.levelHistory.filter(l => l.kind === 'VWAP' && ms(l.available_at) <= ms(b.end) && ms(l.as_of) <= ms(b.end) && ms(b.end) - ms(l.as_of) <= config.freshness.levelsMinutes * 60000)
      .sort((a, c) => ms(c.available_at) - ms(a.available_at))[0];
    return level ? Math.sign(b.close - level.value) : null;
  });
  let crossings = 0, prior = null;
  for (const side of sides) {
    if (side === null) { prior = null; continue; }
    if (side === 0) continue;
    if (prior !== null && prior !== side) crossings++;
    prior = side;
  }
  return { crossings, observedBars: sides.filter(s => s !== null).length };
}
