/** Research defaults, not empirically validated trading thresholds. */
export const defaults = {
  acceptance: { buffer: 0.01, retestTolerance: 0.05, retestMaxBars: 3, maxAgeMinutes: 30 },
  freshness: { barsMinutes: 7, metricsMinutes: 10, positioningMinutes: 60, levelsMinutes: 15 },
  evidence: {
    breadthEnabled: true, adRatioBull: 1.5, adRatioBear: 0.67,
    upDownVolumeBull: 1.5, upDownVolumeBear: 0.67,
    aboveVwapBull: 60, aboveVwapBear: 40, tickBull: 200, tickBear: -200,
    relativeVolumeMin: 0.8, relativeVolumeConfirm: 1.1,
    alignmentReturnPct: 0.1, vixChangeConfirm: 0.3,
    minTradeCategories: 2, highConfidenceCategories: 4,
    divergenceAdDelta: 0.2, divergenceAboveVwapDelta: 5
  },
  structure: {
    minGapPct: 0.1, driveEfficiency: 0.65, trendEfficiency: 0.55,
    compressionBars: 6, compressionAtrFraction: 0.12,
    vwapCrossLookback: 8, vwapCrossMax: 3, sweepLookback: 6,
    pinAtrFraction: 0.08, rangeExhaustion: 0.9
  },
  events: { beforeMinutes: 10, afterMinutes: 10, repricingMinutes: 60 },
  scenarios: {},
  requireEventCoverage: true,
  allowProvisionalWithoutEventCoverage: false,
  requireVwap: true,
  requireVolume: true
};

export function configure(patch = {}, base = defaults) {
  if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw new Error('Config must be an object');
  const result = structuredClone(base);
  for (const [key, value] of Object.entries(patch)) {
    if (!(key in base)) throw new Error(`Unknown config key: ${key}`);
    if (key === 'scenarios') { result[key] = structuredClone(value); continue; }
    if (typeof base[key] === 'object') result[key] = configure(value, base[key]);
    else {
      if (typeof value !== typeof base[key] || (typeof value === 'number' && (!Number.isFinite(value) || value < 0 && !['tickBear'].includes(key)))) {
        throw new Error(`Invalid config value: ${key}`);
      }
      result[key] = value;
    }
  }
  if (base === defaults) {
    for (const [name, n] of Object.entries({ retestMaxBars: result.acceptance.retestMaxBars, compressionBars: result.structure.compressionBars, sweepLookback: result.structure.sweepLookback, vwapCrossLookback: result.structure.vwapCrossLookback, vwapCrossMax: result.structure.vwapCrossMax, minTradeCategories: result.evidence.minTradeCategories, highConfidenceCategories: result.evidence.highConfidenceCategories })) {
      if (!Number.isInteger(n) || n < 1) throw new Error(`${name} must be a positive integer`);
    }
    if (result.evidence.highConfidenceCategories < 2 || result.evidence.highConfidenceCategories < result.evidence.minTradeCategories) throw new Error('High confidence requires multiple categories and cannot be below the trade minimum');
    if (result.evidence.relativeVolumeConfirm < result.evidence.relativeVolumeMin) throw new Error('Relative volume confirmation must be >= minimum');
    for (const [bull, bear] of [['adRatioBull', 'adRatioBear'], ['upDownVolumeBull', 'upDownVolumeBear'], ['aboveVwapBull', 'aboveVwapBear'], ['tickBull', 'tickBear']]) if (result.evidence[bull] <= result.evidence[bear]) throw new Error(`${bull} must exceed ${bear}`);
  }
  return result;
}
