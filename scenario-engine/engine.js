import { configure } from './config.js';
import { normalize, validateDataset } from './model.js';
import { catalogue } from './scenarios.js';
import { evaluateRule } from './rules.js';
import { acceptance, sign, sweep } from './acceptance.js';
import { evidence, vwapCrossings } from './evidence.js';
import { ms, iso, minutesBetween } from './time.js';

const opposite = direction => direction === 'bullish' ? 'bearish' : 'bullish';
const targetKinds = new Set(['PDH', 'PDL', 'PDC', 'PMH', 'PML', 'ONH', 'ONL', 'VAH', 'VAL', 'VPOC', 'VWAP', 'AVWAP', 'OR15H', 'OR15L', 'OR30H', 'OR30L', 'IBH', 'IBL', 'WEEKLY', 'EM_HIGH', 'EM_LOW', 'GAMMA_FLIP', 'STRIKE']);

function targets(state, candidate, config) {
  if (!candidate?.level || candidate.direction === 'neutral') return [];
  const d = sign(candidate.direction), seen = new Set();
  let levels = state.levels.filter(l => targetKinds.has(l.kind) && d * (l.value - state.current.close) > config.acceptance.buffer);
  if (candidate.rule.id === 'gap-rejection-fill') {
    const pdc = state.getLevel('PDC');
    if (pdc) levels = levels.filter(l => d * (l.value - pdc.value) <= 0);
  }
  if (candidate.rule.id === 'inside-value-rotation') {
    const edge = state.getLevel(d === 1 ? 'VAH' : 'VAL');
    if (edge) levels = levels.filter(l => d * (l.value - edge.value) <= 0);
  }
  const em = state.getLevel(d === 1 ? 'EM_HIGH' : 'EM_LOW');
  return levels.sort((a, b) => d * (a.value - b.value)).filter(l => {
    if (seen.has(l.value)) return false; seen.add(l.value); return true;
  }).slice(0, 4).map((l, i) => ({ rank: i + 1, level: l.id, price: l.value, source: l.source, as_of: l.as_of, method: l.method,
    role: ['GAMMA_FLIP', 'STRIKE'].includes(l.kind) ? 'positioning-reference' : 'price-objective',
    stretch: !!(em && d * (l.value - em.value) > 0) || state.rangeConsumed >= config.structure.rangeExhaustion }));
}

/** Stateful per-session reducer. Calling twice at the same timestamp is idempotent. */
export class ScenarioEngine {
  constructor(patch = {}) {
    this.config = configure(patch);
    this.rules = catalogue(this.config);
    this.active = null;
    this.history = [];
    this.lastAt = null;
    this.lastCard = null;
    this.sessionId = null;
  }

  evaluate(dataset, at) {
    validateDataset(dataset);
    if (this.sessionId && this.sessionId !== dataset.session.id) {
      this.active = null; this.lastAt = null; this.lastCard = null;
    }
    this.sessionId = dataset.session.id;
    if (this.lastAt && ms(at) < ms(this.lastAt)) throw new Error('Checkpoints must be chronological; use a fresh engine for replay');
    if (this.lastAt && ms(at) === ms(this.lastAt)) return structuredClone(this.lastCard);
    const s = normalize(dataset, at, this.config), cfg = this.config;
    const previous = this.active;
    const diagnostics = [];
    for (const rule of this.rules.filter(r => r.level && r.enabled !== false && r.windows.includes(s.phase))) {
      for (const direction of rule.direction === 'both' ? ['bullish', 'bearish'] : ['neutral']) diagnostics.push(evaluateRule(rule, s, direction, cfg, previous));
    }
    const ranked = diagnostics.filter(x => x.eligible).sort((a, b) => b.rule.priority - a.rule.priority || a.rule.id.localeCompare(b.rule.id) || a.direction.localeCompare(b.direction));
    const bull = evidence(s, 'bullish', cfg), bear = evidence(s, 'bearish', cfg);
    const cross = vwapCrossings(s, cfg);
    const vetoes = [];
    if (s.phase === 'closed') vetoes.push('Outside the supplied trading session');
    if (s.phase === 'premarket') vetoes.push('Premarket planning checkpoint; RTH trade triggers inactive');
    if (s.dataStale) vetoes.push('SPY bars stale, missing, or incomplete');
    if (cfg.requireEventCoverage && !s.eventCoverage) vetoes.push('Event calendar coverage unavailable');
    if (s.blockingEvents.length) vetoes.push(...s.blockingEvents.map(e => `Event uncertainty / blackout: ${e.type} (${e.id})`));
    if (cfg.requireVwap && !s.getLevel('VWAP')) vetoes.push('Session VWAP unavailable');
    if (cfg.requireVolume && s.metrics.relativeVolume === undefined) vetoes.push('Relative volume unavailable');
    if (s.metrics.relativeVolume < cfg.evidence.relativeVolumeMin) vetoes.push('Weak SPY relative volume');
    if (cross.crossings >= cfg.structure.vwapCrossMax) vetoes.push(`Repeated VWAP crossings: ${cross.crossings}`);
    const internalCategories = new Set(['breadth', 'participation', 'cross-asset', 'macro']);
    const internal = bull.categories.filter(c => internalCategories.has(c.category) && (c.category !== 'participation' || c.items.some(i => i.metric === 'upDownVolumeRatio')));
    const mixedInternals = internal.some(c => c.vote > 0) && internal.some(c => c.vote < 0);
    const conflictedInternals = internal.filter(c => c.conflicted || mixedInternals && c.vote !== 0);
    if (conflictedInternals.length) vetoes.push('Conflicting market internals');
    // Missing coverage may permit analysis, never trade permission. All other vetoes
    // (including known event blackouts) still suppress directional selection.
    const missingCoverage = cfg.requireEventCoverage && !s.eventCoverage;
    const selectionVetoes = cfg.allowProvisionalWithoutEventCoverage
      ? vetoes.filter(v => v !== 'Event calendar coverage unavailable') : vetoes;

    let selected = ranked[0] ?? null;
    let reason = selected ? 'Highest-priority fully qualified rule' : 'No scenario has all required preconditions and acceptance';
    let transitionType = 'selection';
    let unqualifiedSuccessor = null;
    let invalidated = false;
    if (previous && previous.direction !== 'neutral') {
      const reverse = acceptance(s, previous.level, opposite(previous.direction), cfg);
      const stopHit = previous.sweepExtreme != null && s.current && sign(previous.direction) * (s.current.close - previous.sweepExtreme) < -cfg.acceptance.buffer;
      const expired = ms(at) >= ms(previous.expiresAt);
      const rule = this.rules.find(r => r.id === previous.id);
      const phaseExpired = rule.time_expiry.phaseEnd && !rule.windows.includes(s.phase);
      if (reverse.status === 'accepted' || stopHit || expired || phaseExpired) {
        invalidated = true;
        transitionType = expired || phaseExpired ? 'expiry' : 'invalidation';
        const next = rule.next_state_if_invalidated;
        selected = ranked.find(x => x.rule.id === next) ?? null;
        unqualifiedSuccessor = selected ? null : next;
        reason = `${previous.id} ${transitionType}: ${stopHit ? 'SPY closed beyond sweep extreme' : expired || phaseExpired ? 'time limit reached' : reverse.reason}`;
      } else if (!selectionVetoes.length) {
        // Keep a still-held reference stable. Do not silently move stops as dynamic levels update.
        const held = acceptance(s, previous.level, previous.direction, cfg);
        const freshReversal = selected && selected.rule.regime === 'Reversal' && selected.rule.priority > previous.priority;
        if (selected?.direction !== previous.direction && !freshReversal) selected = null;
        if (!freshReversal && (!selected || selected.rule.priority <= previous.priority)) {
          selected = { rule, direction: previous.direction, level: previous.level, eligible: true, acceptance: held, checks: [{ name: 'active-reference-held', pass: held.status === 'accepted', reason: held.status === 'accepted' ? 'Existing scenario reference retains SPY acceptance' : 'Renewed acceptance pending; mechanical invalidation not yet confirmed' }] };
          reason = held.status === 'accepted' ? 'Existing scenario retains acceptance at its original reference' : 'Active scenario awaiting renewed acceptance; mechanical invalidation not yet confirmed';
        }
      }
    }
    const eventRisk = s.blockingEvents.length || cfg.requireEventCoverage && !s.eventCoverage;
    if (selectionVetoes.length) {
      selected = null; reason = selectionVetoes.join('; '); transitionType = 'veto';
    }
    const fallbackId = eventRisk ? 'event-risk' : 'no-trade-chop';
    const rule = selected?.rule ?? this.rules.find(r => r.id === fallbackId);
    if (['invalidation','expiry'].includes(transitionType)) reason += `; transition to ${rule.id}${unqualifiedSuccessor ? ` because ${unqualifiedSuccessor} is not qualified` : ''}`;
    const direction = selected?.direction ?? 'neutral';
    const provisional = !!selected && direction !== 'neutral' && missingCoverage && cfg.allowProvisionalWithoutEventCoverage;
    if (provisional) reason = `Provisional analysis only; event coverage unverified. ${reason}`;
    const ev = direction === 'bearish' ? bear : bull;
    const ladder = targets(s, selected, cfg);
    const modifiers = [];
    const gamma = s.getLevel('GAMMA_FLIP');
    if (gamma && s.current) modifiers.push(`SPY ${s.current.close >= gamma.value ? 'above' : 'below'} supplied gamma flip ${gamma.value}; provider model is context only`);
    if (s.metrics.dealerGammaSign !== undefined) modifiers.push(`Supplied dealer gamma sign ${s.metrics.dealerGammaSign}; ${s.metrics.dealerGammaSign > 0 ? 'compression/rotation context' : 'expansion context'}`);
    if (s.rangeConsumed >= cfg.structure.rangeExhaustion) modifiers.push(`SPY has consumed ${(s.rangeConsumed * 100).toFixed(0)}% of supplied daily ATR; confidence capped at B`);
    if (ladder.some(t => t.stretch)) modifiers.push('Targets beyond expected move or after range exhaustion are marked stretch');
    let grade = !selected ? 'D' : ev.highEligible ? 'A' : ev.independentCount >= cfg.evidence.minTradeCategories && !ev.conflicting.length ? 'B' : 'C';
    // Positioning can reduce conviction, never manufacture acceptance or a higher evidence count.
    const gammaConflict = s.metrics.dealerGammaSign > 0 && rule.regime === 'Continuation' || s.metrics.dealerGammaSign < 0 && rule.regime === 'Balance / mean reversion';
    if (gammaConflict) { modifiers.push('Modeled gamma context conflicts with scenario; confidence reduced one grade'); grade = ({ A: 'B', B: 'C', C: 'C', D: 'D' })[grade]; }
    if (grade === 'A' && (s.rangeConsumed >= cfg.structure.rangeExhaustion || ladder[0]?.stretch)) grade = 'B';
    if (selected && direction !== 'neutral' && selected.acceptance.status !== 'accepted') grade = 'C';
    if (provisional) { grade = 'C'; modifiers.push('Provisional scenario: event coverage unverified; no trade permission'); }
    const decisionBlocks = [...vetoes];
    if (!selected) decisionBlocks.push('No qualified directional scenario');
    if (selected && direction === 'neutral') decisionBlocks.push('Neutral pinning is an observation, not a directional trade');
    if (selected && direction !== 'neutral' && selected.acceptance.status !== 'accepted') decisionBlocks.push('Current price acceptance is not confirmed');
    if (selected && ev.independentCount < cfg.evidence.minTradeCategories) decisionBlocks.push('Insufficient independent confirmation categories');
    if (selected && ev.conflicting.length) decisionBlocks.push('Evidence conflicts with the selected direction');
    if (grade === 'C') decisionBlocks.push('Confidence below trade-permission threshold');
    if (selected && !ladder.length) decisionBlocks.push('No supplied SPY target ahead of current price');
    const tradePermitted = !!selected && direction !== 'neutral' && !decisionBlocks.length;
    const sameActive = selected && previous && previous.id === rule.id && previous.direction === direction && previous.level.id === selected.level.id && previous.level.value === selected.level.value && !invalidated;
    const expiresAt = sameActive ? previous.expiresAt : iso(Math.min(ms(at) + rule.time_expiry.minutes * 60000, ms(s.session.close)));
    const hit = selected && direction !== 'neutral' ? sweep(s, selected.level, direction, cfg) : null;
    const sweepExtreme = sameActive ? previous.sweepExtreme : hit?.extreme;
    const invalidation = selected && direction !== 'neutral' ? [
      { kind: 'opposite-acceptance', price: selected.level.value, level: selected.level.id, direction: opposite(direction), description: `SPY ${opposite(direction)} acceptance at ${selected.level.value}, using the same mechanical close/retest rule` },
      ...(sweepExtreme == null ? [] : [{ kind: 'sweep-extreme-close', price: sweepExtreme, description: `One finalized 5m SPY close beyond sweep extreme plus $${cfg.acceptance.buffer} buffer` }])
    ] : [{ kind: 'reevaluate', description: rule.invalidation.join('; ') }];
    const nextState = rule.next_state_if_invalidated;
    const transition = { at, from: previous?.id ?? 'initial', to: rule.id, fromDirection: previous?.direction ?? 'neutral', toDirection: direction, type: transitionType, reason };
    const card = {
      timestamp: at, symbol: 'SPY', session: s.session.id, checkpoint: s.phase,
      provisional, analysis_status: provisional ? 'Provisional — event coverage unverified' : 'Standard evaluation',
      current_regime: rule.regime, active_scenario: rule.scenario, scenario_id: rule.id,
      directional_bias: direction, trigger_status: selected ? direction === 'neutral' ? 'observed' : selected.acceptance.status === 'accepted' ? 'confirmed' : 'waiting' : 'waiting',
      confirming_evidence: selected ? ev.confirming : [], conflicting_evidence: selected ? ev.conflicting : conflictedInternals,
      acceptance_status: selected?.acceptance ?? { status: 'not-applicable', reason: 'No active directional scenario' },
      expected_path: rule.expected_path, target_ladder: ladder, invalidation,
      next_state_if_invalidated: { scenario_id: nextState, regime: this.rules.find(r => r.id === nextState).regime, fallback: 'no-trade-chop', condition: 'Next scenario must independently qualify; otherwise enter no-trade-chop' },
      confidence_grade: grade, confidence_note: 'Unvalidated rule-based grade; not a probability',
      trade_permitted: tradePermitted, decision: tradePermitted ? 'Trade permitted' : 'No trade', decision_reasons: [...new Set(decisionBlocks)],
      time_expiry: expiresAt, modifiers, range_consumed_atr: s.rangeConsumed,
      unavailable_inputs: [...new Set([...s.issues, ...ev.missing.map(k => `Missing/stale metric: ${k}`), ...['PDH', 'PDL', 'PDC', 'PMH', 'PML', 'ONH', 'ONL', 'VAH', 'VAL', 'VPOC', 'VWAP', 'AVWAP', 'WEEKLY', 'ATR', 'EM_HIGH', 'EM_LOW', 'GAMMA_FLIP', 'STRIKE'].filter(k => !s.getLevel(k)).map(k => `Missing level: ${k}`)])],
      context: { gap_pct: s.first && s.getLevel('PDC') ? (s.first.open / s.getLevel('PDC').value - 1) * 100 : null,
        premarket: { spy_price: s.metrics.spyPremarketPrice ?? null,
          indicative_gap_pct: s.metrics.spyPremarketPrice && s.getLevel('PDC') ? (s.metrics.spyPremarketPrice / s.getLevel('PDC').value - 1) * 100 : null,
          overnight_above_previous_close_pct: s.metrics.overnightAbovePdcPct ?? null,
          posture: 'Planning context only; no RTH trade permission before the open' },
        vwap_crossings: cross, levels: s.levels, metrics: s.metrics, events: s.events },
      explanation: { at, transition, selected_checks: selected?.checks ?? [], candidates: diagnostics.map(x => ({
        id: x.rule.id, name:x.rule.scenario, regime:x.rule.regime, direction: x.direction, eligible: x.eligible,
        reference: x.level?.id ?? null, reference_price:x.level?.value??null, checks: x.checks,
        acceptance:x.acceptance??{status:'unavailable'}, expected_path:x.rule.expected_path,
        // Planning objectives must be beyond both current price and the proposed trigger.
        target_ladder:s.current&&!s.dataStale&&x.level?targets({...s,current:{...s.current,close:x.direction==='bullish'?Math.max(s.current.close,x.level.value):Math.min(s.current.close,x.level.value)}},x,cfg):[],
        invalidation:x.level&&x.direction!=='neutral'?`Opposite SPY acceptance at $${x.level.value.toFixed(2)}; reevaluate all rule conditions before activation.`:'Reference unavailable; no price invalidation issued.',
        next_state:this.rules.find(r=>r.id===x.rule.next_state_if_invalidated).scenario,
      })) }
    };
    this.active = selected ? { id: rule.id, direction, level: selected.level, priority: rule.priority, expiresAt, sweepExtreme } : { id: rule.id, direction: 'neutral', priority: rule.priority };
    this.lastAt = at; this.lastCard = structuredClone(card); this.history.push(transition);
    return card;
  }
}
