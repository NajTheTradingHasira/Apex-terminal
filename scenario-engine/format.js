export function formatCard(c) {
  const evidenceText = items => items.length ? items.map(x => x.category).join(', ') : 'none';
  return [
    `${c.timestamp} | SPY | ${c.checkpoint}`,
    `${c.current_regime} / ${c.active_scenario} / ${c.directional_bias.toUpperCase()}`,
    `${c.decision} | Grade ${c.confidence_grade} (unvalidated) | Trigger: ${c.trigger_status}`,
    `Confirming: ${evidenceText(c.confirming_evidence)} | Conflicting: ${evidenceText(c.conflicting_evidence)}`,
    `Acceptance: ${c.acceptance_status.status} — ${c.acceptance_status.reason ?? 'Neutral observation'}`,
    `Path: ${c.expected_path.join(' → ')}`,
    `Targets (SPY): ${c.target_ladder.map(t => `${t.level} $${t.price.toFixed(2)}${t.stretch ? ' (stretch)' : ''}`).join(' → ') || 'unavailable'}`,
    `Invalidation: ${c.invalidation.map(i => i.description).join('; ')}`,
    `Next: ${c.next_state_if_invalidated.scenario_id}, else no-trade-chop | Expires: ${c.time_expiry}`,
    ...(c.decision_reasons.length ? [`Reasons: ${c.decision_reasons.join('; ')}`] : []),
    `Why: ${c.explanation.transition.reason}`,
    `Unavailable inputs: ${c.unavailable_inputs.length} (see JSON for details)`
  ].join('\n');
}
