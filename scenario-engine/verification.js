import { ScenarioEngine } from './engine.js';
import { replay } from './replay.js';
import { normalize } from './model.js';
import { ms } from './time.js';

const collections=['bars','levels','observations','events','eventCoverage'];
export function availablePrefix(data,at) {
  return {...data,...Object.fromEntries(collections.map(key=>[key,(data[key]??[]).filter(row=>ms(row.available_at)<=ms(at))]))};
}

/** Independent card invariants; these do not assert that a trading thesis is profitable. */
export function auditReplayCard(card,state,config,previous) {
  const failures=[];
  const check=(condition,name)=>{if(!condition)failures.push(name);};
  const transition=card.explanation.transition;
  check(transition.to===card.scenario_id,'Transition destination differs from card state');
  const namedDestination=transition.reason.match(/; transition to ([a-z-]+)/)?.[1];
  if(namedDestination)check(namedDestination===transition.to,'Transition explanation names a different destination');
  const refs=[...card.context.levels,...card.target_ladder,...(card.acceptance_status.level?[card.acceptance_status.level]:[])];
  check(refs.every(l=>(!l.available_at||ms(l.available_at)<=ms(card.timestamp))&&ms(l.as_of)<=ms(card.timestamp)), 'Future reference entered the card');
  check(card.target_ladder.every(t=>state.levels.some(l=>l.id===t.level&&l.value===t.price&&l.source===t.source)), 'Target lacks a contemporaneous source level');
  const d=card.directional_bias==='bullish'?1:-1;
  check(card.target_ladder.every(t=>state.current&&d*(t.price-state.current.close)>config.acceptance.buffer),'Target is not ahead of SPY');
  if(card.trade_permitted) {
    check(card.directional_bias!=='neutral'&&card.trigger_status==='confirmed'&&card.acceptance_status.status==='accepted','Trade permitted before directional acceptance');
    check(!state.dataStale&&!['premarket','closed'].includes(state.phase),'Trade permitted with stale bars or outside regular hours');
    check(!state.blockingEvents.length&&(!config.requireEventCoverage||state.eventCoverage),'Trade permitted with event uncertainty');
    check((!config.requireVwap||!!state.getLevel('VWAP'))&&(!config.requireVolume||state.metrics.relativeVolume!==undefined),'Trade permitted with missing required inputs');
    check(!card.provisional&&!card.decision_reasons.length,'Trade permitted despite explicit blocking reasons');
  }
  if(previous&&['invalidation','expiry'].includes(card.explanation.transition.type)) {
    check([previous.next_state_if_invalidated.scenario_id,'no-trade-chop','event-risk'].includes(card.scenario_id),'Invalidation skipped the defined successor or no-trade fallback');
  }
  return failures;
}

/** Replay full data and arrival-only data through identical chronological checkpoints.
 * PASS verifies supplied timestamps and engine behavior, not vendor accuracy or profitability.
 */
export function verifyReplay(dataset,options={}) {
  const run=replay(dataset,options),arrivalEngine=new ScenarioEngine(run.config);
  const repeated=replay(dataset,{config:run.config,checkpoints:run.cards.map(c=>c.timestamp)});
  const checkpoints=[];
  for(let i=0;i<run.cards.length;i++) {
    const card=run.cards[i],prefix=availablePrefix(dataset,card.timestamp);
    const arrivalCard=arrivalEngine.evaluate(prefix,card.timestamp);
    const failures=auditReplayCard(card,normalize(prefix,card.timestamp,run.config),run.config,run.cards[i-1]);
    if(JSON.stringify(card)!==JSON.stringify(arrivalCard))failures.push('Future-data leakage: full-data replay differs from arrival-only replay');
    if(JSON.stringify(card)!==JSON.stringify(repeated.cards[i]))failures.push('Replay is not deterministic');
    checkpoints.push({at:card.timestamp,scenario:card.active_scenario,direction:card.directional_bias,trigger:card.trigger_status,decision:card.decision,passed:!failures.length,failures});
  }
  return {session:run.session,data_label:run.data_label,config:run.config,passed:checkpoints.every(c=>c.passed),
    methodology:'Every scheduled five-minute checkpoint, including outages, is evaluated twice from full data and once from only inputs available by that time. Cards must match exactly. Source levels, acceptance gates and defined invalidation transitions are checked.',
    limitations:'Uses supplied availability timestamps; does not certify provider timestamps, feed completeness, profitability, fills, or calibrated confidence. Synthetic sessions verify mechanics only.',
    summary:{checkpoints:checkpoints.length,passed:checkpoints.filter(c=>c.passed).length,failed:checkpoints.filter(c=>!c.passed).length,permitted:run.summary.permitted},
    transitions:run.summary.transitions,checkpoints};
}
