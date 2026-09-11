import test from 'node:test';
import assert from 'node:assert/strict';
import {ScenarioEngine,catalogue,configure} from '../index.js';
import {dataset,refreshed,at,observation} from './fixtures.js';
const config={allowProvisionalWithoutEventCoverage:true,scenarios:Object.fromEntries(catalogue(configure()).filter(r=>r.level&&r.id!=='opening-range-breakout').map(r=>[r.id,{enabled:false}]))};
function input(){const d=refreshed(dataset([100,100.2,100.4,100.8,101,100.3,100.2]));d.eventCoverage=[];return d;}
test('missing event coverage allows provisional targets and invalidation, never permission',()=>{
  const c=new ScenarioEngine(config).evaluate(input(),at(25));
  assert.equal(c.scenario_id,'opening-range-breakout');assert.equal(c.provisional,true);
  assert.equal(c.directional_bias,'bullish');assert.ok(c.target_ladder.length);
  assert.equal(c.invalidation[0].kind,'opposite-acceptance');assert.equal(c.trade_permitted,false);
  assert.equal(c.confidence_grade,'C');assert.ok(c.decision_reasons.includes('Event calendar coverage unavailable'));
});
test('provisional scenarios keep their reference and transition on invalidation',()=>{
  const engine=new ScenarioEngine(config),d=input();
  const first=engine.evaluate(d,at(25)),pending=engine.evaluate(d,at(30)),last=engine.evaluate(d,at(35));
  assert.equal(pending.scenario_id,first.scenario_id);assert.equal(pending.trigger_status,'waiting');
  assert.equal(pending.invalidation[0].price,first.invalidation[0].price);
  assert.equal(last.scenario_id,'event-risk');
  assert.equal(last.explanation.transition.type,'invalidation');
  assert.equal(last.directional_bias,'neutral');assert.equal(last.trade_permitted,false);
});
test('provisional opt-in never bypasses known event, stale bars, closed session or conflicting internals',()=>{
  const d=input();d.events=[{id:'release',type:'CPI',at:at(30),available_at:at(0),source:'test',status:'scheduled'}];
  assert.equal(new ScenarioEngine(config).evaluate(d,at(25)).provisional,false);
  assert.equal(new ScenarioEngine(config).evaluate(input(),at(90)).directional_bias,'neutral');
  const closed=input();closed.session.isTradingDay=false;
  assert.equal(new ScenarioEngine(config).evaluate(closed,at(25)).directional_bias,'neutral');
  const mixed=input();mixed.observations.push(observation('qqqReturnPct',-1,25));
  assert.equal(new ScenarioEngine(config).evaluate(mixed,at(25)).directional_bias,'neutral');
});
test('default remains strict and verified coverage removes provisional labeling',()=>{
  assert.equal(new ScenarioEngine().evaluate(input(),at(25)).scenario_id,'event-risk');
  assert.equal(new ScenarioEngine(config).evaluate(refreshed(dataset()),at(25)).provisional,false);
});
