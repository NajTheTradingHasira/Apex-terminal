import test from 'node:test';
import assert from 'node:assert/strict';
import {ScenarioEngine} from '../engine.js';
import {dataset,refreshed,at,level} from './fixtures.js';

test('missing premarket levels preserve an independently confirmed opening-range read',()=>{
 const data=refreshed(dataset());
 const without=new ScenarioEngine().evaluate(data,at(25));
 const withPremarket=new ScenarioEngine().evaluate({...data,levels:[...data.levels,level('PMH',110),level('PML',90)]},at(25));
 assert.equal(without.scenario_id,'opening-range-breakout');
 assert.equal(without.trade_permitted,true);
 for(const key of ['scenario_id','directional_bias','trigger_status','target_ladder','invalidation','confidence_grade'])assert.deepEqual(without[key],withPremarket[key]);
 assert.ok(without.unavailable_inputs.includes('Missing level: PMH'));
 assert.ok(without.target_ladder.every(t=>!['PMH','PML'].includes(t.kind)));
});

test('rules explicitly requiring premarket references stay ineligible without those levels',()=>{
 const data=refreshed(dataset());
 const card=new ScenarioEngine({scenarios:{'opening-range-breakout':{level:'PMH'}}}).evaluate(data,at(25));
 const candidates=card.explanation.candidates.filter(c=>c.id==='opening-range-breakout');
 assert.ok(candidates.length);
 assert.ok(candidates.every(c=>!c.eligible&&c.reference===null&&c.target_ladder.length===0));
 assert.notEqual(card.scenario_id,'opening-range-breakout');
});

test('partial event coverage keeps useful OR analysis separate from trade permission',()=>{
 const data=refreshed(dataset());data.eventCoverage=[];
 const card=new ScenarioEngine({allowProvisionalWithoutEventCoverage:true}).evaluate(data,at(25));
 assert.equal(card.scenario_id,'opening-range-breakout');
 assert.equal(card.directional_bias,'bullish');
 assert.equal(card.trigger_status,'confirmed');
 assert.ok(card.target_ladder.length&&card.expected_path.length&&card.invalidation.length);
 assert.equal(card.next_state_if_invalidated.scenario_id,'failed-opening-range-breakout');
 assert.equal(card.trade_permitted,false);
 assert.ok(card.decision_reasons.includes('Event calendar coverage unavailable'));
});
