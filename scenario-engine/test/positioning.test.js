import test from 'node:test';
import assert from 'node:assert/strict';
import {positioningEvidence} from '../positioning.js';
import {sessionFor} from '../calendar.js';
import {ApexScenarioAdapter} from '../apex-adapter.js';
const at='2026-09-10T15:00:00Z',now=Date.parse(at),session=sessionFor('2026-09-10',at);
const feed=()=>({receivedAt:at,payload:{ticker:'SPY',source:'unusual_whales',date:session.id,expiry:null,strikes:[{strike:760,time:'2026-09-10T14:59:00Z',callGammaOi:100,putGammaOi:20},{strike:761,time:'2026-09-10T14:59:00Z',callGammaOi:30,putGammaOi:50}]}});
test('fresh UW profile supplies strike references and explicit modeled gamma sign',()=>{
 const out=positioningEvidence(feed(),session,now);
 assert.equal(out.status,'Connected');assert.equal(out.netGamma,60);
 assert.equal(out.observations[0].value,1);assert.equal(out.levels[0].value,760);
 assert.ok(out.levels.every(r=>r.kind==='STRIKE'));
 assert.match(out.observations[0].method,/not observed dealer inventory/);
 assert.equal(out.levels[0].available_at,at);
});
for(const [name,modify] of [
 ['prior date',f=>f.payload.date='2026-09-09'],
 ['single expiry',f=>f.payload.expiry=session.id],
 ['wrong ticker',f=>f.payload.ticker='QQQ'],
 ['missing timestamp',f=>delete f.payload.strikes[0].time],
 ['future source',f=>f.payload.strikes[0].time='2026-09-10T15:01:00Z'],
 ['duplicate strike',f=>f.payload.strikes.push({...f.payload.strikes[0]})],
 ['stale source',f=>f.payload.strikes[0].time='2026-09-10T14:40:00Z'],
 ['invalid gamma',f=>f.payload.strikes[0].callGammaOi=NaN],
 ['stale receipt',f=>f.receivedAt='2026-09-10T14:50:00Z']
])test('rejects '+name,()=>{const f=feed();modify(f);const out=positioningEvidence(f,session,now);assert.equal(out.levels.length,0);assert.equal(out.observations.length,0);});
test('positioning cannot override missing candle evidence',()=>{
 const adapter=new ApexScenarioAdapter(),card=adapter.update(null,null,now,{positioning:feed()});
 assert.equal(card.trade_permitted,false);
 assert.equal(card.context.metrics.dealerGammaSign,1);
 assert.ok(adapter.dataset.levels.some(l=>l.kind==='STRIKE'));
});
test('inactive zero-exposure rows do not set freshness, but offsetting exposure does',()=>{
 const f=feed();f.payload.strikes.push({strike:700,time:'2026-09-10T12:00:00Z',callGammaOi:0,putGammaOi:0});
 assert.equal(positioningEvidence(f,session,now).status,'Connected');
 f.payload.strikes[2].callGammaOi=10;f.payload.strikes[2].putGammaOi=-10;
 assert.equal(positioningEvidence(f,session,now).status,'Stale');
});
