import test from 'node:test';
import assert from 'node:assert/strict';
import {aggregateFive,dailyLevels,ApexScenarioAdapter,applyScenarioGate,libraryRows} from '../apex-adapter.js';
import {sessionFor} from '../calendar.js';

const start=Date.parse('2026-09-10T13:30:00Z');
const iso=t=>new Date(t).toISOString();
const session=sessionFor('2026-09-10',iso(start));
const minutes=(count=25)=>({metrics:{},bars:Array.from({length:count},(_,i)=>({t:start+i*60000,o:100+i/10,h:100.2+i/10,l:99.9+i/10,c:100.1+i/10,v:1000,vwap:100+i/20}))});
test('aggregates only complete five-minute groups with truthful receipt time',()=>{
 const now=iso(start+26*60000),out=aggregateFive(minutes(26),session,now);
 assert.equal(out.bars.length,5);assert.equal(out.bars[0].volume,5000);
 assert.ok(Math.abs(out.bars[0].high-100.6)<1e-9);assert.equal(out.bars[0].low,99.9);
 assert.equal(out.bars[0].end,iso(start+300000));assert.equal(out.bars[0].available_at,now);
 assert.equal(out.levels.length,5);assert.equal(out.observations.length,1);
 assert.equal(out.observations[0].value,1);
 assert.match(out.observations[0].method,/within-session/);
});
test('repeated reads preserve availability, revisions receive a new timestamp',()=>{
 const receipts=new Map(),first=iso(start+26*60000),later=iso(start+27*60000);
 const before=aggregateFive(minutes(),session,first,receipts);
 const after=aggregateFive(minutes(),session,later,receipts);
 assert.equal(after.bars[0].available_at,before.bars[0].available_at);
 const revised=minutes();revised.bars[0].v++;
 assert.equal(aggregateFive(revised,session,later,receipts).bars[0].available_at,later);
});
test('incomplete or malformed minute history cannot fabricate 5m bars',()=>{
 const missing=minutes();missing.bars.splice(3,1);
 assert.throws(()=>aggregateFive(missing,session,iso(start+1800000)),/Incomplete/);
 const invalid=minutes();invalid.bars[0].c=1000;
 assert.throws(()=>aggregateFive(invalid,session,iso(start+1800000)),/invalid/);
});
test('today daily close is never used as prior close',()=>{
 const payload={ohlcv:[{date:'2026-09-09',open:100,high:102,low:99,close:101},{date:'2026-09-10',open:105,high:110,low:103,close:109}]};
 const levels=dailyLevels(payload,session,iso(start));
 assert.equal(levels.find(l=>l.kind==='PDC').value,101);
 assert.equal(levels.some(l=>l.kind==='ATR'),false);
 assert.deepEqual(dailyLevels(payload,session,'2026-09-09T15:00:00Z'),[]);
});
test('stale prior session and duplicate dates fail unavailable',()=>{
 const row={date:'2026-09-08',open:100,high:102,low:99,close:101};
 assert.deepEqual(dailyLevels({ohlcv:[row]},session,iso(start)),[]);
 row.date='2026-09-09';assert.deepEqual(dailyLevels({ohlcv:[row,row]},session,iso(start)),[]);
});
test('calendar handles holiday, early close and winter timezone',()=>{
 assert.equal(sessionFor('2026-09-07',iso(start)).isTradingDay,false);
 assert.equal(sessionFor('2026-11-27',iso(start)).close,'2026-11-27T18:00:00.000Z');
 assert.equal(sessionFor('2026-12-01',iso(start)).open,'2026-12-01T14:30:00.000Z');
 assert.throws(()=>sessionFor('2030-01-01',iso(start)),/unavailable/);
});
test('adapter never invents value profile, event coverage or breadth',()=>{
 const engine=new ApexScenarioAdapter(),card=engine.update(minutes(),null,start+26*60000);
 assert.equal(card.trade_permitted,false);
 assert.ok(card.decision_reasons.includes('Event calendar coverage unavailable'));
 assert.ok(card.unavailable_inputs.includes('Missing level: VAH'));
 assert.ok(card.unavailable_inputs.includes('Missing/stale metric: adRatio'));
 assert.equal(card.context.metrics.relativeVolume,1);
 assert.equal(engine.receipts.size,5);
});
test('all 18 original templates appear, including three neutral templates',()=>{
 const rows=libraryRows(null);assert.equal(rows.length,18);
 assert.equal(rows.filter(r=>r.rule.direction==='neutral').length,3);
 assert.ok(rows.some(r=>r.rule.id==='event-repricing'));
 assert.ok(rows.some(r=>r.rule.id==='late-day-strike-pin'));
});
test('required scenario gate can only reduce entry permission',()=>{
 const r={gate:'GO',bias:{dir:'LONG'},reasons:[],stops:{runner:true,maxHoldMin:15,targets:['x'],plan:{runnerEligible:true}}};
 assert.equal(applyScenarioGate(r,null,false),r);
 const blocked=applyScenarioGate(r,null,true);
 assert.equal(blocked.gate,'NO-GO');assert.equal(blocked.stops.runner,false);assert.equal(blocked.stops.plan.runnerEligible,false);
 assert.equal(r.gate,'GO');assert.equal(r.stops.runner,true);
 const approved={trade_permitted:true,directional_bias:'bullish',decision_reasons:[]};
 assert.equal(applyScenarioGate({...r,gate:'CAUTION'},approved,true).gate,'CAUTION');
 assert.equal(applyScenarioGate({...r,gate:'NO-GO'},approved,true).gate,'NO-GO');
 assert.equal(applyScenarioGate(r,{...approved,directional_bias:'bearish'},true).gate,'NO-GO');
});
