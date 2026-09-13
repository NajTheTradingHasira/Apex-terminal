import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const c=vm.createContext({Date,Intl});vm.runInContext(fs.readFileSync('delayed-breadth.js','utf8'),c);
const {read,ai}=c.ApexDelayedBreadth,now=Date.parse('2026-09-11T15:00:00Z');
const fixture=()=>({delayMinutes:15,latest:{source:'Massive minute aggregates / State Street SPY holdings',session:'2026-09-11',as_of:'2026-09-11T14:45:00Z',available_at:'2026-09-11T14:59:00Z',status:'complete',total:502,covered:502,advancing:300,declining:200,unchanged:2,holdingsDate:'2026-09-10',metrics:{adRatio:1.5,upDownVolumeRatio:2,sp500AboveVwapPct:60}}});
assert.equal(read(fixture(),now).status,'Delayed context');assert.equal(ai(fixture(),now).entryEligible,false);
assert.equal(read(fixture(),now+11*60000).status,'Historical / stale');assert.equal(ai(fixture(),now+11*60000).snapshot,undefined);
for(const modify of [p=>p.latest.source='other',p=>p.latest.as_of='bad',p=>p.latest.available_at='2026-09-11T15:01:00Z',p=>p.latest.covered=400,p=>p.latest.metrics.adRatio=99,p=>p.latest.metrics.sp500AboveVwapPct=101,p=>p.latest.advancing=-1,p=>p.latest.session='2026-09-10',p=>p.latest.metrics.upDownVolumeRatio='2']){const p=fixture();modify(p);assert.equal(read(p,now).snapshot,null);}
const zero=fixture();zero.latest.declining=0;zero.latest.advancing=500;zero.latest.metrics.adRatio=null;assert.equal(read(zero,now).snapshot.metrics.adRatio,null);
const recent=fixture();recent.latest.as_of='2026-09-11T14:59:00Z';assert.notEqual(read(recent,now).status,'Delayed context');
const html=fs.readFileSync('index.html','utf8');assert.ok(!html.includes('advancing: 1847'));assert.ok(!html.includes('BD.newHighs = d.new_highs'));assert.match(html,/ctx.breadth = \{status:'unavailable'/);assert.match(html,/ctx.delayedBreadthContext = globalThis.ApexDelayedBreadth/ );assert.ok(!fs.readFileSync('scenario-engine/apex-adapter.js','utf8').includes('ApexDelayedBreadth'));
console.log('PASS: 21 delayed breadth and isolation checks');
