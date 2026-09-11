import test from 'node:test';
import assert from 'node:assert/strict';
import {profileEvidence,previousSession,loadProfile,storeProfile} from '../value-profile.js';
import {sessionFor} from '../calendar.js';
import {ApexScenarioAdapter} from '../apex-adapter.js';
const at='2026-09-10T15:00:00Z',now=Date.parse(at),session=sessionFor('2026-09-10',at);
const profile=()=>({symbol:'SPY',scope:'RTH',valueArea:70,target:session.id,date:'2026-09-09',val:750,poc:755,vah:760,source:'Test chart',settings:'100 rows',receivedAt:at});
test('profile supplies point-in-time levels and explicit manual provenance',()=>{
 const e=profileEvidence(profile(),session,now);assert.equal(e.levels.length,3);
 assert.deepEqual(e.levels.map(l=>l.kind),['VAH','VAL','VPOC']);
 assert.ok(e.levels.every(l=>l.available_at===at&&l.valid_until===session.close));
 assert.match(e.levels[0].method,/manually transcribed/);
});
for(const [name,change] of [
 ['wrong date',p=>p.date='2026-09-08'],['wrong target',p=>p.target='2026-09-11'],
 ['wrong symbol',p=>p.symbol='ES'],['extended hours',p=>p.scope='ETH'],
 ['wrong area',p=>p.valueArea=80],['reversed levels',p=>p.val=761],
 ['POC outside',p=>p.poc=770],['nonfinite',p=>p.vah=Infinity],
 ['missing source',p=>p.source=''],['missing settings',p=>p.settings=''],
 ['future receipt',p=>p.receivedAt='2026-09-10T15:01:00Z'],
 ['before profile close',p=>p.receivedAt='2026-09-09T15:00:00Z']
])test('rejects '+name,()=>{const p=profile();change(p);assert.equal(profileEvidence(p,session,now).levels.length,0);});
test('expiry, holiday and previous early close are enforced',()=>{
 assert.equal(profileEvidence(profile(),session,Date.parse(session.close)+1).levels.length,0);
 assert.equal(previousSession('2026-09-08').id,'2026-09-04');
 assert.equal(previousSession('2026-11-30').close,'2026-11-27T18:00:00.000Z');
});
test('storage reload rebases receipt and clear removes saved values',()=>{
 const m=new Map(),storage={getItem:k=>m.get(k),setItem:(k,v)=>m.set(k,v),removeItem:k=>m.delete(k)};
 storeProfile(storage,profile());assert.equal(loadProfile(storage,now+1000).receivedAt,'2026-09-10T15:00:01.000Z');
 storeProfile(storage,null);assert.equal(loadProfile(storage),null);
});
test('manual profile cannot manufacture candle evidence',()=>{
 const adapter=new ApexScenarioAdapter(),card=adapter.update(null,null,now,{valueProfile:profile()});
 assert.equal(card.trade_permitted,false);assert.equal(adapter.dataset.levels.filter(l=>['VAH','VAL','VPOC'].includes(l.kind)).length,3);
});
test('profile revision and removal discard prior scenario state',()=>{
 const a=new ApexScenarioAdapter(),p=profile();a.update(null,null,now,{valueProfile:p});
 const first=a.engine;a.update(null,null,now,{valueProfile:p});assert.equal(a.engine,first);
 p.poc=756;a.update(null,null,now,{valueProfile:p});assert.notEqual(a.engine,first);
 const revised=a.engine;a.update(null,null,now);assert.notEqual(a.engine,revised);
 assert.equal(a.dataset.levels.filter(l=>l.id.startsWith('manual-profile')).length,0);
});
