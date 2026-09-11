import test from 'node:test';
import assert from 'node:assert/strict';
import {parseDocuments,CalendarEvidence,crossAsset} from '../feeds.js';
import {scheduledCoverage} from '../scheduledCoverage.js';
import {nyFedSource} from '../nyFedCalendar.js';
import {FOMC_SOURCE} from '../fomcCalendar.js';
import {screenContracts} from '../contracts.js';
import {replayRecord} from '../history.js';
import {ApexScenarioAdapter} from '../apex-adapter.js';
import {sessionFor} from '../calendar.js';
const at='2026-09-10T14:30:00.000Z',now=Date.parse(at),session=sessionFor('2026-09-10',at);
const feeds=()=>({nyfed:{source:nyFedSource(now).url,fetchedAt:at,data:[],rejected:0,layoutComplete:true,calendarMonth:'2026-09'},fomc:{source:FOMC_SOURCE,fetchedAt:at,yearCounts:{2026:8},rejected:0,meetings:[]}});
test('coverage requires fresh complete official calendars, not an empty response',()=>{
 assert.equal(scheduledCoverage({},at).passed,false);
 assert.equal(scheduledCoverage(feeds(),at).passed,true);
 const f=feeds();f.nyfed.layoutComplete=false;assert.equal(scheduledCoverage(f,at).passed,false);
 f.nyfed.layoutComplete=true;f.nyfed.rejected=1;assert.equal(scheduledCoverage(f,at).passed,false);
 assert.equal(scheduledCoverage(feeds(),'2026-09-10T14:41:00Z').passed,false);
});
test('meeting day requires verified timing and refuses dates alone',()=>{
 const f=feeds();f.fomc.meetings=[{start:'2026-09-10',end:'2026-09-10'}];
 assert.equal(scheduledCoverage(f,at).passed,false);
});
test('document receipt timestamps cannot be refreshed by parsing or forged sources',()=>{
 assert.equal(Object.keys(parseDocuments({documents:{nyfed:{source:'https://wrong.example',html:'',fetchedAt:at}}},now).feeds).length,0);
 assert.equal(Object.keys(parseDocuments({documents:{nyfed:{source:nyFedSource(now).url,html:'',fetchedAt:'2026-09-10T14:31:00Z'}}},now).feeds).length,0);
});
test('known events survive a subsequent calendar outage',()=>{
 const c=new CalendarEvidence();c.feeds=feeds();c.feeds.nyfed.data=[{event:'Release',time:'2026-09-10T15:00:00Z'}];
 assert.equal(c.read(at).events.length,1);c.receive(null,now+1000);
 const next=c.read('2026-09-10T14:30:01Z');assert.equal(next.events.length,1);assert.equal(next.check.passed,false);
});
const asset=()=>({ticker:'QQQ',interval:'1m',data:Array.from({length:60},(_,i)=>({date:new Date(Date.parse(session.open)+i*60000).toISOString(),open:100,high:102,low:99,close:101,volume:1000}))});
test('cross-asset evidence requires continuous completed fresh same-symbol bars',()=>{
 const p=asset();const t='2026-09-10T14:30:05Z';
 assert.equal(crossAsset(p,'QQQ',session,t)[0].metric,'qqqReturnPct');
 assert.deepEqual(crossAsset(p,'IWM',session,t),[]);
 assert.deepEqual(crossAsset(p,'QQQ',session,'2026-09-10T14:33:00Z'),[]);
 p.data.splice(3,1);assert.deepEqual(crossAsset(p,'QQQ',session,t),[]);
});
test('saved checkpoint reproduces from its original pre-evaluation state',()=>{
 const adapter=new ApexScenarioAdapter();
 const scan={metrics:{},bars:Array.from({length:30},(_,i)=>({t:Date.parse(session.open)+i*60000,o:100,h:101,l:99,c:100.5,v:1000,vwap:100}))};
 const card=adapter.update(scan,null,now);
 const record={at,dataset:adapter.dataset,before:adapter.before,config:adapter.engine.config,card};
 assert.equal(replayRecord(record).match,true);
 record.card={...card,decision:'fabricated'};assert.equal(replayRecord(record).match,false);
});
const payload=()=>({status:'received',expiry:'2026-09-10',receivedAt:at,results:[{details:{ticker:'O:SPY260910C00760000',contract_type:'call',shares_per_contract:100,strike_price:760,expiration_date:'2026-09-10'},last_quote:{bid:1,ask:1.05,bid_size:10,ask_size:10,last_updated:(now-1000)*1e6,timeframe:'REAL-TIME'},greeks:{delta:0.5},day:{volume:500},open_interest:1000}]});
const read={bias:{dir:'LONG'},gate:'NO-GO'};
test('UW snapshot creates a review shortlist without fabricating quote approval',()=>{
 const p=payload();p.source='Unusual Whales';
 p.results[0].details.shares_per_contract=null;
 p.results[0].last_quote={bid:1,ask:1.05,last_updated:null,timeframe:'UNVERIFIED'};
 p.results[0].lastTapeTime=at;
 const out=screenContracts(p,read,760,now);
 assert.equal(out.status,'REVIEW');assert.equal(out.reviewCandidates.length,1);
 assert.equal(out.candidates.length,0);assert.equal(out.reviewCandidates[0].quoteAt,null);
 assert.equal(read.gate,'NO-GO');
 p.results[0].last_quote.ask=2;
 assert.equal(screenContracts(p,read,760,now).reviewCandidates.length,0);
});
test('coverage and adapter evaluate at exactly the same checkpoint',()=>{
 const adapter=new ApexScenarioAdapter();
 const coverage=scheduledCoverage(feeds(),at).coverage;
 const card=adapter.update(null,null,now,{eventCoverage:coverage});
 assert.equal(card.decision_reasons.includes('Event calendar coverage unavailable'),false);
 // Point-only coverage must not be stretched to later timestamps.
 const later=adapter.update(null,null,now+1,{eventCoverage:coverage});
 assert.equal(later.decision_reasons.includes('Event calendar coverage unavailable'),true);
});
test('contract screening keeps entry permission separate from liquidity qualification',()=>{
 const out=screenContracts(payload(),read,760,now);assert.equal(out.candidates.length,1);assert.match(out.reason,/NO-GO/);assert.equal(read.gate,'NO-GO');
});
for(const [name,change] of [
 ['stale quote',p=>p.results[0].last_quote.last_updated=(now-31000)*1e6],
 ['delayed quote',p=>p.results[0].last_quote.timeframe='DELAYED'],
 ['missing quote timestamp',p=>delete p.results[0].last_quote.last_updated],
 ['crossed quote',p=>p.results[0].last_quote.bid=2],
 ['wide spread',p=>p.results[0].last_quote.ask=2],
 ['low liquidity',p=>p.results[0].day.volume=1],
 ['missing delta',p=>delete p.results[0].greeks.delta],
 ['wrong expiry',p=>p.results[0].details.expiration_date='2026-09-11'],
 ['wrong side',p=>p.results[0].details.contract_type='put'],
 ['adjusted contract',p=>p.results[0].details.shares_per_contract=10],
 ['future quote',p=>p.results[0].last_quote.last_updated=(now+1000)*1e6],
])test('contract screen rejects '+name,()=>{const p=payload();change(p);assert.equal(screenContracts(p,read,760,now).candidates.length,0);});
test('closed session and stale snapshot cannot qualify contracts',()=>{
 assert.equal(screenContracts(payload(),read,760,Date.parse('2026-09-10T20:00:00Z')).candidates.length,0);
 const p=payload();p.receivedAt='2026-09-10T14:28:00Z';assert.equal(screenContracts(p,read,760,now).candidates.length,0);
});
