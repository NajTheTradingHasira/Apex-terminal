import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const m = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/i.exec(src);
if (!m) {
    console.error('✗ FAIL: no inline <script> block found in index.html');
    process.exit(1);
}

// ── Just enough DOM to let the panel boot and repaint ────────────────────
const bar = { innerHTML: '' };
const stub = () => ({
    innerHTML: '', textContent: '', value: '', style: {},
    classList: { add() {}, remove() {}, toggle() {} }, dataset: {},
    appendChild() {}, setAttribute() {}, removeAttribute() {}, addEventListener() {},
    querySelector: () => null, querySelectorAll: () => [], insertAdjacentHTML() {},
    focus() {}, blur() {}, remove() {}, closest: () => null,
});
const sandbox = {
    document: {
        getElementById: (id) => (id === 'slStructureBar' ? bar : stub()),
        querySelector: () => stub(), querySelectorAll: () => [],
        createElement: () => stub(), addEventListener() {},
        body: stub(), head: stub(), documentElement: { style: { setProperty() {} } },
    },
    console: { log() {}, warn() {}, error() {}, info() {} },   // silence panel boot chatter
    setTimeout: () => 0, setInterval: () => 0, clearInterval() {}, clearTimeout() {},
    requestAnimationFrame: () => 0,
    fetch: () => Promise.reject(new Error('no network in fixtures')),
    localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
    matchMedia: () => ({ matches: false, addEventListener() {} }),
    location: { href: '', search: '', hash: '' }, navigator: { userAgent: 'node' },
    WebSocket: function () {}, alert() {}, addEventListener() {},
    Date, Math, JSON, Number, String, Array, Object, isFinite, parseFloat, parseInt,
    Intl, RegExp, Error, Promise,
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(m[1], sandbox, { timeout: 20000 });



const start=Date.parse('2026-09-10T13:30:00Z');
const now=start+28*60000+5000;
const scan=sandbox.slCandleScan;
const run=code=>vm.runInContext(code,sandbox);
function fixture(type='orb',sign=1) {
 const data=Array.from({length:25},(_,i)=>({date:new Date(start+i*60000).toISOString(),open:100,high:type==='orb'?100.2:100.1,low:99.9,close:100,volume:1000}));
 const tail=type==='orb'?[[100.1,100.45,100.1,100.4,1000],[100.35,100.42,100.19,100.38,1000],[100.38,100.49,100.35,100.44,2000]]:[[100,100.25,99.99,100.2,1000],[100.04,100.16,100,100.14,1000],[100.14,100.21,100.12,100.18,2000]];
 tail.forEach(([open,high,low,close,volume],i)=>data.push({date:new Date(start+(25+i)*60000).toISOString(),open,high,low,close,volume}));
 if(sign<0) data.forEach(b=>{const {open,high,low,close}=b; Object.assign(b,{open:200-open,high:200-low,low:200-high,close:200-close});});
 return {ticker:'SPY',interval:'1m',data};
}
let count=0;
function check(name,test) {if(!test) throw new Error(name); count++; console.log('  ✓ '+name);}
const candidate=(p,clock=now)=>scan(p,clock).candidates[0];
for(const type of ['orb','vwap']) for(const sign of [1,-1]) {
 const p=fixture(type,sign), s=scan(p,now), c=s.candidates.find(x=>x.setup===type);
 check(type+' '+sign+' completed sequence detected',!!c && c.dir===(sign>0?'LONG':'SHORT'));
 check(type+' '+sign+' levels ordered',sign*(c.trigger-c.stop)>0 && sign*(c.target-c.trigger)>0);
 check(type+' '+sign+' stable event id',scan(p,now+10000).candidates.some(x=>x.id===c.id));
}
check('forming confirmation cannot trigger',!scan(fixture(),now-10000).candidates.length);
check('exact close still needs publication grace',!scan(fixture(),now-5000).candidates.length);
check('grace boundary permits completed bar',!!candidate(fixture()));
let p=fixture(); p.data.pop();
check('break and retest alone do not trigger',!scan(p,now).candidates.length);
p=fixture(); p.data[27].close=100.4;
check('wick through trigger is insufficient',!candidate(p));
p=fixture(); p.data[27].volume=1000;
check('weak confirmation volume rejected',!candidate(p));
p=fixture(); p.data[27].low=100.17;
check('stop hit in confirmation bar rejected',!candidate(p));
p=fixture(); p.data.push({date:new Date(start+28*60000).toISOString(),open:100.44,high:100.46,low:100.17,close:100.44,volume:2000});
check('later stop hit invalidates signal',!candidate(p,now+60000));
p=fixture(); p.data[27].high=101;p.data[27].close=100.6;
check('extended confirmation rejected',!candidate(p));
p=fixture(); p.data[27].high=101;
check('target already touched cannot offer a fresh entry',!candidate(p));
p=fixture(); p.data.splice(10,1);
check('missing middle bar invalidates session',!scan(p,now).valid);
p=fixture(); p.data.shift();
check('missing opening bar invalidates VWAP',!scan(p,now).valid);
p=fixture(); p.data.push({...p.data[10]});
check('identical duplicate deduplicated',scan(p,now).bars.length===28);
p.data[p.data.length-1].volume++;
check('conflicting duplicate rejected',!scan(p,now).valid);
p=fixture(); p.data.reverse();
check('out-of-order response normalized',!!candidate(p));
for(const edit of [b=>b.volume=0,b=>b.close=null,b=>b.high=99,b=>b.low=101,b=>b.open='100',b=>b.volume=Infinity,b=>b.date=0,b=>b.date='2026-09-10T09:30:00']) {
 p=fixture();edit(p.data[0]);check('invalid candle rejected',!scan(p,now).valid);
}
check('stale candles cannot enable detection',!scan(fixture(),now+120000).valid);
check('prior day session cannot roll forward',!scan(fixture(),now+86400000).valid);
p=fixture();p.interval='5m';check('wrong interval rejected',!scan(p,now).valid);
p=fixture();p.ticker='QQQ';check('wrong symbol rejected',!scan(p,now).valid);
p=fixture();p.data.push({...p.data[0],date:new Date(now+60000).toISOString()});check('future data rejected',!scan(p,now).valid);
p=fixture();p.data=p.data.slice(0,24);check('EMA and volume warmup enforced',!scan(p,start+24*60000+5000).valid);
p=fixture();p.data.push({...p.data[0],date:'2026-09-10T13:29:00Z'});check('premarket does not change OR',scan(p,now).metrics.orHigh===100.2);
const metrics=scan(fixture(),now).metrics;
check('volume ratio excludes confirming bar',metrics.relvol===2);
check('source freshness uses candle close',metrics.lastCloseAt===start+28*60000 && metrics.ageMs===5000);
const clock=Date.now; Date.now=()=>now;
try {
 run('SL_CANDLE.payload='+JSON.stringify(fixture())+'; SL_EXEC.setup="orb";');
 check('automatic tape derives higher-low',run('slTapeInputs().retest')==='higherlow');
 check('automatic tape derives ribbon',run('slTapeInputs().ribbon')==='fannedup');
 run('SL_STATE.inputs.internals="strong"; SL_STATE.inputs.opening="above";');
 const r={gate:'GO',bias:{dir:'LONG'},reasons:[],stops:{runner:true}};
 sandbox.slApplySetup(r); // register new event before quote confirmation
 run('SL_EXEC.bid=2;SL_EXEC.ask=2.1;SL_EXEC.expiry="2026-09-10";slConfirmSetup(true);');
 check('auto requires actual contract confirmation',run('SL_EXEC.confirmedAt')===now);
 check('automatic candidate can pass setup gate',sandbox.slApplySetup(r).gate==='GO');
 check('auto never promotes NO-GO',sandbox.slApplySetup({...r,gate:'NO-GO'}).gate==='NO-GO');
 check('auto never promotes CAUTION',sandbox.slApplySetup({...r,gate:'CAUTION'}).gate==='CAUTION');
 check('bias conflict blocks detected setup',sandbox.slApplySetup({...r,bias:{dir:'SHORT'}}).gate==='NO-GO');
 const RealDate=sandbox.Date;
 sandbox.Date=class extends RealDate { constructor(...args) { super(...(args.length?args:[now])); } static now() {return now;} };
 const nodes=new Map();
 sandbox.document.getElementById=id=>{if(!nodes.has(id))nodes.set(id,stub());return nodes.get(id);};
 run('SL_STATE.live={price:100.44}; PRICE_CACHE["^VIX"]={price:16,at:Date.now()}; Object.assign(SL_STRUCTURE,{asOf:"2026-09-10",reclaim:99,support:98,flip:97,degraded:false}); slRenderLocalRead();');
 check('rendered gate uses automatic tape',nodes.get('slGateOut').innerHTML.includes('>GO</span>'));
 check('AI context agrees with rendered gate',run('slStructureContext().local_read.gate')==='GO');
 check('AI context omits the raw candle series',!('bars' in run('slStructureContext().candle_detection')));
 sandbox.Date=RealDate;
 run('SL_CANDLE.error="network failed";');
 check('fetch failure blocks old payload',sandbox.slApplySetup(r).gate==='NO-GO');
 check('lost event clears quote confirmation',run('SL_EXEC.confirmedAt')===null);
 run('SL_CANDLE.mode="manual"; SL_STATE.inputs.retest="lowerhigh";');
 check('manual mode restores manual tape inputs',run('slTapeInputs().retest')==='lowerhigh');
} finally {Date.now=clock;}
sandbox.AbortController=AbortController;
run('SL_CANDLE.mode="auto"; SL_CANDLE.error="";');
let fetchCount=0;
sandbox.fetch=async()=>{fetchCount++; return {ok:true,json:async()=>fixture()};};
await Promise.all([sandbox.slFetchCandles(true),sandbox.slFetchCandles(true)]);
check('overlapping polls share one request',fetchCount===1);
check('successful fetch clears loading and error',!run('SL_CANDLE.loading') && !run('SL_CANDLE.error'));
await sandbox.slFetchCandles();
check('30-second throttle avoids redundant fetch',fetchCount===1);
sandbox.fetch=async()=>({ok:true,json:async()=>({ticker:'QQQ',interval:'1m',data:[]})});
await sandbox.slFetchCandles(true);
check('wrong-ticker HTTP success blocks cached data',!!run('SL_CANDLE.error'));
sandbox.fetch=async()=>{const e=new Error('timeout');e.name='AbortError';throw e;};
await sandbox.slFetchCandles(true);
check('timeout is visible and clears loading',run('SL_CANDLE.error').includes('timed out') && !run('SL_CANDLE.loading'));
console.log('PASS — '+count+' candle checks');
