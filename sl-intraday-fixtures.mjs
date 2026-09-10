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


const now = Date.parse('2026-09-10T14:00:00Z'); // 10:00 ET
const evaluate = sandbox.slEvaluateSetup;
const e = {setup:'vwap',trigger:601,stop:599,target:605,bid:2,ask:2.1,expiry:'2026-09-10',confirmedAt:now};
const live = {price:601,vwap:600,timestamp:new Date(now).toISOString()};
const r = {gate:'GO',bias:{dir:'LONG'},tapeInputs:{retest:'higherlow',ribbon:'fannedup',internals:'strong'}};
let count = 0;
function check(name, test) { if (!test) throw new Error(name); count++; console.log('  ✓ '+name); }
const read = (ee={},ll={},rr={},clock=now) => evaluate({...r,...rr},{...live,...ll},{...e,...ee},clock);
check('confirmed VWAP long is READY',read().status==='READY');
check('existing CAUTION is preserved',read({}, {}, {gate:'CAUTION'}).status==='CAUTION');
check('existing NO-GO cannot upgrade',read({}, {}, {gate:'NO-GO'}).status==='WAIT');
check('neutral cannot be ready',read({}, {}, {bias:{dir:'NEUTRAL'}}).status==='WAIT');
const short = {gate:'GO',bias:{dir:'SHORT'},tapeInputs:{retest:'lowerhigh',ribbon:'fanneddown',internals:'weak'}};
check('symmetric VWAP short is READY',read({trigger:599,stop:601,target:595},{price:599},short).status==='READY');
check('ORB long requires completed range and breakout',read({setup:'orb',orHigh:600.5,orLow:598}).status==='READY');
check('ORB short confirms below range',read({setup:'orb',orHigh:602,orLow:599.5,trigger:599,stop:601,target:595},{price:599},short).status==='READY');
check('missing OR cannot be invented',read({setup:'orb'}).status==='WAIT');
check('inverted OR is rejected',read({setup:'orb',orHigh:598,orLow:600}).status==='WAIT');
check('ORB stop must invalidate breakout',read({setup:'orb',orHigh:598,orLow:596}).status==='WAIT');
for (const [name,ee,ll,rr] of [
 ['unconfirmed',{confirmedAt:null}],['expired confirmation',{confirmedAt:now-300000}],['future confirmation',{confirmedAt:now+1}],
 ['missing trigger',{trigger:''}],['zero risk',{stop:601}],['wrong target',{target:600}],['blank bid',{bid:''}],['crossed market',{bid:3}],['wide spread',{ask:3}],['wrong expiry',{expiry:'2026-09-11'}],
 ['missing source timestamp',{}, {timestamp:undefined}],['stale source timestamp',{}, {timestamp:new Date(now-120001).toISOString()}],['future source timestamp',{}, {timestamp:new Date(now+31000).toISOString()}],
 ['prior day quote',{}, {timestamp:'2026-09-09T14:00:00Z'}],['bad source timestamp',{}, {timestamp:'bad'}],['bad price',{}, {price:null}],['bad VWAP',{}, {vwap:0}],
 ['untriggered',{}, {price:600.9}],['chasing',{}, {price:602.1}],['invalidation',{}, {price:598.9}],['wrong VWAP side',{}, {vwap:602}],
 ['no retest',{}, {}, {tapeInputs:{...r.tapeInputs,retest:'none'}}],['conflicting ribbon',{}, {}, {tapeInputs:{...r.tapeInputs,ribbon:'fanneddown'}}],['mixed internals',{}, {}, {tapeInputs:{...r.tapeInputs,internals:'mixed'}}],
 ['RR below floor',{target:603}],['unknown setup',{setup:'other'}]
]) check(name+' → WAIT',read(ee,ll,rr).status==='WAIT');
check('R:R uses current executable price',Math.abs(read({}, {price:601.5}).rr-1.4)<1e-9);
for (const iso of ['2026-09-10T13:40:00Z','2026-09-10T19:45:00Z','2026-09-10T21:00:00Z','2026-09-12T14:00:00Z']) {
 const t=Date.parse(iso); check('actual clock blocks '+iso,read({confirmedAt:t,expiry:sandbox.slEtDate(t)},{timestamp:iso},{},t).status==='WAIT');
}
check('winter ET date handles UTC rollover',sandbox.slEtDate(Date.parse('2026-01-10T01:00:00Z'))==='2026-01-09');
const lunch = Date.parse('2026-09-10T16:00:00Z');
check('actual lunch cannot be promoted by manual window',read({confirmedAt:lunch},{timestamp:new Date(lunch).toISOString()},{},lunch).status==='CAUTION');
const run = code => vm.runInContext(code,sandbox);
run('slEditSetup("trigger", 601); slConfirmSetup(true); slSetInput("retest", "higherlow");');
check('tape edit invalidates confirmation',run('SL_EXEC.confirmedAt')===null);
run('slConfirmSetup(true); slEditSetup("ask", 2.2);');
check('quote edit invalidates confirmation',run('SL_EXEC.confirmedAt')===null);
const original = {gate:'GO',bias:{dir:'LONG'},reasons:[],stops:{runner:true,maxHoldMin:15,targets:['support']}};
const applied = sandbox.slApplySetup(original);
check('missing live evidence blocks final gate',applied.gate==='NO-GO');
check('blocked entry clears executable target and hold',!applied.stops.runner && applied.stops.maxHoldMin===0 && applied.stops.targets.length===0);
check('overlay does not mutate base gate',original.gate==='GO' && original.reasons.length===0 && original.stops.runner);
check('renderer escapes supplied text',!sandbox.slSetupHtml({status:'WAIT',name:'<img>',reasons:['<script>'],rr:null,spreadPct:null}).includes('<script>'));
console.log('PASS — '+count+' intraday checks');
