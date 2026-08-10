// sl-struct-keys.mjs — the adoption boundary's key set.
//
//   node sl-struct-keys.mjs   (from the repo root)
//
// Same shape as sl-pivot-fixtures.mjs: evaluate the real inline script out of
// index.html under node and drive the SHIPPED slFetchStructure(), so this can
// never drift from what the browser runs.
//
// ── Why an exhaustive key set, when field-level tests already exist ──────
//
// Every field this boundary has ever lost was lost the same way: the backend
// served it, slFetchStructure did not name it, and it vanished with nothing on
// screen to say so. `degraded` went that way, then the four provenance fields a
// release later, and each was found by hand, in production.
//
// Field-level tests cannot catch that class. They assert what IS named; a
// dropped key is precisely the thing nobody wrote an assertion for. Only an
// exhaustive key set fails when a key goes missing that no test knew to ask
// about — which is also how one terminal ends up quietly a field behind the
// other two, since this is a hand-ported block living in three repos.
//
// NOTE the key set here is 14, not nexus's 16. APEX keeps `degraded` and
// `degradedReason` on SL_STRUCTURE_SOURCE rather than on the struct, as hashira
// does. That divergence is pre-existing; it is recorded here so the difference
// reads as known rather than as a porting slip.

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
    console: { log() {}, warn() {}, error() {}, info() {} },
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

// `let SL_STRUCTURE` lives in the context's GLOBAL LEXICAL scope, not as a
// property of the sandbox object, so everything below goes through
// runInContext. Reading sandbox.SL_STRUCTURE would read a shadow that
// slFetchStructure never writes, and every assertion would pass vacuously.
const run = (code) => vm.runInContext(code, sandbox);

// A payload that passes slValidStructure: flip < support < reclaim.
const BASE = {
    asOf: '2026-08-07', reclaim: 745.73, support: 729.10, flip: 725.86,
    staleDays: 10, vixFragile: 25, source: 'fixture', revision: '2026-08-07.1',
};

/** Drive the real slFetchStructure() against a stubbed response. */
async function adopt(extra) {
    const payload = JSON.stringify({ ...BASE, ...extra });
    run(`fetch = () => Promise.resolve({ ok: true, json: () => Promise.resolve(${payload}) });`);
    await run('slFetchStructure()');
    return JSON.parse(run('JSON.stringify(SL_STRUCTURE)'));
}
/** Keys straight off the live object, so `undefined` values still count. */
async function adoptKeys(extra) {
    await adopt(extra);
    return JSON.parse(run('JSON.stringify(Object.keys(SL_STRUCTURE).sort())'));
}

const results = [];
const t = (name, pass, detail) => results.push([name, !!pass, detail]);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Alphabetical, so a new field is inserted rather than appended and the diff
// shows WHERE it landed.
const KEYS = [
    'asOf', 'avwaps', 'calculatedAsOf', 'dataSource', 'derivationVersion',
    'eventBlackouts', 'flip', 'originReview', 'pivot', 'reclaim', 'source',
    'staleDays', 'support', 'vixFragile',
];

const FULL = {
    derivationVersion: 'spy_weekly_structure.v1', dataSource: 'tradingview',
    originReview: '2026-08-09', calculatedAsOf: '2026-08-07',
    pivot: { level: 748.8 },
    avwaps: [{ anchor: '2026-03-30', level: 725.86 }],
    eventBlackouts: [{ start: '2026-08-12T14:00:00Z', end: '2026-08-12T14:30:00Z', label: 'CPI' }],
};

const sparseKeys = await adoptKeys({});
const fullKeys = await adoptKeys(FULL);

t('adopts exactly these keys and no others', same(sparseKeys, KEYS),
  sparseKeys.join(','));

// The "absent -> null with the key present" contract, stated as a property:
// presence of a key can never be conditional on presence of a value.
t('the key set does not depend on what the backend sent', same(fullKeys, sparseKeys),
  fullKeys.join(','));

// undefined is the shape that survives JSON.stringify by disappearing, so it
// reads as "key present" here and "key absent" one hop downstream.
//
// Deliberately re-adopted from the SPARSE payload first. Run against the state
// left by the FULL payload, every field has a served value and nothing can be
// undefined, so the check passes regardless of what the defaults do. Undefined
// can only appear where a DEFAULT is applied.
await adopt({});
const liveKeys = JSON.parse(run('JSON.stringify(Object.keys(SL_STRUCTURE).sort())'));
const rtKeys = Object.keys(JSON.parse(run('JSON.stringify(SL_STRUCTURE)'))).sort();
t('no adopted value is ever undefined', same(liveKeys, rtKeys),
  'live ' + liveKeys.length + ' vs round-tripped ' + rtKeys.length);

const sparse = await adopt({});
t('avwaps defaults to null, not []', sparse.avwaps === null, String(sparse.avwaps));
t('eventBlackouts defaults to null, not []', sparse.eventBlackouts === null,
  String(sparse.eventBlackouts));

const served = await adopt({
    avwaps: [{ anchor: '2026-03-30', level: 725.86 }],
    eventBlackouts: [],
});
t('carries a served avwaps array through',
  same(served.avwaps, [{ anchor: '2026-03-30', level: 725.86 }]));
// A served [] survives as [] — distinguishable from absent, which is null.
t('a served empty array stays [] and does not become null',
  Array.isArray(served.eventBlackouts) && served.eventBlackouts.length === 0);

let allNull = true;
for (const bad of [{}, 'CPI', 42, true]) {
    const a = await adopt({ avwaps: bad, eventBlackouts: bad });
    if (a.avwaps !== null || a.eventBlackouts !== null) allNull = false;
}
t('non-array shapes degrade to null rather than passing through', allNull);

// ── report ──────────────────────────────────────────────────────────────
let failed = 0;
for (const [name, pass, detail] of results) {
    if (!pass) failed += 1;
    console.log((pass ? '  ok  ' : '✗ FAIL') + '  ' + name + (detail && !pass ? '\n          got: ' + detail : ''));
}
console.log('\n' + (failed ? '✗ ' + failed + ' failed / ' : '✓ ') + results.length + ' checks');
process.exit(failed ? 1 : 0);
