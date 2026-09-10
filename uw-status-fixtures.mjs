// uw-status-fixtures.mjs — the Unusual Whales status line.
//
//   node uw-status-fixtures.mjs   (from the repo root)
//
// Same shape as wl-stage-fixtures.mjs / sl-struct-keys.mjs: evaluate the real
// inline script out of index.html under node and drive the SHIPPED
// uwStatusSummary() / uwPartReason(), so this can never drift from what the
// browser runs.
//
// ── Why this boundary needs a guard ─────────────────────────────────────
//
// fetchUWData() used to end, unconditionally, with
//
//     statusEl.textContent = `✓ Unusual Whales data loaded for ${ticker}`;
//
// Observed live 2026-09-10: the Railway key was revoked, all four UW routes
// returned HTTP 401, both tables read "unavailable" — and the status line
// still said "✓ loaded". Two separate mistakes stacked:
//
//   1. `.then(r => r.ok ? r.json() : null)` threw the HTTP status away, so a
//      401 and a genuinely empty result were the same value: null.
//   2. The status line was never derived from the results in the first place.
//
// That is the `degraded` failure mode: a fallback rendered as if it were
// vouched for. A green check on a dead feed is strictly worse than no status
// line at all, because the operator stops looking. The whole point of the
// check mark is that it is load-bearing.
//
// So the assertions below are mostly negative. They pin what uwStatusSummary
// must REFUSE to claim: no '✓' unless every part actually succeeded, no
// 'HTTP 0' standing in for a dead socket, and no silent promotion of a render
// crash to a success.

import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const m = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/i.exec(src);
if (!m) {
    console.error('✗ FAIL: no inline <script> block found in index.html');
    process.exit(1);
}

// ── Just enough DOM for the inline script to evaluate ────────────────────
const stub = () => ({
    innerHTML: '', textContent: '', value: '', style: {},
    classList: { add() {}, remove() {}, toggle() {} }, dataset: {},
    appendChild() {}, setAttribute() {}, removeAttribute() {}, addEventListener() {},
    querySelector: () => null, querySelectorAll: () => [], insertAdjacentHTML() {},
    focus() {}, blur() {}, remove() {}, closest: () => null, options: [],
});
const sandbox = {
    document: {
        getElementById: () => stub(),
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

// `function uwStatusSummary` lives in the context's GLOBAL LEXICAL scope, not
// as a property of the sandbox object, so everything below goes through
// runInContext. Reading sandbox.uwStatusSummary would read undefined and every
// assertion would pass vacuously.
const run = (code) => vm.runInContext(code, sandbox);
const json = (v) => JSON.stringify(v);

const results = [];
const t = (name, pass, detail) => results.push([name, !!pass, detail]);
const eq = (name, got, want) => t(name, got === want, `${json(got)} !== ${json(want)}`);

// ── 0. the function has to exist at all ──────────────────────────────────
// The defect this file guards was invisible precisely because there was no
// function here to call. If the fix is reverted, everything below is a
// ReferenceError rather than a quiet pass.
const typeOf = (n) => run(`typeof ${n}`);
eq('uwStatusSummary is a top-level function', typeOf('uwStatusSummary'), 'function');
eq('uwPartReason is a top-level function', typeOf('uwPartReason'), 'function');
eq('uwFetch is a top-level function', typeOf('uwFetch'), 'function');

if (typeOf('uwStatusSummary') !== 'function') {
    console.log('✗ FAIL  uwStatusSummary is missing — the status line cannot be derived from the results');
    console.log('\n✗ 1 failed / 1 checks');
    process.exit(1);
}

const summary = (ticker, parts) => run(`uwStatusSummary(${json(ticker)}, ${json(parts)})`);

const OK = (name) => ({ name, ok: true, status: 200 });
const HTTP = (name, status) => ({ name, ok: false, status, error: 'HTTP ' + status });
const NET = (name, msg) => ({ name, ok: false, status: 0, error: msg || 'Failed to fetch' });
const RENDER = (name, msg) => ({ name, ok: false, status: 200, error: 'render: ' + (msg || 'x') });

const NAMES = ['flow', 'dark pool', 'IV', 'market tide'];
const KEY_HINT = 'check UNUSUAL_WHALES_API_KEY on Railway';

// ── 1. every route healthy — the only case that earns the check mark ─────
const all = summary('SPY', NAMES.map(OK));
eq('all four ok → level ok', all.level, 'ok');
eq('  text is the shipped success line', all.text, '✓ Unusual Whales data loaded for SPY');

// ── 2. the live 2026-09-10 defect: revoked key, 401 on every route ───────
const dead = summary('SPY', NAMES.map(n => HTTP(n, 401)));
eq('all four 401 → level fail', dead.level, 'fail');
t('  text starts with ✗, never ✓', dead.text.startsWith('✗'), dead.text);
t('  text never claims loaded', !/loaded/.test(dead.text), dead.text);
t('  text names the HTTP status', dead.text.includes('HTTP 401'), dead.text);
t('  text says all 4 routes', /all 4 routes/.test(dead.text), dead.text);
t('  401 points the operator at the key', dead.text.includes(KEY_HINT), dead.text);

// ── 3. some up, some down ────────────────────────────────────────────────
const half = summary('SPY', [HTTP('flow', 401), HTTP('dark pool', 401), OK('IV'), OK('market tide')]);
eq('two ok + two 401 → level partial', half.level, 'partial');
t('  text starts with ⚠', half.text.startsWith('⚠'), half.text);
t('  text names the flow failure', /flow HTTP 401/.test(half.text), half.text);
t('  text names the dark pool failure', /dark pool HTTP 401/.test(half.text), half.text);
t('  text does not name the healthy parts', !/IV|market tide/.test(half.text), half.text);
t('  401 still points at the key', half.text.includes(KEY_HINT), half.text);
eq('  text is the shipped partial line', half.text,
   '⚠ Unusual Whales partial for SPY — dark pool HTTP 401, flow HTTP 401 · check UNUSUAL_WHALES_API_KEY on Railway');

// ── 4. a dead socket is not an HTTP status ───────────────────────────────
// status 0 means the request never reached a server. Printing "HTTP 0" sends
// the operator looking for a server-side 0 that does not exist.
const netOne = summary('SPY', [NET('flow'), OK('dark pool'), OK('IV'), OK('market tide')]);
eq('one network failure → level partial', netOne.level, 'partial');
t('  reads network error', /flow network error/.test(netOne.text), netOne.text);
t('  never prints HTTP 0', !/HTTP 0/.test(netOne.text), netOne.text);
t('  a network error is not a key problem', !netOne.text.includes(KEY_HINT), netOne.text);

const netAll = summary('SPY', NAMES.map(n => NET(n)));
eq('all four network failures → level fail', netAll.level, 'fail');
eq('  text is the negative-control line', netAll.text,
   '✗ Unusual Whales unavailable for SPY — network error on all 4 routes');
t('  never prints HTTP 0', !/HTTP 0/.test(netAll.text), netAll.text);

// ── 5. a 500 is the backend's problem, not the key's ─────────────────────
const five = summary('SPY', [HTTP('flow', 500), OK('dark pool'), OK('IV'), OK('market tide')]);
eq('one 500 → level partial', five.level, 'partial');
t('  names the 500', /flow HTTP 500/.test(five.text), five.text);
t('  no key hint on a 500', !five.text.includes(KEY_HINT), five.text);

const forbidden = summary('SPY', [HTTP('flow', 403), OK('dark pool'), OK('IV'), OK('market tide')]);
t('403 also points at the key', forbidden.text.includes(KEY_HINT), forbidden.text);

// mixed statuses across a total outage must list each one, not flatten them
const mixed = summary('SPY', [HTTP('flow', 401), HTTP('dark pool', 500), NET('IV'), HTTP('market tide', 401)]);
eq('mixed total outage → level fail', mixed.level, 'fail');
t('  does not flatten to one status', !/on all 4 routes/.test(mixed.text), mixed.text);
t('  lists the 500', /dark pool HTTP 500/.test(mixed.text), mixed.text);
t('  lists the network error', /IV network error/.test(mixed.text), mixed.text);
t('  lists the 401s', /flow HTTP 401/.test(mixed.text) && /market tide HTTP 401/.test(mixed.text), mixed.text);

// ── 6. a render crash is a failure, whatever the HTTP status said ────────
// The route returned 200 and the JSON parsed. The card still did not render,
// so the operator is not looking at the data. 200 is not the question.
const rend = summary('SPY', [RENDER('IV', 'x'), OK('flow'), OK('dark pool'), OK('market tide')]);
t('a render crash on a 200 is not ok', rend.level !== 'ok', rend.level);
eq('  level is partial', rend.level, 'partial');
t('  text does not start with ✓', !rend.text.startsWith('✓'), rend.text);
t('  a 200 that failed to render never reads HTTP 200', !/HTTP 200/.test(rend.text), rend.text);
t('  names IV as the broken part', /IV render error/.test(rend.text), rend.text);
t('  a render crash is not a key problem', !rend.text.includes(KEY_HINT), rend.text);

const rendAll = summary('SPY', NAMES.map(n => RENDER(n)));
eq('every part failing to render → level fail', rendAll.level, 'fail');
t('  still no check mark', !rendAll.text.startsWith('✓'), rendAll.text);

// ── 7. NEGATIVE: the check mark, over every combination ──────────────────
// This is the assertion the whole file exists for. Sixteen mixed states of
// four parts; '✓' must appear if and only if all four succeeded. One '✓' on a
// degraded feed here is the shipped defect.
let combosChecked = 0;
const badTick = [];
const badLevel = [];
for (let mask = 0; mask < 16; mask++) {
    const parts = NAMES.map((n, i) => (mask & (1 << i)) ? OK(n) : HTTP(n, 401));
    const nOk = parts.filter(p => p.ok).length;
    const s = summary('SPY', parts);
    const tick = s.text.startsWith('✓');
    const wantTick = nOk === 4;
    const wantLevel = nOk === 4 ? 'ok' : nOk === 0 ? 'fail' : 'partial';
    combosChecked += 1;
    if (tick !== wantTick) badTick.push(`mask ${mask} (${nOk}/4 ok): ${s.text}`);
    if (s.level !== wantLevel) badLevel.push(`mask ${mask} (${nOk}/4 ok): level ${s.level} != ${wantLevel}`);
}
eq('all 16 combinations exercised', combosChecked, 16);
t('✓ appears if and only if all four parts are ok', badTick.length === 0, badTick.join(' | '));
t('level tracks the count of healthy parts', badLevel.length === 0, badLevel.join(' | '));

// ── 8. the ticker is echoed, not hardcoded ───────────────────────────────
const nvdaOk = summary('NVDA', NAMES.map(OK));
const nvdaDead = summary('NVDA', NAMES.map(n => HTTP(n, 401)));
eq('success line carries the requested ticker', nvdaOk.text, '✓ Unusual Whales data loaded for NVDA');
t('failure line carries the requested ticker', /for NVDA/.test(nvdaDead.text), nvdaDead.text);

// ── 9. the colour map matches the level, and fail is never the ok colour ─
const COLORS = run('UW_STATUS_COLORS');
eq('ok keeps the existing magenta', COLORS.ok, '#E040FB');
eq('partial is the warn colour', COLORS.partial, 'var(--warn)');
eq('fail is the bear colour', COLORS.fail, 'var(--bear)');
t('a failed status is never painted as a success', COLORS.fail !== COLORS.ok, json(COLORS));

// ── 10. uwPartReason on its own ──────────────────────────────────────────
const reason = (p) => run(`uwPartReason(${json(p)})`);
eq('401 → HTTP 401', reason(HTTP('flow', 401)), 'HTTP 401');
eq('500 → HTTP 500', reason(HTTP('flow', 500)), 'HTTP 500');
eq('status 0 → network error', reason(NET('flow')), 'network error');
eq('render: prefix → render error', reason(RENDER('flow')), 'render error');

// ── report ──────────────────────────────────────────────────────────────
let failed = 0;
for (const [name, pass, detail] of results) {
    if (!pass) failed += 1;
    console.log((pass ? '  ok  ' : '✗ FAIL') + '  ' + name + (detail && !pass ? '\n          got: ' + detail : ''));
}
console.log('\n' + (failed ? '✗ ' + failed + ' failed / ' : '✓ ') + results.length + ' checks');
process.exit(failed ? 1 : 0);
