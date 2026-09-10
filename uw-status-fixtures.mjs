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
const reason = (p) => run(`uwPartReason(${json(p)})`);

const OK = (name) => ({ name, ok: true, status: 200 });
const HTTP = (name, status) => ({ name, ok: false, status, error: 'HTTP ' + status });
const NET = (name, msg) => ({ name, ok: false, status: 0, error: msg || 'Failed to fetch' });
const RENDER = (name, msg) => ({ name, ok: false, status: 200, error: 'render: ' + (msg || 'x') });
// HTTP 200 carrying something that is not JSON — a proxy error page, a
// truncated body. A server DID answer, so this is not status 0.
const PARSE = (name, msg) => ({ name, ok: false, status: 200, error: 'parse: ' + (msg || 'Unexpected token < in JSON at position 0') });

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

// ── 6b. a bad body is the backend's fault, not the network's ─────────────
// HTTP 200 with an HTML error page inside it. A server answered, so status 0
// would blame a network that worked fine; 'HTTP 200' would report a failure
// using the number that means success. Neither is true, neither is actionable.
const parse1 = summary('SPY', [PARSE('flow'), OK('dark pool'), OK('IV'), OK('market tide')]);
eq('a parse failure on a 200 → level partial', parse1.level, 'partial');
t('  reads bad response', /flow bad response/.test(parse1.text), parse1.text);
t('  never blames the network', !/network error/.test(parse1.text), parse1.text);
t('  never reports the failure as HTTP 200', !/HTTP 200/.test(parse1.text), parse1.text);
t('  a bad body is not a key problem', !parse1.text.includes(KEY_HINT), parse1.text);

const parseAll = summary('SPY', NAMES.map(n => PARSE(n)));
eq('every route returning a bad body → level fail', parseAll.level, 'fail');
eq('  text is the shipped bad-body line', parseAll.text,
   '✗ Unusual Whales unavailable for SPY — bad response on all 4 routes');
t('  never blames the network', !/network error/.test(parseAll.text), parseAll.text);

// The failure kinds have to stay tellable apart. If two of them collapse to one
// string the operator cannot tell whose bug it is: ours, the backend's, or the
// network's. Each one sends them somewhere different.
const KINDS = {
    parse: reason(PARSE('flow')),
    render: reason(RENDER('flow')),
    net: reason(NET('flow')),
    http: reason(HTTP('flow', 500)),
};
eq('a bad body reads bad response', KINDS.parse, 'bad response');
eq('a crashed card reads render error', KINDS.render, 'render error');
t('a bad body and a crashed card are different strings', KINDS.parse !== KINDS.render, json(KINDS));
t('a bad body and a dead socket are different strings', KINDS.parse !== KINDS.net, json(KINDS));
t('all four failure kinds are distinct', new Set(Object.values(KINDS)).size === 4, json(KINDS));

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
eq('401 → HTTP 401', reason(HTTP('flow', 401)), 'HTTP 401');
eq('500 → HTTP 500', reason(HTTP('flow', 500)), 'HTTP 500');
eq('status 0 → network error', reason(NET('flow')), 'network error');
eq('render: prefix → render error', reason(RENDER('flow')), 'render error');
eq('parse: prefix → bad response', reason(PARSE('flow')), 'bad response');

// ── 11. uwFetch: the boundary where the HTTP status is kept or lost ──────
// This is where the original defect lived. `.then(r => r.ok ? r.json() : null)`
// threw the status away before anything downstream could report it.
const fakeRes = (status, jsonImpl) => ({ ok: status >= 200 && status < 300, status, json: jsonImpl });
const withFetch = async (impl) => {
    const prev = sandbox.fetch;
    sandbox.fetch = impl;
    try { return await run(`uwFetch('/api/uw/test')`); }
    finally { sandbox.fetch = prev; }
};

const f200 = await withFetch(() => Promise.resolve(fakeRes(200, () => Promise.resolve({ data: [1] }))));
eq('200 with a good body → ok', f200.ok, true);
eq('  keeps the status', f200.status, 200);

const f401 = await withFetch(() => Promise.resolve(fakeRes(401, () => Promise.resolve({}))));
eq('401 → not ok', f401.ok, false);
eq('  the status survives instead of collapsing to null', f401.status, 401);
eq('  error names the status', f401.error, 'HTTP 401');

const fParse = await withFetch(() => Promise.resolve(fakeRes(200, () => Promise.reject(new Error('Unexpected token <')))));
eq('200 with an unparseable body → not ok', fParse.ok, false);
eq('  the server answered, so its status stands', fParse.status, 200);
t('  error carries the parse: prefix', /^parse: /.test(String(fParse.error)), json(fParse));
eq('  and reads bad response, not network error', reason(fParse), 'bad response');

const fNet = await withFetch(() => Promise.reject(new Error('Failed to fetch')));
eq('fetch() rejecting → not ok', fNet.ok, false);
eq('  status 0 is reserved for a request that reached no server', fNet.status, 0);
eq('  and reads network error', reason(fNet), 'network error');
t('uwFetch resolved in all four cases — it never rejects',
  [f200, f401, fParse, fNet].every(r => r && typeof r.ok === 'boolean'), '');

// ── 12. a throw outside every per-part try ───────────────────────────────
// The button re-enables in a `finally`. If the status line still reads
// 'Fetching...' at that moment, the UI claims work is in flight next to a live
// button — the same false claim as the old unconditional '✓', in other clothes.
//
// Drives the real fetchUWData() with uwRenderAll() replaced by a throw. Needs a
// document that returns the SAME element for an id twice, so the assertions can
// read what the handler wrote; the module-level stub hands out a fresh object
// per call, which would make every check below pass vacuously.
const els = {};
const realGetElementById = sandbox.document.getElementById;
sandbox.document.getElementById = (id) => (els[id] || (els[id] = stub()));
run('uwRenderAll = async function () { throw new Error("boom"); };');
await run('fetchUWData()');
sandbox.document.getElementById = realGetElementById;

const st = els.optionsStatus || { style: {} };
const bt = els.uwFetchBtn || {};
eq('a top-level throw sets the failure line', st.textContent, '✗ Unusual Whales render failed for SPY');
t('  the status never stays on Fetching', !/Fetching/.test(String(st.textContent)), String(st.textContent));
t('  and never claims loaded', !/loaded/.test(String(st.textContent)), String(st.textContent));
eq('  painted with the fail colour', st.style.color, COLORS.fail);
eq('  the button is reset in the finally', bt.textContent, '🐋 Whale Flow');
eq('  and re-enabled', bt.disabled, false);

// ── report ──────────────────────────────────────────────────────────────
let failed = 0;
for (const [name, pass, detail] of results) {
    if (!pass) failed += 1;
    console.log((pass ? '  ok  ' : '✗ FAIL') + '  ' + name + (detail && !pass ? '\n          got: ' + detail : ''));
}
console.log('\n' + (failed ? '✗ ' + failed + ' failed / ' : '✓ ') + results.length + ' checks');
process.exit(failed ? 1 : 0);
