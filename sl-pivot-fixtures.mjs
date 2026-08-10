// sl-pivot-fixtures.mjs — regression fixtures for `pivot`, the near-term
// structural level (spec: docs/structure/near-term-pivot-proposal.md in
// nexus-terminal).
//
//   node sl-pivot-fixtures.mjs
//
// Same shape as gex-regime-fixtures.mjs: evaluate the real inline script out of
// index.html under node and exercise the shipped functions, so the fixtures can
// never drift from what the browser runs.
//
// The motivating board is the week of 08/03: 749.44 / 776.85 / 748.80 / 773.26.
// Its LOW — 748.80 — is what the convention calls `support`, and it printed
// ABOVE `reclaim` 745.73. Under spy_weekly_structure.v1 that fails
// flip < support < reclaim, so the whole payload is rejected and the panel
// silently falls back to a triple anchored 2026-07-26. `pivot` gives that value
// a slot OUTSIDE the ordered triple.
//
// The properties under test, in order:
//   1. a pivot ABOVE reclaim is accepted (the case the field exists for)
//   2. anything unusable degrades to null and NEVER throws
//   3. absent renders nothing at all — no empty row, no placeholder
//   4. `origin` is escaped: it is API-controlled text going into innerHTML
//   5. per-level staleness chips, without touching block staleness

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

// `let SL_STRUCTURE` lives in the context's GLOBAL LEXICAL scope, not as a
// property of the sandbox object — only `var` and function declarations land
// there. Assigning sandbox.SL_STRUCTURE would create a shadow global that
// slRenderStructure never reads, and every render assertion would pass
// vacuously. Everything below goes through runInContext for that reason.
const run = (code) => vm.runInContext(code, sandbox);
const setStruct = (o) => run(`SL_STRUCTURE = Object.assign({}, SL_STRUCTURE, ${JSON.stringify(o)});`);
const setPivot = (p) => run(`SL_STRUCTURE = Object.assign({}, SL_STRUCTURE, { pivot: slPivot(${JSON.stringify(p)}) });`);
const paint = () => { bar.innerHTML = ''; run('slRenderStructure();'); return bar.innerHTML; };

const results = [];
const t = (name, pass, detail) => results.push([name, !!pass, detail]);

const slPivot = sandbox.slPivot;
t('slPivot() is exported from the inline script', typeof slPivot === 'function');

// ── 1. The case the field exists for ─────────────────────────────────────
const WEEK_0803 = { level: 748.80, kind: 'completed_week_low', asOf: '2026-08-07',
                    origin: '08/03 completed week low' };
const p = slPivot(WEEK_0803);
t('accepts a pivot ABOVE reclaim', p && p.level === 748.80);
t('  748.80 > reclaim 745.73 — no ordering check, by design', p && p.level > 745.73);
t('  kind / asOf / origin preserved',
  p && p.kind === 'completed_week_low' && p.asOf === '2026-08-07'
    && p.origin === '08/03 completed week low');

// ── 2. Unusable input degrades to null and never throws ──────────────────
for (const [label, bad] of [
    ['null', null], ['undefined', undefined], ['a string', '748.80'],
    ['a bare number', 748.8], ['an array', [748.8]],
    ['no level', { kind: 'completed_week_low' }], ['level 0', { level: 0 }],
    ['level negative', { level: -5 }], ['level NaN', { level: NaN }],
    ['level Infinity', { level: Infinity }], ['level unparseable', { level: 'seven' }],
    ['level null', { level: null }],
]) {
    let out, threw = false;
    try { out = slPivot(bad); } catch { threw = true; }
    t(`unusable → null, never throws: ${label}`, !threw && out === null);
}

// `??` and not `||`: a level of 0 must be rejected by the RANGE test, not read
// as "absent". Irrelevant for prices, load-bearing once orHigh/orLow arrive.
t('level 0 is rejected by the range test, not collapsed to absent', slPivot({ level: 0 }) === null);
t('numeric string coerces, matching the backend float()', slPivot({ level: '748.80' }).level === 748.80);
t('malformed asOf drops the date, keeps the level', (() => {
    const r = slPivot({ level: 748.8, asOf: 12345 });
    return r && r.level === 748.8 && r.asOf === null;
})());

// ── 3. Render ────────────────────────────────────────────────────────────
t('baked SL_STRUCTURE carries pivot:null — the fallback invents nothing',
  run('"pivot" in SL_STRUCTURE && SL_STRUCTURE.pivot === null'));

setStruct({ asOf: '2026-08-07', reclaim: 745.73, support: 729.1, flip: 725.86, staleDays: 10 });
setPivot(WEEK_0803);
const withPivot = paint();
t('renders a PIVOT row when present', withPivot.includes('PIVOT'));
t('  renders the level', withPivot.includes('748.80'));
t('  renders the origin as the row tail', withPivot.includes('08/03 completed week low'));
t('  the ordered triple still renders', ['RECLAIM', 'SUPPORT', 'FLIP'].every(k => withPivot.includes(k)));
t('  PIVOT comes last, after FLIP', withPivot.indexOf('PIVOT') > withPivot.indexOf('FLIP'));
t('  a hairline separates it from the triple', /border-top:1px solid var\(--border\)/.test(withPivot));

run('SL_STRUCTURE = Object.assign({}, SL_STRUCTURE, { pivot: null });');
const noPivot = paint();
t('renders NOTHING when absent — no empty row', !noPivot.includes('PIVOT'));
t('  and no em-dash placeholder', !/PIVOT[\s\S]{0,80}—/.test(noPivot));
t('  the triple is unaffected', ['RECLAIM', 'SUPPORT', 'FLIP'].every(k => noPivot.includes(k)));

// ── 4. `origin` is API-controlled text reaching innerHTML ────────────────
setPivot({ level: 748.8, origin: '<img src=x onerror=alert(1)>' });
const xss = paint();
t('origin is HTML-escaped', !xss.includes('<img src=x') && xss.includes('&lt;img'));

// ── 5. Per-level staleness, without touching block staleness ─────────────
const daysAgo = (n) => new Date(Date.now() - n * 86400000).toISOString().slice(0, 10);
setStruct({ staleDays: 10 });
setPivot({ level: 748.8, asOf: daysAgo(40), origin: 'stale pivot' });
t('a stale pivot shows its own age chip', /PIVOT \d+D OLD/.test(paint()));
setPivot({ level: 748.8, asOf: daysAgo(2), origin: 'fresh pivot' });
t('a fresh pivot shows no age chip', !/PIVOT \d+D OLD/.test(paint()));

// ── Report ───────────────────────────────────────────────────────────────
for (const [name, pass, detail] of results) {
    console.log((pass ? '  ✓ ' : '  ✗ ') + name + (detail ? '  — ' + detail : ''));
}
const ok = results.every(([, pass]) => pass);
const failed = results.filter(([, pass]) => !pass).length;
console.log(ok ? `\nPASS — ${results.length} fixtures green`
               : `\nFAIL — ${failed} of ${results.length}`);
process.exit(ok ? 0 : 1);
