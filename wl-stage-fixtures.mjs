// wl-stage-fixtures.mjs — the watchlist stage boundary.
//
//   node wl-stage-fixtures.mjs   (from the repo root)
//
// Same shape as sl-struct-keys.mjs: evaluate the real inline script out of
// index.html under node and drive the SHIPPED stageInfo() / wlStageOptionsHtml()
// / buildRow(), so this can never drift from what the browser runs.
//
// ── Why this boundary needs a guard ─────────────────────────────────────
//
// The stage field round trips through the Postgres backend shared with
// trading-hashira. Three apps write it in two dialects: the legacy APEX code
// ('s2') and the canonical label ('Stage 2A'). Every regression this boundary
// has had was the same regression: a value APEX did not recognise was rewritten
// into one it did, and the row then displayed a stage nobody assigned.
//
// That is the `degraded` failure mode again — a fallback rendered as if it were
// vouched for. A stage badge is a directional claim. Printing 'Stage 1' over an
// unparsed value is strictly worse than printing nothing, because the operator
// cannot tell the two apart, and neither can the next reader of the database.
//
// So the assertions below are mostly negative. They pin what stageInfo must
// REFUSE to claim: no code off free text, no invented label, no silent
// collapse of a substage, and no stray 1-4 digit anywhere in a sentence
// promoted to a stage.

import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const src = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const m = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/i.exec(src);
if (!m) {
    console.error('✗ FAIL: no inline <script> block found in index.html');
    process.exit(1);
}

// ── Just enough DOM for buildRow() to assemble a row ─────────────────────
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

// `const WL_STAGES` lives in the context's GLOBAL LEXICAL scope, not as a
// property of the sandbox object, so everything below goes through
// runInContext. Reading sandbox.WL_STAGES would read undefined and every
// assertion would pass vacuously.
const run = (code) => vm.runInContext(code, sandbox);
const call = (fn, arg) => run(`${fn}(${JSON.stringify(arg)})`);
const json = (v) => JSON.stringify(v);

const results = [];
const t = (name, pass, detail) => results.push([name, !!pass, detail]);
const eq = (name, got, want) => t(name, got === want, `${json(got)} !== ${json(want)}`);

// ── 0. the registry is the single source of truth ────────────────────────
const STAGES = run('WL_STAGES');
eq('WL_STAGES holds 12 entries', STAGES.length, 12);
t('WL_STAGES covers all four stages plus A/B substages',
  json(STAGES) === json([
      'Stage 1', 'Stage 1A', 'Stage 1B', 'Stage 2', 'Stage 2A', 'Stage 2B',
      'Stage 3', 'Stage 3A', 'Stage 3B', 'Stage 4', 'Stage 4A', 'Stage 4B',
  ]), json(STAGES));

// Every registry value must survive a parse unchanged. If this fails, a value
// the dropdown can produce is one the parser cannot read back — the exact loop
// that loses a substage on edit.
let roundTripped = true;
for (const v of STAGES) {
    const back = call('stageInfo', v);
    if (back.label !== v || back.known !== true) { roundTripped = false; break; }
}
t('every registry value round trips through stageInfo', roundTripped);

// ── 1. accepted dialects ─────────────────────────────────────────────────
// Legacy APEX codes, canonical labels, bare numbers, and the descriptor form
// that older rows and the two sibling terminals still write.
for (const [input, want] of [
    ['s1', 'Stage 1'], ['s2', 'Stage 2'], ['s3', 'Stage 3'], ['s4', 'Stage 4'],
    ['s2a', 'Stage 2A'], ['S1B', 'Stage 1B'],
    ['Stage 2A', 'Stage 2A'], ['Stage 3B', 'Stage 3B'],
    ['2A', 'Stage 2A'], ['2', 'Stage 2'], ['stage2', 'Stage 2'],
    ['Early Stage 2', 'Stage 2'],
    ['Stage 2A — Breakout', 'Stage 2A'],
    ['Stage 1 — Basing', 'Stage 1'],
    ['Stage 4 Decline', 'Stage 4'],
]) {
    const got = call('stageInfo', input);
    eq(`parses ${json(input)}`, got.label, want);
    eq(`  claims a code for ${json(input)}`, got.code, 's' + want.match(/([1-4])/)[1]);
}

// ── 2. what stageInfo must REFUSE to claim ───────────────────────────────
// An unrecognised value keeps its own text and gets no code. It must not be
// relabelled, because this field is written back to the shared backend.
for (const raw of [
    'Accumulation', 'pre-breakout', 'WATCH', 'Stage V', 'transitional',
    'Top 10 holding',      // a stray 1 inside free text is not a stage claim
    'Base 3 weeks out',    // nor is a stray 3
]) {
    const got = call('stageInfo', raw);
    eq(`preserves unrecognised ${json(raw)}`, got.label, raw);
    eq(`  claims no code for ${json(raw)}`, got.code, '');
    eq(`  flags it unknown: ${json(raw)}`, got.known, false);
}

// An absent stage claims nothing at all — not a code, not a label.
for (const empty of ['', '   ', null, undefined]) {
    const got = call('stageInfo', empty === undefined ? null : empty);
    eq(`absent input ${json(empty)} yields no label`, got.label, '');
    eq(`absent input ${json(empty)} yields no code`, got.code, '');
}

// stageLabel is a DISPLAY helper. Its em dash is for the screen only and must
// never be the value written back to storage.
eq('stageLabel renders absent as an em dash', call('stageLabel', ''), '—');
eq('stageLabel passes unrecognised text through',
   call('stageLabel', 'Accumulation'), 'Accumulation');

// ── 3. the selector preserves what it cannot name ────────────────────────
const known = call('wlStageOptionsHtml', 's3a');
eq('known value yields exactly the registry options',
   (known.match(/<option/g) || []).length, 12);
t('known value is the selected option', /<option value="Stage 3A" selected>/.test(known), known);

const unknown = call('wlStageOptionsHtml', 'Accumulation');
eq('unrecognised value adds one option rather than dropping it',
   (unknown.match(/<option/g) || []).length, 13);
t('unrecognised value stays selected, so it round trips unchanged',
  /<option value="Accumulation" selected>/.test(unknown), unknown);
t('unrecognised value is labelled as such on screen',
  /Accumulation — unrecognized/.test(unknown), unknown);
eq('exactly one option is ever selected',
   (unknown.match(/ selected>/g) || []).length, 1);

const blank = call('wlStageOptionsHtml', '');
eq('absent value adds no phantom option', (blank.match(/<option/g) || []).length, 12);

// The preserved value goes into an attribute, so a quote must not escape it.
const hostile = call('wlStageOptionsHtml', 'x" onerror="boom');
t('option value is attribute-escaped', !/onerror="boom"/.test(hostile), hostile);
t('  escaped form is present instead', /&quot;/.test(hostile), hostile);

// ── 4. the rendered badge never overstates ───────────────────────────────
// buildRow is the only place a stage reaches the screen. Drive the shipped
// function so the badge class and text are asserted as shipped, not as
// reimplemented here.
const badge = (stage) => {
    const tr = run(`buildRow(${json({ sym: 'TEST', stage })}, 'daily')`);
    return { html: tr.innerHTML, code: tr.dataset.stage, label: tr.dataset.stageLabel };
};

const b2a = badge('Stage 2A');
t('substage survives buildRow', /<span class="wl-stage s2">Stage 2A<\/span>/.test(b2a.html), b2a.html);
eq('  dataset keeps the derived code', b2a.code, 's2');
eq('  dataset keeps the canonical label', b2a.label, 'Stage 2A');

const bUnk = badge('Accumulation');
t('unrecognised badge is not coloured as a verified stage',
  /<span class="wl-stage unknown">Accumulation<\/span>/.test(bUnk.html), bUnk.html);
eq('  no stage code is stored', bUnk.code, '');
t('  the badge does not read Stage 1', !/Stage 1/.test(bUnk.html), bUnk.html);

const bNone = badge('');
t('absent stage renders an em dash, not Stage 1',
  /<span class="wl-stage unknown">—<\/span>/.test(bNone.html), bNone.html);
eq('  no stage code is stored', bNone.code, '');

// ── report ──────────────────────────────────────────────────────────────
let failed = 0;
for (const [name, pass, detail] of results) {
    if (!pass) failed += 1;
    console.log((pass ? '  ok  ' : '✗ FAIL') + '  ' + name + (detail && !pass ? '\n          got: ' + detail : ''));
}
console.log('\n' + (failed ? '✗ ' + failed + ' failed / ' : '✓ ') + results.length + ' checks');
process.exit(failed ? 1 : 0);
