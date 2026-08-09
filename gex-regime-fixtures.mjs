// gex-regime-fixtures.mjs — §2.3 regression fixtures for the gamma regime
// classifier (spec: apex-gex-regime-classifier v1.0).
//
//   node gex-regime-fixtures.mjs
//
// Extracts the GRX pure block from index.html by its marker lines (same
// pattern as trading-hashira's inline-parity.mjs) and runs grxSelfTest()
// under node — the identical fixtures run in-browser via ?grxtest=1.
//
// Fixture 1 is the board that motivated the build: the 08/08 board for the
// 08/10 chain (spot 773.38, walls 770/775, magnet 775, flip 784.02, dominant
// positive cluster 770-780). The OLD flip-only logic printed NEGATIVE GAMMA /
// trend extension on it; the classifier must print PIN, and the suite also
// asserts the flip-only label was wrong (regression proof, §7 checkbox 1).

import { readFileSync } from 'node:fs';

const src = readFileSync(new URL('./index.html', import.meta.url), 'utf8');

const m = src.match(/\/\/ ── GRX-PURE-BEGIN[\s\S]*?\/\/ ── GRX-PURE-END/);
if (!m) {
    console.error('✗ FAIL: GRX-PURE-BEGIN/END markers not found in index.html');
    process.exit(1);
}

const api = new Function(
    m[0] + '\nreturn { GRX_CONFIG, grxNormPdf, grxBsGamma, grxStrikesFromChain, grxClassify, grxSelfTest };'
)();

const st = api.grxSelfTest();
for (const r of st.results) {
    console.log((r.pass ? '  ✓ ' : '  ✗ ') + r.name + (r.detail ? ' — ' + r.detail : ''));
}

// Banner regression — the flip-only NEGATIVE GAMMA banner must be gone from
// renderGexLevels (§7: "Banner never says 'expect trend extension' while
// rule 1 conditions hold" — structurally guaranteed because the only banner
// carrying that text is regime-keyed, and PIN's copy contains no such claim).
const extra = [];
extra.push(['flip-only NEGATIVE GAMMA banner removed from renderGexLevels',
    !src.includes('expect vol expansion and trend extension')]);
extra.push(['PIN banner copy contains no trend-extension language',
    /TREND — DEALERS SHORT GAMMA/.test(src) && !/PIN — DEALERS LONG GAMMA[\s\S]{0,400}?trend extension/.test(src)]);
extra.push(['regime log uses its own storage key (nx_gammaRegime_log), TRADES key untouched',
    src.includes("'nx_gammaRegime_log'") &&
    !/nx_gammaRegime_log[\s\S]{0,120}apex_journal_trades|apex_journal_trades[\s\S]{0,120}nx_gammaRegime_log/.test(src)]);
extra.push(['slRunGate parity surface untouched (overlay pattern in use)',
    src.includes('grxApplyToGate(slRunGate(') && !/function slRunGate[\s\S]{0,8000}?grx/i.test(src.slice(src.indexOf('function slRunGate'), src.indexOf('function slRunGate') + 8200))]);

for (const [name, pass] of extra) console.log((pass ? '  ✓ ' : '  ✗ ') + name);

const ok = st.pass && extra.every(([, p]) => p);
console.log(ok ? '\nPASS — all fixtures green' : '\nFAIL');
process.exit(ok ? 0 : 1);
