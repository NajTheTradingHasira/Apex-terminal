// sl-dte-parity.mjs — the SL_DTE block must be byte-identical in both terminals.
//
//   node sl-dte-parity.mjs <apex/index.html> <hashira/index.html>
//
// Glue (slDteContext and the call sites) is terminal-specific and lives
// OUTSIDE the markers; everything between them is shared. A one-character
// drift fails here with the first differing line, instead of surfacing weeks
// later as two terminals that print different contracts for the same tape.

import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const [fa, fb] = process.argv.slice(2);
if (!fa || !fb) { console.error('usage: node sl-dte-parity.mjs <apex index.html> <hashira index.html>'); process.exit(2); }

function block(file) {
    const s = readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
    const begins = s.split('// ══ SL_DTE BEGIN').length - 1, ends = s.split('// ══ SL_DTE END').length - 1;
    if (begins !== 1 || ends !== 1) { console.error(`✗ FAIL: ${file} has ${begins} BEGIN / ${ends} END markers (need exactly 1 each)`); process.exit(1); }
    const a = s.indexOf('// ══ SL_DTE BEGIN'), b = s.indexOf('// ══ SL_DTE END');
    return s.slice(a, s.indexOf('\n', b));
}
const A = block(fa), B = block(fb);
const h = x => createHash('sha256').update(x).digest('hex').slice(0, 16);
if (A === B) {
    console.log(`✓ SL_DTE parity: identical (${A.split('\n').length} lines, sha256 ${h(A)})`);
    process.exit(0);
}
const la = A.split('\n'), lb = B.split('\n');
const i = la.findIndex((l, k) => l !== lb[k]);
console.error(`✗ FAIL: SL_DTE blocks differ (${h(A)} vs ${h(B)}). First difference at block line ${i + 1}:`);
console.error('  A: ' + (la[i] ?? '<end>'));
console.error('  B: ' + (lb[i] ?? '<end>'));
process.exit(1);
