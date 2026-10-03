// sl-dte-fixtures.mjs — regression fixtures for the Day Swing Planner (SL_DTE).
//
//   node sl-dte-fixtures.mjs                 # tests the block inside ./index.html
//   node sl-dte-fixtures.mjs path/to/file    # any file that contains the markers
//
// Extracts the code between `SL_DTE BEGIN` and `SL_DTE END` from the file and
// evaluates it under node, so the fixtures exercise exactly what the browser
// runs. Every guard is tested in BOTH directions: it fires when it should and
// stays quiet when it should not.

import { readFileSync, existsSync } from 'node:fs';
import vm from 'node:vm';

// Default: ./index.html (APEX keeps fixtures at the root) or ../index.html
// (hashira keeps them in spy-logic/).
const file = process.argv[2] || [new URL('./index.html', import.meta.url), new URL('../index.html', import.meta.url)].find(u => existsSync(u));
if (!file) { console.error('✗ FAIL: no index.html next to or above this file'); process.exit(1); }
const src = readFileSync(file, 'utf8');
const a = src.indexOf('// ══ SL_DTE BEGIN'), b = src.indexOf('// ══ SL_DTE END');
if (a < 0 || b < 0 || b < a) { console.error('✗ FAIL: SL_DTE markers not found in ' + file); process.exit(1); }
const block = src.slice(a, src.indexOf('\n', b));

const sandbox = {
    console: { log() {}, warn() {}, error() {} },
    Date, Math, JSON, Number, String, Array, Object, Map, Intl, RegExp, Error, Promise, isFinite, encodeURIComponent,
    fetch: () => Promise.reject(new Error('no network in fixtures')),
};
vm.createContext(sandbox);
// const/function declarations stay in the script scope; export what we test.
vm.runInContext(block + `
;globalThis.T = { SL_DTE, slDteNcdf, slDteBS, slDteIv, slDteEtEpoch, slDteEtYmd, slDteSession,
  slDtePickExpiries, slDteEval, slDtePickStyles, slDteBuild, slDteHtml, slDteLots, slDteParitySpot,
  slDteNormChain, slDteRowsExpiry, slDteEsc, slDteEligible, slDteFlags, slDteVarDays, slDteImpliedVol, slDteUwIv, slDteSourceInfo };`, sandbox);
const T = sandbox.T;

let pass = 0, fail = 0;
const ok = (cond, name, detail) => {
    if (cond) { pass++; } else { fail++; console.log('✗ ' + name + (detail !== undefined ? '  → ' + JSON.stringify(detail) : '')); }
};
const near = (x, y, tol) => typeof x === 'number' && Math.abs(x - y) <= tol;
const iso = ms => new Date(ms).toISOString();

// ── 1. math ─────────────────────────────────────────────────────────────
ok(near(T.slDteNcdf(0), 0.5, 1e-7), 'N(0) = 0.5');
ok(near(T.slDteNcdf(1.959964), 0.975, 2e-7), 'N(1.96) = 0.975');
ok(near(T.slDteNcdf(-1.959964), 0.025, 2e-7), 'N(-1.96) = 0.025');
{
    const c = T.slDteBS('C', 100, 100, 1, 0.2, 0), p = T.slDteBS('P', 100, 100, 1, 0.2, 0);
    ok(near(c.price, 7.9656, 2e-4), 'BS call textbook value 7.9656', c);
    ok(near(p.price, 7.9656, 2e-4), 'BS put = call at r=0, S=K', p);
    ok(near(c.delta, 0.5398, 2e-4) && near(p.delta, -0.4602, 2e-4), 'BS deltas 0.5398 / -0.4602');
    const c2 = T.slDteBS('C', 100, 95, 0.5, 0.25, 0.04), p2 = T.slDteBS('P', 100, 95, 0.5, 0.25, 0.04);
    ok(near(c2.price - p2.price, 100 - 95 * Math.exp(-0.04 * 0.5), 1e-6), 'put-call parity holds');
    ok(T.slDteBS('C', 100, 90, 0, 0.2, 0).price === 10 && T.slDteBS('P', 100, 90, 0, 0.2, 0).price === 0, 'expiry → intrinsic');
    ok(T.slDteBS('C', 0, 90, 1, 0.2, 0) === null && T.slDteBS('C', 100, 90, 1, 0, 0) === null, 'bad input → null, no throw');
}
ok(near(T.slDteIv(15.3), 0.153, 1e-12) && near(T.slDteIv(0.153), 0.153, 1e-12), 'IV percent and decimal both read as 0.153');
ok(T.slDteIv(0) === null && T.slDteIv(-4) === null && T.slDteIv('x') === null && T.slDteIv(null) === null, 'unusable IV → null');

// ── 2. Eastern clock (DST both sides) ──────────────────────────────────
ok(iso(T.slDteEtEpoch('2026-10-05', 9, 30)) === '2026-10-05T13:30:00.000Z', 'EDT 09:30 = 13:30Z');
ok(iso(T.slDteEtEpoch('2026-12-07', 9, 30)) === '2026-12-07T14:30:00.000Z', 'EST 09:30 = 14:30Z');
ok(iso(T.slDteEtEpoch('2026-11-02', 16, 0)) === '2026-11-02T21:00:00.000Z', 'day after DST ends: 16:00 = 21:00Z');
ok(T.slDteEtYmd(Date.parse('2026-10-02T03:30:00Z')) === '2026-10-01', '23:30 ET still the previous ET date');


// ── 3. sessions and expiry labels ──────────────────────────────────────
const EXPS = ['2026-09-30', '2026-10-01', '2026-10-02', '2026-10-05', '2026-10-06', '2026-10-07', '2026-10-08', '2026-10-09',
              '2026-11-24', '2026-11-25', '2026-11-27', '2026-11-30', '2026-12-01'];   // no 11-26 (Thanksgiving)
const et = (ymd, h, m) => T.slDteEtEpoch(ymd, h, m);

// ── 2b. variance clock + implied vol ───────────────────────────────────
{
    const D = ['2026-09-30', '2026-10-01', '2026-10-02', '2026-10-05', '2026-11-25', '2026-11-27'];
    const V = (a, b) => T.slDteVarDays(a, b, D);
    ok(near(V(et('2026-09-30', 11, 0), et('2026-09-30', 16, 0)), 300 / 390, 1e-9), 'clock: 11:00→16:00 = 300/390 day');
    ok(near(V(et('2026-09-30', 16, 0), et('2026-10-01', 9, 30)), 0.25, 1e-9), 'clock: weeknight gap = 0.25 day');
    ok(near(V(et('2026-09-30', 9, 30), et('2026-10-01', 9, 30)), 1.25, 1e-9), 'clock: open→next open = 1.25 days');
    ok(near(V(et('2026-10-02', 16, 0), et('2026-10-05', 9, 30)), 0.35, 1e-9), 'clock: weekend gap = 0.25 + 2 × 0.05');
    ok(near(V(et('2026-11-25', 16, 0), et('2026-11-27', 9, 30)), 0.30, 1e-9), 'clock: holiday gap = 0.25 + 0.05');
    ok(near(V(et('2026-09-30', 20, 0), et('2026-10-01', 9, 30)), 0.25 * (13.5 / 17.5), 1e-9), 'clock: part of a gap counts pro rata');
    ok(V(et('2026-10-01', 9, 30), et('2026-09-30', 9, 30)) === 0, 'clock: reversed interval = 0');
    {
        // the clock-implied vol must reprice the entry quote exactly
        const row = { strike: 768, bid: 3.10, ask: 3.20, impliedVolatility: 14, openInterest: 9000, contractSymbol: 'SPY261002C00768000' };
        const clk = { now: 1.769, same: 1.0, over: 0.75, calYears: 1.2 / 365 };
        const e = T.slDteEval(row, 'C', 765, clk);
        ok(e && near(T.slDteBS('C', 765, 768, clk.now / 252, e.ivClock, 0.04).price, 3.15, 1e-4), 'model reprices the entry mid exactly', e && e.ivClock);
        ok(e && e.decayOver > e.decaySame && e.decaySame > 0, 'more clock time elapsed → more decay', e && [e.decaySame, e.decayOver]);
        const bad = T.slDteEval({ ...row, bid: 0.01, ask: 0.02 }, 'C', 800, clk);   // quote below intrinsic → IV fallback
        ok(bad && near(bad.ivClock, 0.14 * Math.sqrt(clk.calYears / (clk.now / 252)), 1e-9), 'unpriceable quote → chain IV rescaled to the clock', bad && bad.ivClock);
    }
    const px = T.slDteBS('C', 765, 768, 2 / 252, 0.18, 0.04).price;
    ok(near(T.slDteImpliedVol('C', 765, 768, 2 / 252, px, 0.04), 0.18, 1e-6), 'implied vol round trip 0.18');
    ok(T.slDteImpliedVol('C', 765, 700, 2 / 252, 60, 0.04) === null, 'price below intrinsic → null');
    ok(T.slDteImpliedVol('C', 765, 768, 0, 1, 0.04) === null, 'T = 0 → null');
}
{
    const s = T.slDteSession(et('2026-09-30', 11, 0), EXPS);
    ok(s.live && s.date === '2026-09-30', 'Wed 11:00 → live session today', s);
    const p = T.slDtePickExpiries(EXPS, s.date, 3);
    ok(p.map(x => x.expiry + ':' + x.dte).join(',') === '2026-10-01:1,2026-10-02:2,2026-10-05:3', '1/2/3 DTE from Wednesday', p);
    ok(p[2].calDays === 5, 'Mon expiry from Wed = 5 calendar days', p[2]);
}
{
    const s = T.slDteSession(et('2026-10-02', 11, 0), EXPS);
    const p = T.slDtePickExpiries(EXPS, s.date, 3);
    ok(p[0].expiry === '2026-10-05' && p[0].dte === 1 && p[0].calDays === 3, 'Friday: 1DTE is Monday, 3 calendar days', p[0]);
}
{
    const s = T.slDteSession(et('2026-10-03', 12, 0), EXPS);     // Saturday
    ok(!s.live && s.date === '2026-10-05' && s.t0 === et('2026-10-05', 9, 30), 'Saturday → plan for Monday open', s);
    const s2 = T.slDteSession(et('2026-09-30', 17, 0), EXPS);   // after the close
    ok(!s2.live && s2.date === '2026-10-01', 'Wed 17:00 → plan for Thursday', s2);
    const s3 = T.slDteSession(et('2026-09-30', 8, 0), EXPS);    // pre-market
    ok(!s3.live && s3.date === '2026-09-30', 'Wed 08:00 → plan for today\'s open', s3);
    const s4 = T.slDteSession(et('2026-11-25', 11, 0), EXPS);   // day before Thanksgiving
    const p4 = T.slDtePickExpiries(EXPS, s4.date, 1);
    ok(p4[0].expiry === '2026-11-27' && p4[0].calDays === 2, 'holiday skipped via the expiry list', p4[0]);
    ok(T.slDteSession(et('2026-12-30', 11, 0), EXPS) === null, 'no upcoming session in list → null, no throw');
}
{
    const s = T.slDteSession(et('2026-10-02', 11, 0), null);    // no list: weekday fallback
    const p = T.slDtePickExpiries(null, s.date, 3);
    ok(s.live && p.map(x => x.expiry).join(',') === '2026-10-05,2026-10-06,2026-10-07', 'weekday fallback skips the weekend', p);
}

// ── 4. synthetic chain priced by BS: the picker must recover the deltas ─
const SPOT = 765;
function chain(expiry, nowMs, opts = {}) {
    const tExp = T.slDteEtEpoch(expiry, 16, 0), Tyr = (tExp - nowMs) / (365 * 24 * 3600 * 1000);
    const yymmdd = expiry.slice(2).replace(/-/g, '');
    const mk = cp => {
        const rows = [];
        for (let k = 735; k <= 795; k++) {
            const iv = 0.14 + (cp === 'P' ? (SPOT - k) * 0.0015 : 0);
            const bs = T.slDteBS(cp, SPOT, k, Tyr, Math.max(0.08, iv), 0.04);
            const mid = Math.max(0.01, bs.price), half = opts.spread != null ? opts.spread / 2 : Math.max(0.01, mid * 0.01);
            const row = { strike: k, bid: +(mid - half).toFixed(4), ask: +(mid + half).toFixed(4), lastPrice: mid,
                        impliedVolatility: Math.max(0.08, iv) * 100, openInterest: opts.oi != null ? opts.oi : 5000, volume: 100,
                        contractSymbol: 'SPY' + (opts.symExp || yymmdd) + cp + String(k * 1000).padStart(8, '0') };
            if (opts.deltaShift != null) {   // vendor delta = model delta of the strike `deltaShift` below
                const d = T.slDteBS(cp, SPOT, k - opts.deltaShift, Tyr, Math.max(0.08, iv), 0.04).delta;
                row.delta = opts.putDeltaPositive ? Math.abs(d) : d;
            }
            if (opts.badDelta) row.delta = 1.7;
            if (opts.theta != null) row.theta = opts.theta;
            rows.push(row);
        }
        return rows;
    };
    return { source: opts.source || 'test', _route: opts.route, spot: opts.noSpot ? undefined : SPOT, expirations: EXPS, chain: { calls: mk('C'), puts: mk('P') } };
}
const NOW = et('2026-09-30', 11, 0);
const chains = { '2026-10-01': chain('2026-10-01', NOW), '2026-10-02': chain('2026-10-02', NOW), '2026-10-05': chain('2026-10-05', NOW) };
const base = { nowMs: NOW, expirations: EXPS, chains, risk: 150 };
{
    const m = T.slDteBuild({ ...base, ctx: { dir: 'LONG', gate: 'GO', spot: SPOT, ivp: 13, hviv: 'hvgtiv' } });
    ok(!m.error && m.side === 'C' && m.sideSource === 'read', 'LONG read → calls', m.error);
    const e1 = m.expiries[0];
    const d = Object.fromEntries(e1.cands.map(c => [c.style, Math.abs(c.delta)]));
    ok(near(d.ITM, 0.65, 0.05) && near(d.ATM, 0.50, 0.05) && near(d.OTM, 0.30, 0.05), 'picker hits 0.65 / 0.50 / 0.30', d);
    const s = e1.cands.map(c => c.strike);
    ok(s[0] < s[1] && s[1] < s[2], 'calls: ITM strike < ATM < OTM', s);
    const atm = m.expiries.map(e => e.cands.find(c => c.style === 'ATM'));
    ok(atm.every(c => c.costOver > c.costSame), 'overnight costs more points than same-day (longer hold)', atm.map(c => [c.costSame, c.costOver]));
    ok(atm[0].decayOverPct > atm[1].decayOverPct && atm[1].decayOverPct > atm[2].decayOverPct, 'overnight decay %: 1DTE > 2DTE > 3DTE', atm.map(c => c.decayOverPct));
    ok(m.bestSame && m.bestOver && m.bestSame.style !== 'OTM' && m.bestOver.style !== 'OTM', 'OTM is never recommended', [m.bestSame && m.bestSame.style, m.bestOver && m.bestOver.style]);
    ok(!m.weekendOver && m.overCal === 1, 'Wednesday overnight is one night — no weekend flag', m.overCal);
    ok(m.expiries.every(e => e.cands.every(c => !c.flags.some(f => f.startsWith('overnight spans')))), 'no overnight-span flag on a weekday');
    ok(near(m.expiries[0].move1d, SPOT * 0.14 * Math.sqrt(1 / 252), 0.4), '1-day implied move from ATM IV', m.expiries[0].move1d);
    const html = T.slDteHtml(m);
    ok(html.includes('Best overnight hold') && html.includes('Best same-day hold') && !html.includes('NO-GO — plan only'), 'render: both best cards, no NO-GO banner');
    ok(html.includes('IVP 13%') && html.includes('HV &gt; IV') , 'render: IVP and HV/IV are flagged', null);
}
{
    const m = T.slDteBuild({ ...base, ctx: { dir: 'SHORT', gate: 'NO-GO', spot: SPOT } });
    ok(m.side === 'P', 'SHORT read → puts');
    const s = m.expiries[0].cands.map(c => c.strike);
    ok(s[0] > s[1] && s[1] > s[2], 'puts: ITM strike > ATM > OTM', s);
    ok(m.expiries[0].cands.every(c => c.delta < 0), 'put deltas negative');
    ok(T.slDteHtml(m).includes('NO-GO — plan only'), 'NO-GO banner shows when the gate is NO-GO');
    const n = T.slDteBuild({ ...base, ctx: { dir: 'NEUTRAL', spot: SPOT } });
    ok(n.side === 'C' && n.sideSource === 'default', 'NEUTRAL → default side, labelled as default');
    const man = T.slDteBuild({ ...base, side: 'P', ctx: { dir: 'LONG', spot: SPOT } });
    ok(man.side === 'P' && man.sideSource === 'manual', 'manual override beats the read');
}
// Friday: overnight spans the weekend → flagged, both directions covered above.
{
    const FRI = et('2026-10-02', 11, 0);
    const fc = { '2026-10-05': chain('2026-10-05', FRI), '2026-10-06': chain('2026-10-06', FRI), '2026-10-07': chain('2026-10-07', FRI) };
    const m = T.slDteBuild({ nowMs: FRI, expirations: EXPS, chains: fc, risk: 150, ctx: { dir: 'LONG', spot: SPOT } });
    ok(m.weekendOver && m.overCal === 3, 'Friday overnight spans 3 days', m.overCal);
    ok(m.expiries[0].cands.some(c => c.flags.includes('overnight spans 3 days')), 'rows carry the 3-day flag');
    ok(T.slDteHtml(m).includes('SOP: no weekend/holiday long premium'), 'render: weekend SOP warning on the overnight card');
}
// clock-consistent repricing + overnight decay level, Wednesday vs Friday (both directions)
{
    const m = T.slDteBuild({ ...base, ctx: { dir: 'LONG', spot: SPOT } });
    const atm1 = m.expiries[0].cands.find(c => c.style === 'ATM');
    ok(atm1.ivClock > 0 && atm1.decayOverPct > 25 && atm1.decayOverPct < 35, 'Wed 1DTE ATM overnight decay ≈ 30% on the clock (not ~51% calendar)', atm1.decayOverPct);
    ok(atm1.decaySamePct > 15 && atm1.decaySamePct < atm1.decayOverPct, 'same-day decay below overnight decay', [atm1.decaySamePct, atm1.decayOverPct]);
    const FRI = et('2026-10-02', 11, 0);
    const fc = { '2026-10-05': chain('2026-10-05', FRI), '2026-10-06': chain('2026-10-06', FRI), '2026-10-07': chain('2026-10-07', FRI) };
    const f = T.slDteBuild({ nowMs: FRI, expirations: EXPS, chains: fc, risk: 150, ctx: { dir: 'LONG', spot: SPOT } });
    const fatm = f.expiries[0].cands.find(c => c.style === 'ATM');
    ok(fatm.decayOverPct > atm1.decayOverPct, 'Friday 1DTE overnight decays more than Wednesday 1DTE (weekend gap)', [fatm.decayOverPct, atm1.decayOverPct]);
    ok(fatm.decayOverPct < atm1.decayOverPct + 5, 'but a weekend is not three trading days of decay', [fatm.decayOverPct, atm1.decayOverPct]);
}
// ── 5. liquidity + premium gates (both directions) ─────────────────────
{
    const wide = { ...chains, '2026-10-01': chain('2026-10-01', NOW, { spread: 0.50 }) };
    const m = T.slDteBuild({ ...base, chains: wide, ctx: { dir: 'LONG', spot: SPOT } });
    ok(m.expiries[0].cands.every(c => !c.eligible && c.flags.some(f => f.startsWith('wide spread'))), 'wide spread → flagged and ineligible');
    ok(m.expiries[1].cands.filter(c => c.style !== 'OTM').every(c => c.eligible), 'normal spread on 2DTE stays eligible');
    ok(m.bestOver && m.bestOver.expiry !== '2026-10-01', 'best skips the illiquid expiry', m.bestOver && m.bestOver.expiry);
    const thin = { ...chains, '2026-10-02': chain('2026-10-02', NOW, { oi: 40 }) };
    const t = T.slDteBuild({ ...base, chains: thin, ctx: { dir: 'LONG', spot: SPOT } });
    ok(t.expiries[1].cands.every(c => !c.eligible && c.flags.includes('thin OI 40')), 'thin OI → flagged and ineligible');
}
{
    const good = { delta: 0.52, spreadPct: 3, oi: 5000, mid: 0.80 };
    ok(T.slDteEligible(good) === true, 'eligible: liquid, delta 0.52, $0.80');
    ok(T.slDteEligible({ ...good, mid: 0.25 }) === false, 'cheap premium $0.25 → not eligible');
    ok(T.slDteEligible({ ...good, delta: -0.40 }) === false && T.slDteEligible({ ...good, delta: -0.52 }) === true, 'delta gate uses |delta| for puts');
    ok(T.slDteEligible({ ...good, spreadPct: null }) === false, 'unknown spread → not eligible');
    ok(T.slDteFlags({ ...good, mid: 0.25 }, {}).some(f => f.startsWith('cheap premium')) && !T.slDteFlags(good, {}).length, 'cheap flag fires only on cheap premium');
}
ok(T.slDteLots(0.5, 150) === 3 && T.slDteLots(0.6, 150) === 2 && T.slDteLots(2.0, 150) === 0 && T.slDteLots(0.1, 5000) === 3, 'lots: premium-as-stop with 3-lot cap');
ok(T.slDteLots(null, 150) === 0 && T.slDteLots(1, 0) === 0, 'lots: bad input → 0');

// ── 6. data faults never throw ─────────────────────────────────────────
{
    const empty = T.slDteBuild({ nowMs: NOW, expirations: EXPS, chains: {}, risk: 150, ctx: { dir: 'LONG', spot: SPOT } });
    ok(!empty.error && empty.expiries.every(e => e.cands.length === 0 && e.error === 'chain not loaded'), 'missing chains → "chain not loaded", no throw');
    const nospot = T.slDteBuild({ nowMs: NOW, expirations: EXPS, chains: {}, risk: 150, ctx: { dir: 'LONG' } });
    ok(typeof nospot.error === 'string', 'no spot anywhere → error message, no throw');
    const junk = { '2026-10-01': { chain: { calls: [null, 'x', { strike: 'abc' }, { strike: 765, bid: null, ask: null, lastPrice: 2, impliedVolatility: 14 }], puts: [] } } };
    const j = T.slDteBuild({ nowMs: NOW, expirations: EXPS, chains: junk, risk: 150, ctx: { dir: 'LONG', spot: SPOT } });
    const jc = j.expiries[0].cands.find(c => !c.missing);
    ok(jc && jc.mid === 2 && jc.costSame === null && jc.eligible === false, 'no bid/ask → last price used, cost unknown, not eligible', jc);
    const errc = { '2026-10-01': { error: 'HTTP 502 <img src=x onerror=alert(1)>' } };
    const e = T.slDteBuild({ nowMs: NOW, expirations: EXPS, chains: errc, risk: 150, ctx: { dir: 'LONG', spot: SPOT } });
    const html = T.slDteHtml(e);
    ok(e.expiries[0].error.startsWith('HTTP 502') && !html.includes('<img') && html.includes('&lt;img'), 'API error text is escaped in the render');
}
// expiry verification from OCC symbols (both directions)
{
    const wrong = { ...chains, '2026-10-02': chain('2026-10-02', NOW, { symExp: '261001' }) };
    const m = T.slDteBuild({ ...base, chains: wrong, ctx: { dir: 'LONG', spot: SPOT } });
    ok(/server returned 2026-10-01 for 2026-10-02/.test(m.expiries[1].error || ''), 'server returns the wrong expiry → caught', m.expiries[1].error);
    ok(!m.expiries[0].error && !m.expiries[2].error, 'matching expiries are not flagged');
    ok(T.slDteRowsExpiry([{ contractSymbol: 'SPY261005C00765000' }]) === '2026-10-05', 'OCC symbol → expiry date');
    ok(T.slDteRowsExpiry([{}, { contractSymbol: 'junk' }]) === null, 'no symbol → null');
}
// spot fallback by put-call parity
{
    const nos = { '2026-10-01': chain('2026-10-01', NOW, { noSpot: true }), '2026-10-02': chain('2026-10-02', NOW, { noSpot: true }), '2026-10-05': chain('2026-10-05', NOW, { noSpot: true }) };
    const m = T.slDteBuild({ nowMs: NOW, expirations: EXPS, chains: nos, risk: 150, ctx: { dir: 'LONG' } });
    ok(m.spotSource === 'put-call parity' && near(m.spot, SPOT, 0.15), 'no live or chain spot → parity estimate within 0.15', [m.spotSource, m.spot]);
}
// market closed: same-day column hidden, plan for next open
{
    const SAT = et('2026-10-03', 12, 0);
    const sc = { '2026-10-06': chain('2026-10-06', SAT), '2026-10-07': chain('2026-10-07', SAT), '2026-10-08': chain('2026-10-08', SAT) };
    const m = T.slDteBuild({ nowMs: SAT, expirations: EXPS, chains: sc, risk: 150, ctx: { dir: 'LONG', spot: SPOT } });
    ok(!m.session.live && m.bestSame === null && m.expiries[0].expiry === '2026-10-06', 'weekend: plan for Monday, 1DTE = Tuesday, no same-day pick', [m.session, m.expiries[0].expiry]);
    const html = T.slDteHtml(m);
    ok(html.includes('Market closed') && !html.includes('Same-day pts'), 'render: closed-market note, same-day column hidden');
}

// ── 7. UW data: quote source, IV percentile, vendor Greeks (both directions) ──
{
    const mk = o => ({ '2026-10-01': chain('2026-10-01', NOW, o), '2026-10-02': chain('2026-10-02', NOW, o), '2026-10-05': chain('2026-10-05', NOW, o) });
    const uw = T.slDteBuild({ ...base, chains: mk({ source: 'unusual_whales' }), ctx: { dir: 'LONG', spot: SPOT } });
    ok(!uw.delayed && uw.sources.join() === 'UW' && uw.spotSource === 'SPY Logic live', 'UW quotes: live, labelled UW, live spot used', [uw.sources, uw.spotSource]);
    const cb = T.slDteBuild({ ...base, chains: mk({ source: 'cboe' }), ctx: { dir: 'LONG', spot: 770 } });   // live spot 5 pts away from the quotes
    ok(cb.delayed && cb.sources.join() === 'CBOE 15m' && /parity/.test(cb.spotSource) && near(cb.spot, SPOT, 0.15), 'CBOE quotes: delayed, live spot ignored, parity spot used', [cb.spot, cb.spotSource]);
    ok(T.slDteHtml(cb).includes('delayed: spot taken from the chain') && !T.slDteHtml(uw).includes('delayed: spot taken'), 'delayed warning renders only for delayed quotes');
    const lg = T.slDteBuild({ ...base, chains: mk({ route: 'legacy' }), ctx: { dir: 'LONG', spot: SPOT } });
    ok(lg.delayed && lg.sources.join() === 'legacy route (Yahoo)', 'legacy route: delayed and labelled', lg.sources);
    ok(T.slDteSourceInfo(T.slDteNormChain({ source: 'unusual_whales', _route: 'unified' })).delayed === false, 'normChain carries source and route');
    const ev = T.slDteBuild({ ...base, chains: mk({ source: '<img src=x>' }), ctx: { dir: 'LONG', spot: SPOT } });
    ok(!T.slDteHtml(ev).includes('<img') , 'API source string is escaped');
}
{
    const apexShape = { data: [{ days: 7, percentile: 0.9, volatility: 0.2 }, { days: 30, percentile: 0.13, volatility: 0.14, implied_move_perc: 0.04 }, { days: 60, percentile: 0.55, volatility: 0.16 }] };
    const r1 = T.slDteUwIv(apexShape);
    ok(r1 && near(r1.ivp, 13, 1e-9) && near(r1.iv, 0.14, 1e-12) && r1.days === 30, 'UW IV, APEX shape: 30-day record, percentile decimal → 13%', r1);
    const r2 = T.slDteUwIv({ data: [{ iv_percentile: 10 }, { iv_percentile: 42, iv30: 0.14, hv30: 0.12 }] });
    ok(r2 && r2.ivp === 42 && near(r2.hv, 0.12, 1e-12), 'UW IV, hashira shape: last record, iv_percentile in percent', r2);
    ok(near(T.slDteUwIv({ data: [{ iv_rank: 0.37 }] }).ivp, 37, 1e-9), 'iv_rank as a decimal → 37%');
    ok(T.slDteUwIv({ data: [{ foo: 1 }] }) === null && T.slDteUwIv(null) === null && T.slDteUwIv({ data: [] }) === null, 'no IV fields → null, never 0');
    ok(T.slDteUwIv({ data: [{ percentile: 3.5, volatility: 0.14 }] }).ivp === null, 'out-of-range percentile → null');
    const a = T.slDteBuild({ ...base, uwIv: apexShape, ctx: { dir: 'LONG', spot: SPOT, ivp: 80, hviv: 'ivgthv' } });
    ok(near(a.ivp, 13, 1e-9) && a.ivpSource === 'UW 30d', 'UW IVP overrides the manual input', [a.ivp, a.ivpSource]);
    ok(a.hviv === 'ivgthv' && a.hvivSource === 'manual', 'no UW HV → manual HV/IV kept');
    const m = T.slDteBuild({ ...base, uwIv: null, uwIvError: 'HTTP 502', ctx: { dir: 'LONG', spot: SPOT, ivp: 55, hviv: 'inline' } });
    ok(m.ivp === 55 && m.ivpSource === 'manual' && T.slDteHtml(m).includes('UW IV unavailable'), 'UW down → manual IVP, labelled');
    const h = T.slDteBuild({ ...base, uwIv: { data: [{ iv_percentile: 20, iv30: 0.15, hv30: 0.12 }] }, ctx: { dir: 'LONG', spot: SPOT, hviv: 'hvgtiv' } });
    ok(h.hviv === 'ivgthv' && h.hvivSource === 'UW' && near(h.hvRatio, 0.8, 1e-9), 'UW HV/IV 0.80 → IV > HV, overrides manual', [h.hviv, h.hvRatio]);
    const h2 = T.slDteBuild({ ...base, uwIv: { data: [{ iv_percentile: 20, iv30: 0.12, hv30: 0.15 }] }, ctx: { dir: 'LONG', spot: SPOT } });
    ok(h2.hviv === 'hvgtiv', 'UW HV/IV 1.25 → HV > IV');
}
{
    const mk = o => ({ '2026-10-01': chain('2026-10-01', NOW, o), '2026-10-02': chain('2026-10-02', NOW, o), '2026-10-05': chain('2026-10-05', NOW, o) });
    const plain = T.slDteBuild({ ...base, ctx: { dir: 'LONG', spot: SPOT } });
    const ven = T.slDteBuild({ ...base, chains: mk({ deltaShift: 3 }), ctx: { dir: 'LONG', spot: SPOT } });
    const atmP = plain.expiries[0].cands.find(c => c.style === 'ATM'), atmV = ven.expiries[0].cands.find(c => c.style === 'ATM');
    ok(plain.deltaSrc === 'model' && ven.deltaSrc === 'UW' && atmV.deltaSrc === 'UW', 'vendor delta used when present, model otherwise');
    ok(atmV.strike === atmP.strike + 3, 'strike picking follows the vendor delta', [atmP.strike, atmV.strike]);
    const pv = T.slDteBuild({ ...base, chains: mk({ deltaShift: 0, putDeltaPositive: true }), ctx: { dir: 'SHORT', spot: SPOT } });
    ok(pv.expiries[0].cands.every(c => c.delta < 0 && c.deltaSrc === 'UW'), 'positive vendor put delta normalised negative');
    const bad = T.slDteBuild({ ...base, chains: mk({ badDelta: true }), ctx: { dir: 'LONG', spot: SPOT } });
    ok(bad.deltaSrc === 'model', '|delta| > 1 from the vendor → ignored');
    const th = T.slDteBuild({ ...base, chains: mk({ theta: -0.42 }), ctx: { dir: 'LONG', spot: SPOT } });
    ok(th.hasTheta && th.expiries[0].cands[0].thetaUw === -42 && T.slDteHtml(th).includes('UW θ/day'), 'vendor theta shown per contract per day');
    ok(!plain.hasTheta && !T.slDteHtml(plain).includes('UW θ/day'), 'no vendor theta → no column');
}
ok(T.slDteHtml(null).includes('Load 1–3 DTE'), 'render before load shows the Load button');

console.log((fail ? '✗ ' : '✓ ') + pass + ' passed, ' + fail + ' failed');
process.exit(fail ? 1 : 0);
