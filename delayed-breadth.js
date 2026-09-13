/* Background-only adapter. Never emits scenario observations or entry permission. */
globalThis.ApexDelayedBreadth = (() => {
  const source='Massive minute aggregates / State Street SPY holdings';
  const finite=n=>typeof n==='number'&&Number.isFinite(n);
  const et=t=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(t));
  const timestamp=s=>typeof s==='string'&&/(Z|[+-]\d{2}:\d{2})$/.test(s)?Date.parse(s):NaN;
  function read(payload,now=Date.now()) {
    const out={status:'Unavailable',entryEligible:false,reason:'Waiting for delayed holdings breadth',snapshot:null};
    const p=payload?.latest;if(!p){out.reason=payload?.collecting?'Collecting delayed holdings data':String(payload?.reason||out.reason);return out;}
    const asof=timestamp(p.as_of),received=timestamp(p.available_at),m=p.metrics;
    if(p.source!==source||payload.delayMinutes!==15||!Number.isFinite(asof)||!Number.isFinite(received)||asof>received||received>now||et(asof)!==p.session){out.reason='Invalid source or timestamps';return out;}
    if(p.status!=='complete'||!Number.isInteger(p.total)||p.total<490||p.total>510||!Number.isInteger(p.covered)||p.covered>p.total||p.covered/p.total<0.98){out.reason='Incomplete holdings coverage; metrics withheld';return out;}
    if(![p.advancing,p.declining,p.unchanged].every(n=>Number.isInteger(n)&&n>=0)||p.advancing+p.declining+p.unchanged!==p.covered){out.reason='Invalid participation counts';return out;}
    if(!m||!['adRatio','upDownVolumeRatio','sp500AboveVwapPct'].every(k=>m[k]===null||finite(m[k])&&m[k]>=0)||m.sp500AboveVwapPct===null||m.sp500AboveVwapPct>100||p.declining===0&&m.adRatio!==null||p.declining>0&&(!finite(m.adRatio)||Math.abs(m.adRatio-p.advancing/p.declining)>0.001)){out.reason='Invalid breadth measurements';return out;}
    const age=(now-asof)/60000;
    out.status=et(now)===p.session&&age>=15&&age<=25?'Delayed context':'Historical / stale';
    out.reason=out.status==='Delayed context'?'15-minute-delayed background only; never live entry confirmation':'Outside the delayed-context freshness window; excluded from current AI context';
    out.snapshot={session:p.session,as_of:p.as_of,available_at:p.available_at,source,universe:'SPY equity holdings proxy; not exchange-wide breadth',covered:p.covered,total:p.total,coveragePct:100*p.covered/p.total,holdingsDate:p.holdingsDate,ageMinutes:Math.floor(age),advancing:p.advancing,declining:p.declining,unchanged:p.unchanged,metrics:{...m}};
    return out;
  }
  function ai(payload,now=Date.now()) {
    const c=read(payload,now);
    return c.status==='Delayed context'?{...c,instruction:'Discuss only as delayed background at as_of. Do not describe as current, infer current breadth, or use to approve an entry.'}:{status:c.status,entryEligible:false,reason:c.reason};
  }
  return {read,ai};
})();
