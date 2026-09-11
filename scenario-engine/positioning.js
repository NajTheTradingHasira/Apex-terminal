import {eastern} from './time.js';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
export const POSITION_METHOD='All-expiry SPY OI gamma proxy: sum(callGammaOi - abs(putGammaOi)). Assumes positive call and negative put contributions; not observed dealer inventory. Top five strikes ranked by absolute net contribution. No gamma-flip calculation.';

export function positioningEvidence(feed,session,now=Date.now()) {
  const out={levels:[],observations:[],status:'Unavailable',reason:'Waiting for UW positioning',count:0,asOf:null,model:POSITION_METHOD};
  const p=feed?.payload,receipt=Date.parse(feed?.receivedAt);
  if(!p)return out;
  if(p.source!=='unusual_whales'||p.ticker!=='SPY'||p.date!==session.id||p.expiry!=null||!Array.isArray(p.strikes)||!p.strikes.length){out.reason='Expected same-day all-expiry SPY exposure';return out;}
  if(!Number.isFinite(receipt)||receipt>now||now-receipt>120000){out.reason='Positioning receipt stale or invalid';return out;}
  const seen=new Set(),rows=[];
  for(const r of p.strikes) {
    const t=typeof r.time==='string'&&/(Z|[+-]\d{2}:\d{2})$/.test(r.time)?Date.parse(r.time):NaN;
    if(!finite(r.strike)||r.strike<=0||seen.has(r.strike)||![r.callGammaOi,r.putGammaOi].every(finite)||r.callGammaOi<0||!Number.isFinite(t)||t>receipt||eastern(r.time).date!==session.id){out.reason='Invalid, duplicated or misdated UW exposure rows';return out;}
    seen.add(r.strike);
    if(r.callGammaOi!==0||r.putGammaOi!==0)rows.push({strike:r.strike,t,net:r.callGammaOi-Math.abs(r.putGammaOi)});
  }
  out.count=p.strikes.length;
  if(p.strikes.length>=500){out.reason='Exposure profile may be truncated at the endpoint limit';return out;}
  if(!rows.length){out.reason='No nonzero exposure available';return out;}
  out.asOf=new Date(Math.min(...rows.map(r=>r.t))).toISOString();
  if(rows.some(r=>now-r.t>600000)){out.status='Stale';out.reason='UW source observations are older than 10 minutes; no positioning evidence applied';return out;}
  const net=rows.reduce((sum,r)=>sum+r.net,0);
  if(!finite(net)||!rows.some(r=>r.net!==0)){out.reason='No nonzero exposure available';return out;}
  const expiry=new Date(Math.min(Date.parse(session.close),Date.parse(out.asOf)+600000)).toISOString();
  // Source timestamps determine freshness; browser receipt determines when known.
  out.levels=rows.filter(r=>r.net!==0).sort((a,b)=>Math.abs(b.net)-Math.abs(a.net)||a.strike-b.strike).slice(0,5).map(r=>({id:'uw-position-strike-'+r.strike,kind:'STRIKE',value:r.strike,as_of:new Date(r.t).toISOString(),available_at:feed.receivedAt,valid_until:expiry,source:'UW SPY all-expiry spot exposures by strike',method:POSITION_METHOD}));
  out.observations=[{metric:'dealerGammaSign',value:Math.sign(net),as_of:out.asOf,available_at:feed.receivedAt,source:'UW exposure / Apex assumption-based gamma model',method:POSITION_METHOD}];
  out.status='Connected';out.reason='Fresh modeled positioning context';out.netGamma=net;
  return out;
}
