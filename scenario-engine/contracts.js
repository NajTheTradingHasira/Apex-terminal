import {sessionFor} from './calendar.js';
import {eastern} from './time.js';

const finite=n=>typeof n==='number'&&Number.isFinite(n);
/** Screening defaults are explicit research filters, not calibrated trade advice. */
export function screenContracts(payload,read,spot,now=Date.now()) {
  const day=eastern(new Date(now).toISOString()).date,session=sessionFor(day,new Date(now).toISOString());
  const type=read?.bias?.dir==='LONG'?'call':read?.bias?.dir==='SHORT'?'put':null;
  const out={candidates:[],reviewCandidates:[],source:payload?.source||null,providerStatus:payload?.status||'waiting',rejected:{},status:'WAIT',reason:'',partial:!!payload?.partial};
  if(!session.isTradingDay||now<Date.parse(session.open)+15*60000||now>=Date.parse(session.close)-15*60000){out.reason='Outside 9:45 ET to 15 minutes before session close';return out;}
  if(!type||!finite(spot)||spot<=0){out.reason='Directional SPY setup and current spot required';return out;}
  if(payload?.status!=='received'||payload.expiry!==day||!Array.isArray(payload.results)){out.reason=payload?.reason||'Waiting for today’s option snapshot';return out;}
  const receiptAge=now-Date.parse(payload.receivedAt);
  if(!Number.isFinite(receiptAge)||receiptAge<0||receiptAge>60000){out.reason='Snapshot receipt is stale';return out;}
  const seen=new Set();
  for(const r of payload.results) {
    const d=r.details||{},q=r.last_quote||{},g=r.greeks||{};
    let reason;
    const quoteMs=finite(q.last_updated)?q.last_updated/1e6:NaN,age=now-quoteMs;
    const mid=(q.bid+q.ask)/2,spread=(q.ask-q.bid)/mid;
    const delta=g.delta;
    if(d.expiration_date!==day||d.contract_type!==type||(d.shares_per_contract!==100&&!(payload.source==='Unusual Whales'&&d.shares_per_contract==null))||!/^O:SPY\d{6}[CP]\d{8}$/.test(d.ticker||''))reason='Wrong expiry, direction or contract';
    else if(seen.has(d.ticker))reason='Duplicate contract';
    else if(![q.bid,q.ask].every(x=>finite(x)&&x>0)||q.ask<q.bid)reason='Invalid or empty two-sided quote';
    else if(spread>0.10)reason='Spread above 10% of midpoint';
    else if(!finite(delta)||(type==='call'?delta<0.35||delta>0.65:delta> -0.35||delta< -0.65))reason='Delta outside 0.35–0.65 magnitude';
    else if(!finite(r.day?.volume)||r.day.volume<100||!finite(r.open_interest)||r.open_interest<100)reason='Volume or prior-day open interest below 100';
    else if(!finite(d.strike_price)||Math.abs(d.strike_price-spot)>10)reason='Strike outside $10 of SPY';
    seen.add(d.ticker);
    if(reason){out.rejected[reason]=(out.rejected[reason]||0)+1;continue;}
    const candidate={symbol:d.ticker,strike:d.strike_price,type,expiry:day,bid:q.bid,ask:q.ask,spreadPct:spread*100,delta,volume:r.day.volume,oi:r.open_interest,quoteAt:Number.isFinite(quoteMs)?new Date(quoteMs).toISOString():null,premium:q.ask*100};
    const verified=q.timeframe==='REAL-TIME'&&Number.isFinite(age)&&age>=0&&age<=30000&&[q.bid_size,q.ask_size].every(x=>finite(x)&&x>0)&&d.shares_per_contract===100;
    if(verified)out.candidates.push(candidate);
    else if(payload.source==='Unusual Whales')out.reviewCandidates.push({...candidate,verification:'Confirm current bid/ask, sizes and standard deliverable with broker',lastTapeTime:r.lastTapeTime||null});
    else out.rejected['Quote delayed, unverified or older than 30s']=(out.rejected['Quote delayed, unverified or older than 30s']||0)+1;
  }
  const rank=(a,b)=>a.spreadPct-b.spreadPct||Math.abs(Math.abs(a.delta)-0.5)-Math.abs(Math.abs(b.delta)-0.5)||b.volume-a.volume;
  out.candidates.sort(rank);out.reviewCandidates.sort(rank);out.reviewCandidates=out.reviewCandidates.slice(0,5);
  out.candidates=out.candidates.slice(0,5);
  out.status=out.candidates.length?'SCREENED':out.reviewCandidates.length?'REVIEW':'WAIT';
  out.reason=out.candidates.length?'Liquidity filters passed; entry permission remains '+read.gate:out.reviewCandidates.length?'UW shortlist — broker quote confirmation required; entry permission remains '+read.gate:'No contracts pass all filters';
  return out;
}
