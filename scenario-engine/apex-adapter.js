import {ScenarioEngine} from './engine.js';
import {scenarioDefinitions} from './scenarios.js';
import {sessionFor} from './calendar.js';
import {eastern} from './time.js';
import {CalendarEvidence,crossAsset} from './feeds.js';
import {SessionHistory,engineState} from './history.js';
import {screenContracts} from './contracts.js';

const iso=t=>new Date(t).toISOString();
const positive=n=>typeof n==='number'&&Number.isFinite(n)&&n>0;
export const ADAPTER_VERSION='1.1.0';

/** One-minute bars are already validated by Apex. Recheck the boundaries here. */
export function aggregateFive(scan, session, receivedAt, receipts=new Map()) {
  if(!scan?.metrics || !Array.isArray(scan.bars))return {bars:[],levels:[],observations:[]};
  const rows=scan.bars.filter(b=>b.t>=Date.parse(session.open)&&b.t+60000<=Date.parse(session.close));
  if(rows.some((b,i)=>!['o','h','l','c','v'].every(k=>positive(b[k]))||b.h<Math.max(b.o,b.c,b.l)||b.l>Math.min(b.o,b.c)||b.t!==Date.parse(session.open)+i*60000))throw new Error('Incomplete or invalid minute history');
  const bars=[],levels=[],observations=[];
  for(let i=0;i+5<=rows.length;i+=5) {
    const chunk=rows.slice(i,i+5),last=chunk.at(-1),end=last.t+60000;
    if(end>Date.parse(receivedAt))break;
    const bar={start:iso(chunk[0].t),end:iso(end),timeframe:5,open:chunk[0].o,high:Math.max(...chunk.map(b=>b.h)),low:Math.min(...chunk.map(b=>b.l)),close:last.c,volume:chunk.reduce((s,b)=>s+b.v,0),source:'Yahoo via Nexus; five complete 1m candles'};
    const key=JSON.stringify(bar);
    if(!receipts.has(key))receipts.set(key,receivedAt);
    bar.available_at=receipts.get(key); bars.push(bar);
    if(positive(last.vwap))levels.push({id:'session-vwap',kind:'VWAP',value:last.vwap,as_of:bar.end,available_at:bar.available_at,source:'Apex completed SPY minute candles',method:'Cumulative HLC3 × volume / cumulative volume; estimate, not exchange VWAP'});
    if(bars.length>=5) {
      const baseline=bars.slice(-5,-1).reduce((s,b)=>s+b.volume,0)/4;
      if(baseline>0)observations.push({metric:'relativeVolume',value:bar.volume/baseline,as_of:bar.end,available_at:bar.available_at,source:'Apex five-minute candles',method:'Completed 5m volume / average preceding four 5m blocks; within-session ratio, not historical same-time RVOL'});
    }
  }
  return {bars,levels,observations};
}

export function dailyLevels(payload,session,receivedAt) {
  if(!Array.isArray(payload?.ohlcv))return [];
  let prior;
  for(let i=1;i<=15;i++) {
    const day=iso(Date.parse(session.id+'T12:00:00Z')-i*86400000).slice(0,10);
    const s=sessionFor(day,receivedAt);if(s.isTradingDay){prior=s;break;}
  }
  const byDay=new Map();
  for(const raw of payload.ohlcv) {
    const day=typeof raw.date==='string'?raw.date.slice(0,10):'';
    if(!/^\d{4}-\d{2}-\d{2}$/.test(day)||day>=session.id)continue;
    if(!['open','high','low','close'].every(k=>positive(raw[k]))||raw.low>Math.min(raw.open,raw.close)||raw.high<Math.max(raw.open,raw.close)||byDay.has(day))return [];
    byDay.set(day,{...raw,day});
  }
  const rows=[...byDay.values()].sort((a,b)=>a.day.localeCompare(b.day)),last=rows.at(-1);
  if(!prior||last?.day!==prior.id||Date.parse(receivedAt)<Date.parse(prior.close))return [];
  const level=(kind,value,method)=>({id:kind,kind,value,as_of:prior.close,available_at:receivedAt,valid_until:session.close,source:'Nexus SPY daily OHLCV',method});
  const out=[['PDH','high'],['PDL','low'],['PDC','close']].map(([kind,key])=>level(kind,last[key],prior.id+' completed daily '+key+'; provider adjustment policy'));
  if(rows.length>=15) {
    const tail=rows.slice(-15);
    const atr=tail.slice(1).reduce((sum,b,i)=>sum+Math.max(b.high-b.low,Math.abs(b.high-tail[i].close),Math.abs(b.low-tail[i].close)),0)/14;
    if(positive(atr))out.push(level('ATR',atr,'14 completed daily true ranges; simple mean, excludes today'));
  }
  return out;
}

export class ApexScenarioAdapter {
  constructor(){this.reset();}
  reset(){this.engine=new ScenarioEngine({allowProvisionalWithoutEventCoverage:true});this.receipts=new Map();this.day=null;this.session=null;this.card=null;this.dataset=null;this.error='';}
  update(scan,daily,now=Date.now(),supplement={}) {
    const at=iso(now),day=eastern(at).date;
    if(day!==this.day){this.reset();this.day=day;this.session=sessionFor(day,at);}
    try {
      const data=aggregateFive(scan,this.session,at,this.receipts);
      const supplied=daily?.receivedAt?dailyLevels(daily.payload,this.session,daily.receivedAt):[];
      const extra=Object.entries(supplement.assets||{}).flatMap(([symbol,p])=>crossAsset(p.payload,symbol,this.session,p.receivedAt));
      const dataset={symbol:'SPY',session:this.session,bars:data.bars,levels:[...data.levels,...supplied],observations:[...data.observations,...extra],events:supplement.events||[],eventCoverage:supplement.eventCoverage||[]};
      this.dataset=dataset;
      this.before=engineState(this.engine);
      this.card=this.engine.evaluate(dataset,at);this.error='';return this.card;
    } catch(e){this.card=null;this.error=e.message;return null;}
  }
}

/** Optional scenario approval is downward-only; unavailable never means approved. */
export function applyScenarioGate(read,card,required=false) {
  if(!required)return read;
  const expected=read.bias.dir==='LONG'?'bullish':read.bias.dir==='SHORT'?'bearish':'neutral';
  if(card?.trade_permitted&&card.directional_bias===expected&&expected!=='neutral')return read;
  return {...read,gate:'NO-GO',reasons:[...read.reasons,'SCENARIO APPROVAL — '+(!card?'Scenario data unavailable.':card.decision_reasons.join('; ')||'Scenario direction does not agree with the entry.')],stops:{...read.stops,runner:false,maxHoldMin:0,time:'No new entry',targets:[],target:'No new entry — scenario confirmation required',...(read.stops.plan?{plan:{...read.stops.plan,runnerEligible:false,maxHoldMin:0,targets:[]}}:{})}};
}

export function libraryRows(card) {
  return scenarioDefinitions.map(rule=>{
    const checks=(card?.explanation?.candidates||[]).filter(c=>c.id===rule.id);
    const active=card?.scenario_id===rule.id;
    return {rule,checks,status:active?'CURRENT CONTEXT':!checks.length?'OUTSIDE WINDOW':checks.some(c=>c.eligible)?'PRICE RULES MET':checks.some(c=>c.checks.some(x=>x.unavailable))?'MISSING INPUTS':'WATCHING'};
  });
}

const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const runtime=new ApexScenarioAdapter();
const history=new SessionHistory();
let replayStatus='';
let contractPayload=null,contractLoading=false,contractAttempt=0,contractScreen=null;
const calendar=new CalendarEvidence();
let feedLoading=false,feedAttempt=0,calendarAttempt=0,assets={},coverage=null;
let required=false,daily=null,loading=false,lastAttempt=0;
const BASE='https://nexus-terminal-production-9d34.up.railway.app';
async function getJSON(path,timeout=15000) {
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),timeout);
  try {const r=await fetch(BASE+path,{signal:controller.signal});if(!r.ok)throw Error('HTTP '+r.status);return await r.json();}
  finally{clearTimeout(timer);}
}
async function loadContracts(spot) {
  if(contractLoading||Date.now()-contractAttempt<30000)return;
  contractLoading=true;contractAttempt=Date.now();
  try{contractPayload=await getJSON('/api/scenario/spy-contracts?spot='+encodeURIComponent(spot));}
  catch{contractPayload=null;}
  finally{contractLoading=false;window.slRenderLocalRead?.();}
}
async function loadFeeds() {
  if(feedLoading||Date.now()-feedAttempt<60000)return;
  feedLoading=true;feedAttempt=Date.now();
  const jobs=['QQQ','IWM'].map(async symbol=>{
    try {assets[symbol]={payload:await getJSON('/api/stock/'+symbol+'/intraday?interval=1m'),receivedAt:iso(Date.now())};}
    catch {delete assets[symbol];}
  });
  if(Date.now()-calendarAttempt>=300000) {
    calendarAttempt=Date.now();
    jobs.push((async()=>{try{calendar.receive(await getJSON('/api/scenario/calendars',45000),Date.now());}catch{calendar.receive(null,Date.now());calendarAttempt=Date.now()-240000;}})());
  }
  await Promise.allSettled(jobs);feedLoading=false;window.slRenderLocalRead?.();
}
async function loadDaily() {
  if(loading||Date.now()-lastAttempt<900000)return;
  loading=true;lastAttempt=Date.now();let timer;
  try {
    const controller=new AbortController();timer=setTimeout(()=>controller.abort(),15000);
    const r=await fetch(BASE+'/api/stock/SPY?period=3mo&interval=1d',{signal:controller.signal});
    if(!r.ok)throw new Error('Daily history unavailable');
    const payload=await r.json();if(!Array.isArray(payload.ohlcv))throw new Error('Invalid daily history');
    const day=eastern(iso(Date.now())).date;
    const signature=JSON.stringify(payload.ohlcv.filter(r=>typeof r.date==='string'&&r.date.slice(0,10)<day));
    daily={payload,receivedAt:daily?.signature===signature?daily.receivedAt:iso(Date.now()),signature};
  } catch {daily=null;} finally {clearTimeout(timer);loading=false;window.slRenderLocalRead?.();}
}

export function update(read,scan) {
  if(scan?.bars?.length){loadDaily();loadFeeds();}
  const now=Date.now();
  coverage=calendar.read(iso(now));
  const card=runtime.update(scan,daily,now,{...coverage,assets});
  const gated=applyScenarioGate(read,card,required);
  const spot=scan?.valid?scan.bars?.at(-1)?.c:null;
  if(spot)loadContracts(spot);
  contractScreen=screenContracts(contractPayload,gated,spot);
  if(card&&runtime.dataset)history.capture({at:card.timestamp,dataset:runtime.dataset,before:runtime.before,config:runtime.engine.config,card,entry:{gate:gated.gate,setup:read.setup?.status||null},contracts:contractScreen});
  render(card);
  return gated;
}
export function summary(){const c=runtime.card;return c?{scenario:c.active_scenario,direction:c.directional_bias,provisional:c.provisional,decision:c.decision,decision_reasons:c.decision_reasons,expected_path:c.expected_path,target_ladder:c.target_ladder,invalidation:c.invalidation,next_state:c.next_state_if_invalidated,required_for_entry:required,scheduled_coverage:coverage?.check||null,contract_screen:contractScreen}:null;}
export function fail(read,message){runtime.card=null;runtime.error=message;render(null);return applyScenarioGate(read,null,required);}

export function render(card) {
  const el=document.getElementById('slScenarioLibrary');if(!el)return;
  const expanded=new Set([...el.querySelectorAll('details[open]')].map(x=>x.dataset.id));
  const closed=card?.checkpoint==='closed';
  el.innerHTML='<div style="padding:1rem;background:var(--surface-2);border:1px solid var(--border-strong);border-radius:6px">'+
    '<strong style="color:var(--accent)">SCENARIO CONTEXT · 18 TEMPLATES</strong><div style="margin:0.5rem 0;font-size:0.8rem">'+esc(closed?'Market closed — scenario entries inactive':card?.active_scenario||runtime.error||'Waiting for completed five-minute bars')+'</div>'+
    '<div style="font-size:0.72rem;color:var(--text-muted)">Original five-minute scenario rules from your scenario engine. The one-minute entry detector remains separate. Levels become eligible only after Apex receives them; opening the panel mid-session does not invent earlier observations.</div>'+
    '<div style="margin:0.5rem 0;font-size:0.72rem;color:var(--warn)">Scenario permission: '+esc(card?.decision||'Unavailable')+'. '+esc(card?.decision_reasons?.join(' · ')||runtime.error)+'</div>'+
    '<label style="font-size:0.72rem"><input id="slRequireScenario" type="checkbox" '+(required?'checked':'')+'> Require scenario approval for new entries</label>'+
    '<div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.5rem">'+(loading?'Loading prior-day levels…':daily?'Prior-day OHLC and daily ATR connected.':'Prior-day levels unavailable.')+' Value profile, intraday breadth and positioning remain unavailable. Missing data never counts as confirmation.</div>'+
    '<div style="font-size:0.72rem;margin-top:0.6rem"><b>Scheduled events:</b> '+esc(coverage?.check.passed?'Verified within stated scope':coverage?.check.reasons.join(' · ')||'Loading official calendars')+'<br>'+esc(coverage?.check.exclusions||'')+'<br><b>Cross-asset:</b> '+esc(['QQQ','IWM'].map(s=>s+': '+(card?.context?.metrics?.[s.toLowerCase()+'ReturnPct']!==undefined?'connected':'unavailable')).join(' · '))+'</div>'+
    (card&&!closed?'<div style="font-size:0.72rem;margin-top:0.7rem"><b>Expected path:</b> '+card.expected_path.map(esc).join(' → ')+'<br><b>Invalidation:</b> '+card.invalidation.map(x=>esc(x.description)).join(' · ')+'<br><b>Next state:</b> '+esc(card.next_state_if_invalidated.scenario_id)+'</div>':'')+
    '<details data-id="contracts" '+(expanded.has('contracts')?'open':'')+' style="margin-top:0.8rem"><summary>0DTE contract screen · '+esc(contractScreen?.status||'WAIT')+'</summary><div style="font-size:0.72rem;padding:0.5rem 0">'+esc(contractScreen?.reason||'Waiting for current SPY candles')+(contractLoading?' · Refreshing…':'')+'<br>Same-day standard contracts · real-time quote ≤30s · spread ≤10% · |delta| 0.35–0.65 · volume/OI ≥100. Sorted by spread, distance from 0.50 delta, then volume. Premium is ask ×100; entry checks still apply.'+(contractScreen?.partial?'<br>Partial provider result; ranking covers received contracts only.':'')+'<br>'+(contractScreen?.candidates||[]).map(c=>esc(c.symbol)+' · Bid/ask '+c.bid.toFixed(2)+' / '+c.ask.toFixed(2)+' · '+c.spreadPct.toFixed(1)+'% spread · Δ '+c.delta.toFixed(2)+' · Ask premium $'+c.premium.toFixed(0)+' · Quote '+esc(c.quoteAt)).join('<br>')+'<br>'+Object.entries(contractScreen?.rejected||{}).map(([reason,n])=>esc(reason)+': '+n).join(' · ')+'</div></details>'+
    '<details data-id="history" '+(expanded.has('history')?'open':'')+' style="margin-top:0.8rem"><summary>Session history · '+history.rows.length+' checkpoints</summary><div style="font-size:0.72rem;padding:0.5rem 0">'+esc(history.status)+'. One checkpoint per minute while this panel receives data; up to 500 retained. Replay checks rule reproducibility, not trading performance.<br><button class="api-btn" style="margin:0.5rem 0" id="slReplayHistory">Verify replay</button> <button class="api-btn" style="margin:0.5rem 0" id="slExportHistory">Export inputs</button> '+esc(replayStatus)+'<br>'+history.rows.slice(-10).reverse().map(r=>esc(new Date(r.at).toLocaleTimeString('en-US',{timeZone:'America/New_York'}))+' ET · '+esc(r.card.active_scenario)+' · '+esc(r.card.decision)+' · Entry '+esc(r.entry.gate)).join('<br>')+'</div></details>'+
    '<details data-id="library" '+(expanded.has('library')?'open':'')+' style="margin-top:0.8rem"><summary style="cursor:pointer">Scenario library · 18 templates</summary>'+libraryRows(card).map(({rule,checks,status})=>
      '<details data-id="'+rule.id+'" '+(expanded.has(rule.id)?'open':'')+' style="border-top:1px solid var(--border-strong);padding:0.65rem 0"><summary style="cursor:pointer;font-size:0.8rem">'+esc(rule.scenario)+' · '+(rule.direction==='both'?'bullish / bearish':'neutral')+' <span style="font-size:0.6rem;color:var(--text-muted)">'+status+'</span></summary><div style="font-size:0.72rem;line-height:1.6;padding:0.5rem">'+
      '<b>Window:</b> '+rule.windows.map(esc).join(', ')+'<br><b>Path:</b> '+rule.expected_path.map(esc).join(' → ')+'<br><b>Expires:</b> '+rule.time_expiry.minutes+' minutes or phase end<br><b>Next state:</b> '+esc(rule.next_state_if_invalidated)+'<br>'+
      (checks.length?checks.map(c=>'<b>'+esc(c.direction)+'</b>: '+c.checks.map(x=>(x.pass?'✓ ':x.unavailable?'Missing: ':'Waiting: ')+esc(x.reason)).join(' · ')+(c.reference_price?'<br>Reference $'+c.reference_price.toFixed(2):'')).join('<br>'):'No live evaluation in the current time window.')+'</div></details>').join('')+'</details></div>';
  document.getElementById('slReplayHistory').onclick=async()=>{try{const r=await history.verify();replayStatus=r.checked?r.matched+'/'+r.checked+' checkpoints reproduced':'No saved checkpoints yet';}catch{replayStatus='Replay failed: saved inputs could not be verified';}render(runtime.card);};
  document.getElementById('slExportHistory').onclick=()=>history.download().catch(()=>{replayStatus='Export failed';render(runtime.card);});
  document.getElementById('slRequireScenario').onchange=e=>{required=e.target.checked;window.slRenderLocalRead?.();};
}
