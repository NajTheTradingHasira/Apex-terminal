import {parseNyFedCalendar,nyFedSource} from './nyFedCalendar.js';
import {parseFomcCalendar,FOMC_SOURCE} from './fomcCalendar.js';
import {parseFomcTiming,timingSource,easternDay} from './fomcTiming.js';
import {combinedEventMonitor} from './eventMonitor.js';
import {scheduledCoverage} from './scheduledCoverage.js';

export function parseDocuments(payload,now=Date.now()) {
  const feeds={},errors=[];
  for(const [key,parse,url] of [['nyfed',parseNyFedCalendar,nyFedSource(now).url],['fomc',parseFomcCalendar,FOMC_SOURCE],['fomcTimes',parseFomcTiming,timingSource(easternDay(now))]]) {
    const doc=payload?.documents?.[key],stamp=Date.parse(doc?.fetchedAt);
    try {
      if(doc?.source!==url||!Number.isFinite(stamp)||stamp>now||now-stamp>600000)throw Error('Missing or stale official document');
      feeds[key]=parse(doc.html,stamp);
    } catch(e){errors.push(key+': '+e.message);}
  }
  return {feeds,errors};
}

export class CalendarEvidence {
  constructor(){this.feeds={};this.events=new Map();this.errors=[];}
  receive(payload,now){const result=parseDocuments(payload,now);this.feeds=result.feeds;this.errors=result.errors;}
  read(at) {
    const status=combinedEventMonitor(null,null,at,this.feeds.nyfed,this.feeds.fomc,this.feeds.fomcTimes);
    const day=easternDay(Date.parse(at));
    for(const [id,e] of this.events)if(easternDay(Date.parse(e.at))<day)this.events.delete(id);
    for(const e of status.events)if(!this.events.has(e.id))this.events.set(e.id,e);
    const check=scheduledCoverage(this.feeds,at);
    return {events:[...this.events.values()],eventCoverage:check.coverage,check,errors:this.errors};
  }
}

/** QQQ/IWM are cross-asset evidence, never a substitute for constituent breadth. */
export function crossAsset(payload,symbol,session,receivedAt) {
  if(payload?.ticker!==symbol||payload.interval!=='1m'||!Array.isArray(payload.data))return [];
  const now=Date.parse(receivedAt),open=Date.parse(session.open),close=Date.parse(session.close);
  const rows=payload.data.map(b=>({...b,t:Date.parse(b.date)})).filter(b=>b.t>=open&&b.t+60000<=Math.min(now-5000,close)).sort((a,b)=>a.t-b.t);
  if(!rows.length||rows.some((b,i)=>b.t!==open+i*60000||!['open','high','low','close','volume'].every(k=>typeof b[k]==='number'&&Number.isFinite(b[k])&&b[k]>0)||b.low>Math.min(b.open,b.close)||b.high<Math.max(b.open,b.close)))return [];
  const end=rows.at(-1).t+60000;
  if(now-end>120000)return [];
  return [{metric:symbol.toLowerCase()+'ReturnPct',value:(rows.at(-1).close/rows[0].open-1)*100,as_of:new Date(end).toISOString(),available_at:receivedAt,source:'Nexus '+symbol+' completed 1m candles',method:'Regular-session open to latest completed close return; contiguous RTH candles; cross-asset confirmation only'}];
}
