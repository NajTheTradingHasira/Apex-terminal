import {fomcMonitor} from './fomcCalendar.js';
import {freshTiming,easternDay} from './fomcTiming.js';
// Provider contract: event (name), time (UTC), type. No coverage interval is supplied.
export function eventMonitor(payload, receivedAt) {
  const source='Unusual Whales /market/economic-calendar';
  if(!Array.isArray(payload?.data)) return {source,status:'unavailable',message:'Event feed unavailable or malformed.',events:[],receivedAt};
  if(payload.data.length>2000) return {source,status:'unavailable',message:'Event response exceeds the supported size.',events:[],receivedAt};
  const events=[],seen=new Set();let rejected=0;
  for(const row of payload.data) {
    if(typeof row.event!=='string'||!row.event.trim()||row.event.length>500||typeof row.time!=='string'||!/(Z|[+-]\d{2}:\d{2})$/.test(row.time)||!Number.isFinite(Date.parse(row.time))) {rejected++;continue;}
    const at=new Date(row.time).toISOString(),name=row.event.trim(),id=`uw:${at}:${name}`;
    if(seen.has(id))continue;seen.add(id);
    events.push({id,type:name,at,available_at:receivedAt,source,status:'scheduled',macroSensitive:true});
  }
  events.sort((a,b)=>Date.parse(a.at)-Date.parse(b.at));
  return {source,status:events.length?'partial':'unverified',receivedAt,events,rejected,
    message:events.length?`${events.length} scheduled events received. Coverage dates and completeness are not certified; trade permission remains blocked.`:'Provider returned no usable events. This is not confirmation of an event-free session.'};
}

export function combinedEventMonitor(uw,bls,receivedAt,nyfed,fomc,timing) {
  const primary=eventMonitor(uw,receivedAt),official=eventMonitor(bls,receivedAt);
  const fresh=Number.isFinite(Date.parse(bls?.fetchedAt))&&Date.parse(receivedAt)-Date.parse(bls.fetchedAt)>=0&&Date.parse(receivedAt)-Date.parse(bls.fetchedAt)<=600000;
  const officialEvents=fresh?official.events.map(e=>({...e,id:`bls:${e.at}:${e.type}`,source:'BLS official release calendar https://www.bls.gov/schedule/news_release/bls.ics'})):[];
  const events=[...primary.events,...officialEvents].sort((a,b)=>Date.parse(a.at)-Date.parse(b.at));
  const sources=[{name:'Unusual Whales',status:primary.status,count:primary.events.length},{name:'BLS',status:fresh?official.status:'unavailable',count:officialEvents.length,fetchedAt:bls?.fetchedAt??null}];
  const fed=eventMonitor(nyfed,receivedAt),age=Date.parse(receivedAt)-Date.parse(nyfed?.fetchedAt);
  const fedFresh=Number.isFinite(age)&&age>=0&&age<=600000&&/^https:\/\/www\.newyorkfed\.org\/research\/calendars\/i-[a-z]{3}\d{2}\.html$/.test(nyfed?.source);
  const fedEvents=fedFresh?fed.events.map(e=>({...e,id:`nyfed:${e.at}:${e.type}`,source:`New York Fed economic indicators calendar ${nyfed.source}`})):[];
  events.push(...fedEvents);events.sort((a,b)=>Date.parse(a.at)-Date.parse(b.at));
  sources.push({name:'New York Fed',status:fedFresh?fed.status:'unavailable',count:fedEvents.length,fetchedAt:nyfed?.fetchedAt??null});
  const timingEvents=freshTiming(timing,receivedAt)?eventMonitor(timing,receivedAt).events.filter(e=>easternDay(Date.parse(e.at))>=easternDay(Date.parse(receivedAt))).map(e=>({...e,id:`fomc-timing:${e.at}:${e.type}`,source:timing.source,status:'uncertain'})):[];
  events.push(...timingEvents);events.sort((a,b)=>Date.parse(a.at)-Date.parse(b.at));
  sources.push({name:'Fed FOMC times',status:timingEvents.length?'partial':'unavailable',count:timingEvents.length});
  const contributors=sources.filter(s=>s.count>0).map(s=>s.name);
  const source=contributors.length?contributors.join(' + '):'No usable event source';
  return {events,receivedAt,source,status:events.length?'partial':sources.every(s=>s.status==='unavailable')?'unavailable':'unverified',
    message:events.length?`${events.length} scheduled entries available from ${source}. Full macro and headline coverage remains unverified.`:'No usable scheduled events received. This does not establish an event-free session.',sources,fomc:{...fomcMonitor(fomc,receivedAt),timingEntries:timingEvents}};
}
