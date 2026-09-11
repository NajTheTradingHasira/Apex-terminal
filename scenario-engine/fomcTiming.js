import {dateTime} from './blsCalendar.js';
const months=['january','february','march','april','may','june','july','august','september','october','november','december'];
export function timingSource(day){return `https://www.federalreserve.gov/newsevents/${day.slice(0,4)}-${months[Number(day.slice(5,7))-1]}.htm`;}
export const easternDay=now=>new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now));
const clean=s=>s.replace(/<[^>]*>/g,'').replace(/&nbsp;/g,' ').trim();
export function parseFomcTiming(html,now=Date.now()){
 const day=easternDay(now),month=months[Number(day.slice(5,7))-1],title=month[0].toUpperCase()+month.slice(1)+' '+day.slice(0,4);
 if(typeof html!=='string'||html.length>2000000||!html.includes(title))throw Error('Wrong Fed calendar month');
 const section=/<h4[^>]*>FOMC Meetings\s*<\/h4>([\s\S]*?)(?=<h4|$)/.exec(html)?.[1];
 if(!section)throw Error('FOMC timing section unavailable');
 const data=[];let rejected=0;
 for(const row of section.matchAll(/<div class="col-xs-2">\s*(<p>[\s\S]*?<\/p>)\s*<\/div>\s*<div class="col-xs-7">([\s\S]*?)<\/div>\s*<div class="col-xs-3">([\s\S]*?)<\/div>/g)){
  try{
   const name=clean(/<p>([\s\S]*?)<\/p>/.exec(row[2])?.[1]??'');
   const kind=name==='FOMC Meeting'?'decision':name==='FOMC Press Conference'?'conference':null;
   if(!kind)throw Error('Unknown FOMC event');
   const clock=/^(\d{1,2}):(\d{2}) ([ap])\.m\.$/.exec(clean(row[1])),date=clean(row[3]);
   if(!clock||!/^\d{1,2}$/.test(date)||+clock[1]<1||+clock[1]>12)throw Error('Unrecognized FOMC time');
   const hour=+clock[1]%12+(clock[3]==='p'?12:0);
   const time=dateTime(`${day.slice(0,7).replace('-','')}${date.padStart(2,'0')}T${String(hour).padStart(2,'0')}${clock[2]}00`,'America/New_York');
   if(time.slice(0,10)<day)continue;
   data.push({event:name,time,kind});
  }catch{rejected++;}
 }
 if(!data.length&&!rejected)throw Error('No usable FOMC timing rows');
 return {data,source:timingSource(day),fetchedAt:new Date(now).toISOString(),rejected};
}
export function freshTiming(payload,at){const age=Date.parse(at)-Date.parse(payload?.fetchedAt);return Number.isFinite(age)&&age>=0&&age<=600000&&payload?.source===timingSource(easternDay(Date.parse(at)))&&payload?.rejected===0&&Array.isArray(payload.data);}
export function meetingTiming(payload,meeting,at){
 if(!freshTiming(payload,at))return false;
 const rows=payload.data.filter(e=>e.time.slice(0,10)===meeting.end);
 const decision=rows.filter(e=>e.kind==='decision'),conference=rows.filter(e=>e.kind==='conference');
 return decision.length===1&&conference.length===1&&Date.parse(conference[0].time)>Date.parse(decision[0].time);
}
