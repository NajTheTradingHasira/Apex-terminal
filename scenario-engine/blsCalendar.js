export const BLS_SOURCE='https://www.bls.gov/schedule/news_release/bls.ics';
export function dateTime(value,zone) {
  const m=/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})(Z?)$/.exec(value);
  if(!m)throw Error('Unsupported event date');
  const [,y,mo,d,h,mi,se,z]=m,base=Date.UTC(+y,+mo-1,+d,+h,+mi,+se);
  if(new Date(base).toISOString().replace(/[-:]/g,'').slice(0,15)!==`${y}${mo}${d}T${h}${mi}${se}`)throw Error('Invalid event date');
  if(z)return new Date(base).toISOString();
  if(!['US-Eastern','America/New_York'].includes(zone))throw Error('Unsupported calendar timezone');
  const parts=t=>Object.fromEntries(new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).formatToParts(new Date(t)).map(p=>[p.type,p.value]));
  const p=parts(base),local=Date.UTC(+p.year,+p.month-1,+p.day,+p.hour,+p.minute,+p.second),adjusted=base+(base-local),check=parts(adjusted);
  if([check.year,check.month,check.day,check.hour,check.minute,check.second].join('')!==[y,mo,d,h,mi,se].join(''))throw Error('Ambiguous or invalid event time');
  return new Date(adjusted).toISOString();
}
export function parseBlsCalendar(text,now=Date.now()) {
  if(typeof text!=='string'||text.length>2000000||!text.includes('BEGIN:VCALENDAR')||!text.includes('END:VCALENDAR'))throw Error('Invalid BLS calendar response');
  const unfolded=text.replace(/\r?\n[ \t]/g,''),data=[];let rejected=0;
  for(const match of unfolded.matchAll(/BEGIN:VEVENT\r?\n([\s\S]*?)END:VEVENT/g)) {
    try {
      const lines=match[1].split(/\r?\n/),get=name=>lines.find(l=>l.startsWith(name+':'))?.slice(name.length+1);
      if(lines.some(l=>l.startsWith('RRULE:')))throw Error('Recurring event unsupported');
      if(get('STATUS')==='CANCELLED')continue;
      const uid=get('UID'),name=get('SUMMARY'),start=lines.find(l=>l.startsWith('DTSTART'));
      if(!uid||!name||!start)throw Error('Missing event fields');
      const split=start.indexOf(':'),params=start.slice(0,split),value=start.slice(split+1),zone=/TZID=([^;:]+)/.exec(params)?.[1];
      const time=dateTime(value,zone),t=Date.parse(time);
      if(t<now-86400000||t>now+45*86400000)continue;
      data.push({uid,event:name.replace(/\\n/gi,' ').replace(/\\([,;\\])/g,'$1'),time});
    }catch{rejected++;}
  }
  data.sort((a,b)=>Date.parse(a.time)-Date.parse(b.time));
  return {data,source:BLS_SOURCE,fetchedAt:new Date(now).toISOString(),rejected,
    status:data.length?'partial':'unverified',limitations:'BLS releases only. No claim of full macro-event or headline coverage. Cancellations and schedule changes require conservative reevaluation.'};
}
