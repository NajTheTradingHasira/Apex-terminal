import {dateTime} from './blsCalendar.js';
const months=['January','February','March','April','May','June','July','August','September','October','November','December'];
export function nyFedSource(now=Date.now()) {
  const parts=Object.fromEntries(new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',year:'numeric',month:'numeric'}).formatToParts(new Date(now)).map(p=>[p.type,p.value]));
  return {year:+parts.year,month:+parts.month,url:`https://www.newyorkfed.org/research/calendars/i-${months[+parts.month-1].slice(0,3).toLowerCase()}${parts.year.slice(-2)}.html`};
}
function plain(value) {
  return value.replace(/<[^>]*>/g,'').replace(/&amp;/g,'&').replace(/&nbsp;/g,' ').replace(/&#39;|&apos;/g,"'").replace(/&quot;/g,'"').trim();
}
export function parseNyFedCalendar(html,now=Date.now()) {
  const {year,month,url}=nyFedSource(now);
  if(typeof html!=='string'||html.length>2000000||!html.includes(`${months[month-1]} ${year}`)||!html.includes('all Eastern Time'))throw Error('Unrecognized New York Fed calendar');
  const table=/<table\b[^>]*class="research-table-1col greyborder"[^>]*>([\s\S]*?)<\/table>/i.exec(html)?.[1];
  if(!table)throw Error('New York Fed calendar layout unavailable');
  const data=[],parsedDays=[];let rejected=0,days=0;
  for(const cell of table.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/gi)) {
    const day=/<div>\s*(\d{2})\s*(?:<|$)/.exec(cell[1])?.[1];
    if(!day)continue;days++;parsedDays.push(Number(day));
    const content=/<span class="ts-accordion-content">([\s\S]*?)<\/span>/.exec(cell[1])?.[1];
    if(!content)continue;
    // Bind each time only to its own anchor segment, never to a later event.
    for(const item of content.matchAll(/<a\b[^>]*>([\s\S]*?)<\/a>((?:(?!<a\b)[\s\S])*)/gi)) {
      try {
        const name=plain(item[1]),times=[...plain(item[2]).matchAll(/\((\d{2}):(\d{2})\)/g)];
        if(!name||name.length>500||times.length!==1)throw Error('Unrecognized release time');
        const time=dateTime(`${year}${String(month).padStart(2,'0')}${day}T${times[0][1]}${times[0][2]}00`,'America/New_York');
        if(Date.parse(time)<now-86400000)continue;
        data.push({event:name,time});
      }catch{rejected++;}
    }
  }
  if(!days)throw Error('New York Fed calendar contains no dated cells');
  const expectedDays=[];
  for(let d=1;d<=new Date(Date.UTC(year,month,0)).getUTCDate();d++){const weekday=new Date(Date.UTC(year,month-1,d)).getUTCDay();if(weekday!==0&&weekday!==6)expectedDays.push(d);}
  const layoutComplete=parsedDays.length===expectedDays.length&&expectedDays.every(d=>parsedDays.filter(v=>v===d).length===1);
  return {data,source:url,fetchedAt:new Date(now).toISOString(),rejected,calendarMonth:`${year}-${String(month).padStart(2,'0')}`,layoutComplete,status:data.length?'partial':'unverified',limitations:'Current-month economic indicators only; does not certify full macro, FOMC, or headline coverage.'};
}
