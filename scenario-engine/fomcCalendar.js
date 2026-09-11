export const FOMC_SOURCE='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm';
const months=['January','February','March','April','May','June','July','August','September','October','November','December'];
const clean=s=>s.replace(/<[^>]*>/g,'').replace(/&nbsp;/g,' ').trim();
function day(year,month,date) {
  const value=`${year}-${String(month).padStart(2,'0')}-${String(date).padStart(2,'0')}`;
  if(!month||!Number.isFinite(Date.parse(value))||new Date(value).toISOString().slice(0,10)!==value)throw Error('Invalid meeting date');
  return value;
}
export function parseFomcCalendar(html,now=Date.now()) {
  if(typeof html!=='string'||html.length>2000000)throw Error('Invalid Fed calendar response');
  const today=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now));
  const year=+today.slice(0,4),meetings=[],yearCounts={};let rejected=0,recognized=0;
  const headings=[...html.matchAll(/<h4\b[^>]*>[\s\S]*?(\d{4}) FOMC Meetings[\s\S]*?<\/h4>/g)];
  for(let i=0;i<headings.length;i++) {
    const y=+headings[i][1];if(y<year||y>year+1)continue;
    const section=html.slice(headings[i].index,headings[i+1]?.index??html.length);
    const rows=[...section.matchAll(/<div class="[^"]*fomc-meeting__month[^"]*">([\s\S]*?)<\/div>\s*<div class="[^"]*fomc-meeting__date[^"]*">([\s\S]*?)<\/div>/g)];
    recognized+=rows.length;yearCounts[y]=rows.length;
    for(const row of rows) {
      try {
        const names=clean(row[1]).split('/'),text=clean(row[2]),m=/^(\d{1,2})(?:-(\d{1,2}))?(\*)?$/.exec(text);
        if(!m||names.length>2)throw Error('Unsupported meeting format');
        const indices=names.map(name=>months.findIndex(v=>v===name||v.slice(0,3)===name)+1);
        const start=day(y,indices[0],+m[1]),end=day(y,indices.at(-1),+(m[2]??m[1]));
        if(end<start||Date.parse(end)-Date.parse(start)>7*86400000)throw Error('Invalid meeting range');
        if(end<today)continue;
        meetings.push({id:`fomc:${start}:${end}`,start,end,projections:!!m[3],timeVerified:false,source:FOMC_SOURCE,available_at:new Date(now).toISOString()});
      }catch{rejected++;}
    }
  }
  if(!recognized)throw Error('Current Fed meeting calendar layout unavailable');
  meetings.sort((a,b)=>a.start.localeCompare(b.start));
  return {meetings,source:FOMC_SOURCE,fetchedAt:new Date(now).toISOString(),rejected,yearCounts,status:meetings.length?'dates-only':'unverified'};
}
export function fomcMonitor(payload,receivedAt) {
  const age=Date.parse(receivedAt)-Date.parse(payload?.fetchedAt);
  if(payload?.source!==FOMC_SOURCE||!Number.isFinite(age)||age<0||age>600000||!Array.isArray(payload.meetings))return {status:'unavailable',meetings:[]};
  return {status:payload.meetings.length?'dates-only':'unverified',meetings:payload.meetings,source:FOMC_SOURCE};
}
