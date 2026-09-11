import {FOMC_SOURCE} from './fomcCalendar.js';
import {nyFedSource} from './nyFedCalendar.js';
import {eventMonitor} from './eventMonitor.js';
import {meetingTiming} from './fomcTiming.js';
export const SCHEDULE_POLICY='NY Fed indicators + verified FOMC times v2';
export const SCHEDULE_EXCLUSIONS='Excludes FOMC minutes, speeches, unscheduled announcements, and live headline monitoring. This is not complete macro-event coverage.';
export function scheduledCoverage(feeds,at) {
  const now=Date.parse(at),day=new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date(now));
  const ny=feeds.nyfed,fed=feeds.fomc,reasons=[];
  const fresh=p=>{const age=now-Date.parse(p?.fetchedAt);return Number.isFinite(age)&&age>=0&&age<=600000;};
  if(!fresh(ny)||ny?.source!==nyFedSource(now).url)reasons.push('New York Fed calendar missing, stale, or for a different month.');
  const parsed=eventMonitor(ny,at);
  if(ny?.layoutComplete!==true||ny?.calendarMonth!==day.slice(0,7)||ny?.rejected!==0||!Array.isArray(ny?.data)||parsed.status==='unavailable'||parsed.rejected>0)reasons.push('New York Fed calendar has incomplete date coverage or rejected entries.');
  if(!fresh(fed)||fed?.source!==FOMC_SOURCE)reasons.push('Federal Reserve meeting calendar missing or stale.');
  if(fed?.yearCounts?.[day.slice(0,4)]!==8||fed?.rejected!==0||!Array.isArray(fed?.meetings))reasons.push('Federal Reserve calendar does not contain eight parsed current-year meeting rows without rejections.');
  if(fed?.meetings?.some(m=>m.start<=day&&m.end>=day&&!meetingTiming(feeds.fomcTimes,m,at)))reasons.push('Listed FOMC meeting day: decision and press-conference times are not both verified.');
  const passed=reasons.length===0;
  return {policy:SCHEDULE_POLICY,passed,reasons,exclusions:SCHEDULE_EXCLUSIONS,coverage:passed?[{from:at,through:at,available_at:at,source:SCHEDULE_POLICY,method:'Fresh official calendars with complete parsed date coverage. On listed FOMC meeting days, both decision and conference times must be verified. FOMC timing events remain unresolved through session close, beginning the configured pre-event blackout. Checkpoint-only coverage; all other gates apply.',scope:SCHEDULE_EXCLUSIONS}]:[]};
}
