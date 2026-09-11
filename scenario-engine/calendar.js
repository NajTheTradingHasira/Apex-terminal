import {eastern} from './time.js';
const calendarSource = 'NYSE published calendar https://www.nyse.com/trade/hours-calendars (checked 2026-09-07)';
const holidays = {
  2026: ['01-01','01-19','02-16','04-03','05-25','06-19','07-03','09-07','11-26','12-25'],
  2027: ['01-01','01-18','02-15','03-26','05-31','06-18','07-05','09-06','11-25','12-24'],
  2028: ['01-17','02-21','04-14','05-29','06-19','07-04','09-04','11-23','12-25'],
};
const early = new Set(['2026-11-27','2026-12-24','2027-11-26','2028-07-03','2028-11-24']);
function etTime(day, hour, minute=0) {
  const noon = Date.parse(`${day}T12:00:00Z`);
  const offset = 12 - Math.floor(eastern(new Date(noon).toISOString()).minute / 60);
  return new Date(Date.parse(`${day}T00:00:00Z`) + ((hour+offset)*60+minute)*60000).toISOString();
}
export function sessionFor(day, receivedAt) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day) || !holidays[day.slice(0,4)]) throw new Error('Exchange calendar unavailable for this year');
  const weekday = new Date(`${day}T12:00:00Z`).getUTCDay();
  return { id:day, open:etTime(day,9,30), close:etTime(day,early.has(day)?13:16),
    isTradingDay:weekday!==0 && weekday!==6 && !holidays[day.slice(0,4)].includes(day.slice(5)),
    available_at:receivedAt, source:calendarSource };
}
