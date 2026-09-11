export const ms = value => Date.parse(value);
export const iso = value => new Date(value).toISOString();
export const minutesBetween = (a, b) => (ms(a) - ms(b)) / 60000;
const formatter = new Intl.DateTimeFormat('en-CA', {
  timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', hourCycle: 'h23'
});
export function eastern(value) {
  const p = Object.fromEntries(formatter.formatToParts(new Date(value)).map(x => [x.type, x.value]));
  return { date: `${p.year}-${p.month}-${p.day}`, minute: Number(p.hour) * 60 + Number(p.minute) };
}
export function phase(at, session) {
  if (!session.isTradingDay || ms(at) >= ms(session.close)) return 'closed';
  if (ms(at) < ms(session.open)) return 'premarket';
  const m = eastern(at).minute;
  if (m < 585) return 'opening';
  if (m < 630) return 'confirmation';
  if (m < 720) return 'morning';
  if (m < 840) return 'midday';
  if (m < 900) return 'afternoon';
  return 'power-hour';
}
