import {sessionFor} from './calendar.js';
import {eastern} from './time.js';
const key='apex-spy-value-profile-v1';
const iso=n=>new Date(n).toISOString();
export function previousSession(day) {
  for(let i=1;i<=15;i++) {
    const s=sessionFor(iso(Date.parse(day+'T12:00:00Z')-i*86400000).slice(0,10),iso(Date.now()));
    if(s.isTradingDay)return s;
  }
  throw new Error('Previous session unavailable');
}
export function profileEvidence(p,session,now=Date.now()) {
  const out={levels:[],reason:'No saved SPY value profile'};
  if(!p)return out;
  try {
    if(!/^\d{4}-\d{2}-\d{2}$/.test(p.target)||iso(Date.parse(p.target+'T12:00:00Z')).slice(0,10)!==p.target)throw new Error('Invalid target session date');
    const prior=previousSession(session.id),at=Date.parse(p.receivedAt);
    if(!session.isTradingDay||p.target!==session.id)throw new Error('Saved profile is for another trading session');
    if(p.date!==prior.id)throw new Error('Profile must cover the previous trading session: '+prior.id);
    if(p.symbol!=='SPY'||p.scope!=='RTH'||p.valueArea!==70)throw new Error('Use SPY regular-session volume profile with 70% value area');
    if(![p.val,p.poc,p.vah].every(n=>typeof n==='number'&&Number.isFinite(n)&&n>0)||p.val>p.poc||p.poc>p.vah||p.val===p.vah)throw new Error('Require positive VAL ≤ POC ≤ VAH, with VAL below VAH');
    if(typeof p.source!=='string'||!p.source.trim()||typeof p.settings!=='string'||!p.settings.trim())throw new Error('Chart source and profile row settings are required');
    if(!Number.isFinite(at)||at<Date.parse(prior.close)||at>now||now>Date.parse(session.close))throw new Error('Profile is unavailable at this time or has expired');
    out.levels=[['VAH',p.vah],['VAL',p.val],['VPOC',p.poc]].map(([kind,value])=>({id:'manual-profile-'+kind,kind,value,as_of:prior.close,available_at:p.receivedAt,valid_until:session.close,source:'User supplied: '+p.source.trim(),method:'SPY '+prior.id+' completed RTH volume profile; 70% value area; '+p.settings.trim()+'; manually transcribed, not independently verified'}));
    out.reason='Manual profile active · '+prior.id+' → '+session.id;
  }catch(e){out.reason=e.message;}
  return out;
}
export function loadProfile(storage,now=Date.now()) {
  try {const p=JSON.parse(storage.getItem(key));return p?{...p,receivedAt:iso(now)}:null;}catch{return null;}
}
export function storeProfile(storage,p) {if(p)storage.setItem(key,JSON.stringify(p));else storage.removeItem(key);}
export function profileForm(saved,onSave,onClear) {
  const el=document.createElement('details');el.dataset.id='value-profile';
  el.innerHTML='<summary>SPY value profile · manual input</summary><p data-status role="status"></p><p>Copy the previous completed SPY regular-session profile from your chart (9:30 ET to session close), using a 70% value area. POC is the price with the most volume. Enter chart row size/settings so the profile can be reproduced. Values are saved in this browser and expire at the target session close.</p><form style="display:flex;flex-wrap:wrap;gap:0.6rem;align-items:end"></form>';
  el.style.cssText='margin-top:0.8rem;font-size:0.75rem';
  const form=el.querySelector('form');
  for(const [name,label,type] of [['target','Use for session','date'],['date','Profile session','date'],['val','VAL','number'],['poc','POC','number'],['vah','VAH','number'],['source','Chart source','text'],['settings','Profile row settings','text']]) {
    const l=document.createElement('label');l.textContent=label+' ';const input=document.createElement('input');input.name=name;input.type=type;input.required=true;input.style.cssText='display:block;width:155px;box-sizing:border-box;padding:0.4rem;background:var(--surface-2);color:var(--text);border:1px solid var(--border-strong);border-radius:4px;color-scheme:dark';
    if(type==='number'){input.step='any';input.min='0.01';}
    if(type==='text')input.maxLength=200;
    input.value=saved?.[name]??(name==='target'?eastern(iso(Date.now())).date:'');l.append(input);form.append(l);
  }
  const save=document.createElement('button');save.type='submit';save.className='api-btn';save.textContent='Save profile';form.append(save);
  const clear=document.createElement('button');clear.type='button';clear.className='api-btn';clear.textContent='Clear profile';form.append(clear);
  form.onsubmit=e=>{e.preventDefault();const p=Object.fromEntries(new FormData(form));for(const k of ['val','poc','vah'])p[k]=Number(p[k]);onSave({...p,symbol:'SPY',scope:'RTH',valueArea:70,receivedAt:iso(Date.now())});};
  clear.onclick=()=>{onClear();for(const input of form.querySelectorAll('input'))input.value='';};
  return el;
}
