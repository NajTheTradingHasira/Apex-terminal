import {ScenarioEngine} from './engine.js';

export const decisionKey=c=>JSON.stringify([c.scenario_id,c.directional_bias,c.decision,c.trade_permitted,c.decision_reasons,c.invalidation,c.target_ladder]);
export function engineState(engine) {
  return structuredClone({active:engine.active,lastAt:engine.lastAt,lastCard:engine.lastCard,sessionId:engine.sessionId,history:engine.history});
}
export function replayRecord(record) {
  const engine=new ScenarioEngine(record.config);
  Object.assign(engine,structuredClone(record.before));
  const actual=engine.evaluate(record.dataset,record.at);
  return {at:record.at,match:decisionKey(actual)===decisionKey(record.card),actual};
}

/** IndexedDB stores exact checkpoint inputs; summary rows alone are not replay data. */
export class SessionHistory {
  constructor(){this.rows=[];this.status='Opening local history';this.lastBucket=null;this.db=null;this.ready=this.open();}
  async open() {
    if(typeof indexedDB==='undefined'){this.status='History storage unavailable';return;}
    try {
      this.db=await new Promise((resolve,reject)=>{
        const req=indexedDB.open('apex-spy-checkpoints',1);
        req.onupgradeneeded=()=>req.result.createObjectStore('records',{keyPath:'id'});
        req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error);
      });
      this.rows=await this.read();this.status='Saved locally in this browser';
    }catch{this.status='History storage unavailable';}
  }
  async read() {
    if(!this.db)return [];
    return new Promise((resolve,reject)=>{
      const r=this.db.transaction('records').objectStore('records').getAll();
      r.onsuccess=()=>resolve(r.result.sort((a,b)=>a.at.localeCompare(b.at)));r.onerror=()=>reject(r.error);
    });
  }
  capture(record) {
    const bucket=record.dataset.session.id+':'+Math.floor(Date.parse(record.at)/60000);
    if(bucket===this.lastBucket||!record.dataset.bars.length)return;
    this.lastBucket=bucket;
    // Copy before awaiting storage: live state may advance while the write is queued.
    const copy=structuredClone({...record,id:bucket,version:1});
    this.ready.then(()=>{
      if(!this.db)return;
      const tx=this.db.transaction('records','readwrite'),store=tx.objectStore('records');
      store.put(copy);
      // Keep at most 500 checkpoints (roughly one full session) across reloads.
      const rows=[...this.rows.filter(r=>r.id!==copy.id),copy].sort((a,b)=>a.at.localeCompare(b.at));
      for(const r of rows.slice(0,Math.max(0,rows.length-500)))store.delete(r.id);
      tx.oncomplete=()=>{this.rows=rows.slice(-500);this.status='Saved locally in this browser';};
      tx.onerror=()=>{this.status='History write failed — storage may be full';};
    }).catch(()=>{this.status='History write failed';});
  }
  async verify() {
    await this.ready;
    const rows=await this.read();
    let matched=0;
    for(const row of rows)if(replayRecord(row).match)matched++;
    return {checked:rows.length,matched};
  }
  async download() {
    const rows=await this.read();
    const blob=new Blob([JSON.stringify({version:1,method:'Exact saved checkpoint inputs and pre-evaluation state. Rule reproducibility only; no fill or profitability simulation.',records:rows})],{type:'application/json'});
    const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='apex-spy-history.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
}
