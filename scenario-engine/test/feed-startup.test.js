import test from 'node:test';
import assert from 'node:assert/strict';

test('candle timeout does not prevent independent feed requests',async()=>{
  const requests=[];
  const original=globalThis.fetch;
  globalThis.window={};
  globalThis.document={getElementById:()=>null};
  globalThis.fetch=async url=>{requests.push(url);return {ok:true,json:async()=>({status:'received',source:'Unusual Whales',results:[]})};};
  try {
    const {update}=await import('../apex-adapter.js?startup-test');
    const read={bias:{dir:'NEUTRAL'},gate:'NO-GO',reasons:[],stops:{},setup:{status:'WAIT'}};
    const result=update(read,{valid:false,reason:'Candle feed timed out',bars:[],metrics:null});
    await new Promise(resolve=>setTimeout(resolve,0));
    assert.ok(requests.some(url=>url.endsWith('/api/scenario/spy-contracts')));
    assert.ok(requests.some(url=>url.endsWith('/api/scenario/calendars')));
    assert.ok(requests.some(url=>url.includes('/api/stock/SPY?')));
    assert.equal(result.gate,'NO-GO');
    assert.equal(result.setup.status,'WAIT');
  } finally {globalThis.fetch=original;delete globalThis.window;delete globalThis.document;}
});
