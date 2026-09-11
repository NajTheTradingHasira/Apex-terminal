import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const html=fs.readFileSync(new URL('./index.html',import.meta.url),'utf8');
const section=html.slice(html.indexOf('let breadthChecking=false;'),html.indexOf('// ── 8. BOOT SEQUENCE'));
async function check(response,pattern,archive=false){
 const c=vm.createContext({breadthConnection:'',breadthChecked:null,breadthArchive:'',NEXUS_API:'https://test',AbortController,AbortSignal,setTimeout,clearTimeout,Date,Intl,buildBreadthMetrics(){},fetch:async()=>response});
 await vm.runInContext(section+`;${archive?'fetchRegimeScan':'fetchBreadthData'}()`,c);
 assert.match(archive?c.breadthArchive:c.breadthConnection,pattern);
}
await check({ok:false,json:async()=>({detail:'Finviz login failed: HTTP 400'})},/HTTP 400/);
await check({ok:true,json:async()=>({results:[],count:0})},/not a complete breadth feed/);
await check({ok:true,json:async()=>({advancing:1847})},/incomplete/);
await check({ok:true,json:async()=>({date:'2026-04-06',time:'21:46',regime_label:'BUY'})},/STALE.*excluded/,true);
assert.ok(!html.includes('advancing: 1847'));
assert.ok(!html.includes('BD.newHighs = d.new_highs'));
assert.match(html,/ctx.breadth = \{status:'unavailable'/);
console.log('PASS: seven breadth integrity checks');
