(async()=>{
const db=await new Promise((res,rej)=>{const r=indexedDB.open('kx',1);r.onupgradeneeded=()=>r.result.createObjectStore('kv');r.onsuccess=()=>res(r.result);r.onerror=()=>rej(r.error)});
const get=k=>new Promise(res=>{const q=db.transaction('kv').objectStore('kv').get(k);q.onsuccess=()=>res(q.result)});
const put=(k,v)=>new Promise(res=>{const t=db.transaction('kv','readwrite');t.objectStore('kv').put(v,k);t.oncomplete=()=>res()});
const qn=sessionStorage.q||'a', QK='Q:'+qn;
let st=await get(QK); if(!st||!st.queue.length) return qn+' empty';
const cur=st.queue[0];
const norm=u=>new URL(u,location.origin).href;
if(document.readyState!=='complete'||location.href!==norm(cur.u)){ location.href=cur.u; return qn+' nav '+st.queue.length; }
let j; try{ j=JSON.parse(document.body.innerText) }catch(e){ st.err=(st.err||0)+1; await put(QK,st); setTimeout(()=>location.reload(),1500); return qn+' retry'; }
const f=x=>x==null||x===''?null:+x;
const feat=(tk,cs,occ)=>{
  const K=Date.parse(occ)/1000-3*3600;
  const n=cs.map(c=>{const P=c.price||{},B=c.yes_bid||{},A=c.yes_ask||{};
    const p=f(P.close_dollars??P.close),b=f(B.close_dollars??B.close),a=f(A.close_dollars??A.close),v=f(c.volume_fp??c.volume)||0;
    let m=null; if(b!=null&&a!=null&&b>0&&a<1&&a>=b&&a-b<=0.10) m=(a+b)/2; else if(p!=null&&v>0) m=p;
    return {t:c.end_period_ts,m,p,v};});
  const pre=n.filter(c=>c.t<=K);
  const s=[]; for(let h=192;h>=0;h-=3){const T=K-h*3600; let val=null; for(const c of pre){ if(c.t<=T&&c.m!=null) val=c.m; } s.push(val==null?null:Math.round(val*1000)/10);}
  const fv=pre.find(c=>c.m!=null); let lt=null; for(const c of pre) if(c.p!=null&&c.v>0) lt=c.p;
  return {tk,K,s,f:fv?[Math.round(fv.m*1000)/10,fv.t]:null,lt,pv:Math.round(pre.reduce((x,c)=>x+c.v,0)),tv:Math.round(n.reduce((x,c)=>x+c.v,0))};
};
let msg='';
if(cur.k==='L'){
  const ms=(j.markets||[]).map(m=>[m.ticker,m.event_ticker,m.yes_sub_title,m.result,m.occurrence_datetime||m.expected_expiration_time,m.open_time,f(m.volume_fp??m.volume),m.close_time,cur.s]);
  await put('M:'+cur.s+':'+(cur.h?'h':'l')+':'+(cur.pg||0),ms);
  st.queue.shift();
  if(j.cursor&&ms.length){ const base=cur.u.replace(/&cursor=[^&]*/,''); st.queue.unshift({...cur,u:base+'&cursor='+encodeURIComponent(j.cursor),pg:(cur.pg||0)+1}); }
  msg='L '+cur.s+' '+ms.length;
}else if(cur.k==='B'){
  for(const m of (j.markets||[])){ await put('F:'+m.market_ticker,feat(m.market_ticker,m.candlesticks,cur.occ[m.market_ticker])); }
  st.queue.shift(); msg='B '+(j.markets||[]).length;
}else if(cur.k==='H'){
  if(j.candlesticks) await put('F:'+cur.tk,feat(cur.tk,j.candlesticks,cur.occ)); else await put('E:'+cur.tk,JSON.stringify(j).slice(0,200));
  st.queue.shift(); msg='H';
}
st.done=(st.done||0)+1; await put(QK,st);
if(st.queue.length) location.href=st.queue[0].u;
return qn+' '+msg+' left '+st.queue.length;
})()
