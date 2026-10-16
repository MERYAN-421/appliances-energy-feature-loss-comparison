import {predictModel,predictOriginal,parseCSV} from './core.mjs';
const $=id=>document.getElementById(id);
const charts=new Map();const cache=new Map();
let view='explorer',problem=null,method=null,k=null,research=null,customCSV=null,customPreview=null,customResult=null,worker=null,timer=null,loadGeneration=0;
const colors=['#6558db','#009b80','#e79b45','#5487c4','#d45d87','#797f90'];
const fmt=(n,d=3)=>n===null||!Number.isFinite(n)?'—':Math.abs(n)>0&&Math.abs(n)<.0001?n.toExponential(2):new Intl.NumberFormat('en-US',{maximumFractionDigits:d}).format(n);
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
function status(text='',error=false){$('status').textContent=text;$('status').classList.toggle('error',error);}
async function loadJSON(name){
  if(!cache.has(name))cache.set(name,fetch(`data/${name}.json`).then(r=>{if(!r.ok)throw Error('資料載入失敗，請重新整理。');return r.json();}).catch(e=>{cache.delete(name);throw e;}));
  return cache.get(name);
}
function table(target,rows,columns){
  const host=$(target);host.replaceChildren();if(!rows.length){host.append(el('p','沒有可顯示的資料。','muted'));return;}
  const keys=columns||Object.keys(rows[0]);const t=el('table');const head=el('thead');const hr=el('tr');
  keys.forEach(key=>hr.append(el('th',key)));head.append(hr);t.append(head);
  const body=el('tbody');rows.forEach(row=>{const tr=el('tr');keys.forEach(key=>tr.append(el('td',typeof row[key]==='number'?fmt(row[key],5):row[key]??'—')));body.append(tr);});t.append(body);host.append(t);
}
function chart(id,type,labels,datasets,options={}){
  charts.get(id)?.destroy();
  const chart=new Chart($(id),{type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:datasets.length>1,labels:{boxWidth:10,font:{size:10}}}},scales:{x:{grid:{color:'#f0f2f8'},ticks:{font:{size:10},maxTicksLimit:8}},y:{grid:{color:'#f0f2f8'},ticks:{font:{size:10}}}},...options}});charts.set(id,chart);
}
function metricCards(target,values){
  const host=$(target);host.replaceChildren();
  values.forEach(([label,value,note],i)=>{const box=el('div',undefined,'metric'+(i===0?' green':''));box.append(el('div',label,'label'),el('strong',value),el('small',note));host.append(box);});
}
async function setView(next){
  view=next;loadGeneration++;const gen=loadGeneration;
  location.hash=next;document.querySelectorAll('.tabs button').forEach(b=>b.classList.toggle('active',b.dataset.view===next));
  $('problem-panel').classList.toggle('hidden',next!=='explorer');$('custom-panel').classList.toggle('hidden',next!=='custom');$('research').classList.toggle('hidden',next!=='research');$('analysis').classList.add('hidden');status();
  try{
    if(next==='research'){research=await loadJSON('research');if(gen!==loadGeneration)return;renderResearch();}
    else if(next==='custom'){problem=customResult;if(problem){method=Object.keys(problem.methods)[0];k=problem.methods[method].best_k;render();}}
    else{status('載入模型與驗證結果…');const loaded=await loadJSON($('problem').value);if(gen!==loadGeneration)return;problem=loaded;method=Object.keys(problem.methods)[0];k=problem.methods[method].best_k;render();status();}
  }catch(e){status(e.message,true);}
}
function render(){
  $('analysis').classList.remove('hidden');
  $('analysis-title').textContent=(view==='custom'?'自訂資料':$('problem').value==='startups'?'50 Startups':'Appliances Energy')+' · 特徵選擇與簡約模型';
  $('split-note').textContent=`${problem.rows.toLocaleString()} 列 · ${problem.features} 個編碼特徵 · ${problem.split}`;
  $('runtime-note').textContent=`${problem.split}；Train ${problem.train_n} 筆、Validation ${problem.val_n} 筆、Test ${problem.test_n} 筆。${view==='custom'?'本次以瀏覽器 Pyodide / scikit-learn 1.6.1 計算。':'預先計算於 Python / scikit-learn 1.9.0。'}新探索模型未經 Conformal 校準，不能沿用原研究區間。`;
  const buttons=$('methods');buttons.replaceChildren();
  Object.keys(problem.methods).forEach(name=>{const b=el('button',name,name===method?'selected':'');b.onclick=()=>{method=name;k=problem.methods[method].best_k;render();};buttons.append(b);});
  const group=problem.methods[method];const models=group.models;const model=models.find(m=>m.k===k)||models[0];k=model.k;
  const ks=$('k-options');ks.replaceChildren();models.forEach(m=>{const b=el('button',`K = ${m.k}${m.k===group.best_k?' ★ Validation 最佳':''}`,(m.k===k?'selected ':'')+(m.k===group.best_k?'sweet':''));b.onclick=()=>{k=m.k;render();};ks.append(b);});
  $('method-note').textContent=`${method} → Top K → OLS。K 是編碼後特徵數，含 State dummy。星號只依 Validation RMSE 選擇，不使用 Test。`;
  const v=model.metrics;
  metricCards('metrics',[['Validation RMSE',fmt(v.RMSE,2),'越低越好'],['Validation R²',fmt(v['R²'],4),'解釋變異比例'],['Validation Adjusted R²',fmt(v['Adjusted R²'],4),'描述性調整指標'],['Validation MAE',fmt(v.MAE,2),'平均絕對誤差'],['Training F-statistic',fmt(model.inference.f,2),`p = ${fmt(model.inference.p,5)}`]]);
  $('active-features').replaceChildren(...model.names.map(name=>el('span',name,'feature-tag')));$('best-label').textContent=`Validation 最佳 K=${group.best_k}`;
  chart('rmse-chart','line',models.map(m=>m.k),[{label:method,data:models.map(m=>m.metrics.RMSE),borderColor:colors[0],backgroundColor:colors[0],tension:.15,pointRadius:4}]);
  chart('r2-chart','line',models.map(m=>m.k),[{label:method,data:models.map(m=>m.metrics['R²']),borderColor:colors[1],backgroundColor:colors[1],tension:.15,pointRadius:4}]);
  $('equation').textContent=`${problem.target} = ${fmt(model.intercept,6)} `+model.names.map((name,i)=>`${model.coef[i]>=0?'+':'−'} ${fmt(Math.abs(model.coef[i]),6)} × z[${name}]`).join(' ');
  if(model.inference.error){$('inference').replaceChildren(el('p',model.inference.error,'note'));}else table('inference',model.inference.table);
  buildInputs('predict-inputs',problem.schema.inputs);
  $('prediction-value').textContent='—';$('prediction-note').textContent='輸入值更新後按「計算預測」';
  $('predict-form').onsubmit=e=>{e.preventDefault();try{const values=readInputs('predict-inputs');const value=predictModel(model,problem.schema,values);$('prediction-value').textContent=`${fmt(value,2)} ${unit()}`;$('prediction-note').textContent='以已擬合模型推論，未重新訓練；此模型沒有校準的預測區間。';}catch(e){status(e.message,true);}};
  $('comparison').classList.add('hidden');renderDiagnostics();
}
function unit(){return view==='custom'?problem.target:$('problem').value==='startups'?'$':'Wh';}
function buildInputs(id,specs){
  const host=$(id);host.replaceChildren();specs.forEach(spec=>{const label=el('label',spec.name);let input;
    if(spec.kind==='category'){input=el('select');spec.options.forEach(option=>{const o=el('option',String(option));o.value=JSON.stringify(option);input.append(o);});input.value=JSON.stringify(spec.default);}
    else{input=el('input');input.type='number';input.step='any';input.required=true;input.value=Number(Number(spec.default).toFixed(8));}
    input.dataset.field=spec.name;input.setAttribute('aria-label',spec.name);label.append(input);host.append(label);
  });
}
function readInputs(id){const values=Object.create(null);$(id).querySelectorAll('input,select').forEach(input=>{values[input.dataset.field]=input.type==='number'?Number(input.value):JSON.parse(input.value);});return values;}
function renderDiagnostics(){
  if(!problem)return;const group=problem.methods[method];const model=group.models.find(m=>m.k===k);const test=$('eval-split').value==='test';const d=test?group.test.diagnostics:model.diagnostics;
  $('diagnostic-note').textContent=`${test?'Test（驗證選出的 K='+group.best_k+'）':'Validation（目前 K='+k+'）'} · 完整評估集 ${d.n} 筆`;
  $('test-metrics').textContent=test?Object.entries(group.test.metrics).map(([name,value])=>`${name}: ${fmt(value,4)}`).join('  ·  '):'';
  chart('actual-chart','scatter',[],[{label:'樣本',data:d.points.map(p=>({x:p[0],y:p[1]})),backgroundColor:'#009b8080',pointRadius:3}]);
  chart('residual-chart','bar',d.histogram.x.map(x=>fmt(x,1)),[{label:'次數',data:d.histogram.y,backgroundColor:'#6558db90'}]);
  chart('residual-scatter','scatter',[],[{label:'殘差',data:d.points.map(p=>({x:p[1],y:p[2]})),backgroundColor:'#5487c480',pointRadius:3}]);table('errors',d.top);
}
function compare(){
  $('comparison').classList.remove('hidden');const entries=Object.entries(problem.methods);const labels=entries[0][1].models.map(m=>m.k);
  chart('compare-chart','line',labels,entries.map(([name,group],i)=>({label:name,data:group.models.map(m=>m.metrics.RMSE),borderColor:colors[i%colors.length],backgroundColor:colors[i%colors.length],pointRadius:3})));
  table('comparison-table',entries.flatMap(([name,group])=>group.models.map(m=>({演算法:name,K:m.k,...m.metrics}))));
}
function renderResearch(){
  const meta=research.metadata,tm=meta.test_metrics;
  metricCards('research-kpis',[['最佳模型',meta.best_model_name,'Validation RMSE 選模'],['Test RMSE',fmt(tm.RMSE,3),'Wh'],['Test MAE',fmt(tm.MAE,3),'Wh'],['Test R²',fmt(tm.R2,4),'獨立測試集'],['95% PI Coverage',fmt(tm.interval_coverage*100,2)+'%','名義覆蓋目標 95%']]);
  researchRanking();buildInputs('original-inputs',research.original.names.map((name,i)=>({name,kind:'number',default:research.original.defaults[i]})));
  $('original-form').onsubmit=e=>{e.preventDefault();const pred=predictOriginal(research.original,readInputs('original-inputs'));const w=research.original.half_width;$('original-value').textContent=fmt(pred,2)+' Wh';$('original-interval').textContent=`95% Conformal 預測區間：[${fmt(Math.max(0,pred-w),2)}, ${fmt(pred+w,2)}] Wh`;};
  const data=research.predictions;
  chart('interval-chart','line',data.map(d=>d.date),[
    {label:'95% PI 上界',data:data.map(d=>d.upper_95PI),borderWidth:0,pointRadius:0,backgroundColor:'#6558db10'},
    {label:'95% PI 下界',data:data.map(d=>Math.max(0,d.lower_95PI)),borderWidth:0,pointRadius:0,fill:'-1',backgroundColor:'#6558db20'},
    {label:'實際值',data:data.map(d=>d.actual),borderColor:colors[1],borderWidth:1,pointRadius:0},
    {label:'預測值',data:data.map(d=>d.predicted),borderColor:colors[0],borderWidth:2,pointRadius:0}]);table('research-errors',research.diagnostics.top);
}
function researchRanking(){
  if(!research)return;const metric=$('research-metric').value;const sorted=[...research.validation].sort((a,b)=>metric==='R2'?b[metric]-a[metric]:a[metric]-b[metric]);
  chart('research-chart','bar',sorted.slice(0,10).map(r=>`${r.feature_selection} × ${r.loss_function}`),[{label:'Validation '+metric,data:sorted.slice(0,10).map(r=>r[metric]),backgroundColor:colors[0]}],{indexAxis:'y',responsive:true,maintainAspectRatio:false,animation:false,plugins:{legend:{display:false}}});table('research-table',sorted);
}
function configureCSV(text){
  stopWorker();customCSV=text;customPreview=parseCSV(text);customResult=null;problem=null;$('analysis').classList.add('hidden');
  $('dataset-config').classList.remove('hidden');const preview=customPreview;
  table('preview',preview.rows.slice(0,6).map(row=>Object.fromEntries(preview.headers.map((h,i)=>[h,row[i]]))),preview.headers);
  const target=$('target');target.replaceChildren(...preview.numeric.map(name=>{const o=el('option',name);o.value=name;return o;}));
  if(!preview.numeric.length)throw Error('資料沒有可用的數值目標欄位。');
  target.value=preview.numeric.includes('Profit')?'Profit':preview.numeric.at(-1);renderPredictorOptions();status(`已載入 ${preview.rows.length} 列、${preview.headers.length} 欄。選擇目標與預測欄位後開始分析。`);
}
function renderPredictorOptions(){
  $('predictor-options').replaceChildren(...customPreview.headers.filter(h=>h!==$('target').value).map(name=>{const label=el('label');const input=el('input');input.type='checkbox';input.checked=true;input.value=name;label.append(input,document.createTextNode(' '+name));return label;}));
  customResult=null;$('analysis').classList.add('hidden');
}
async function loadURL(){
  try{
    const u=new URL($('csv-url').value);
    if(u.protocol!=='https:'||u.hostname!=='raw.githubusercontent.com'||u.username||u.password||u.port||!u.pathname.toLowerCase().endsWith('.csv')||u.hash)throw Error('僅允許 raw.githubusercontent.com 的公開 HTTPS CSV 網址。');
    status('載入公開 CSV…');const response=await fetch(u,{redirect:'error',signal:AbortSignal.timeout(20000)});if(!response.ok)throw Error(`HTTP ${response.status}`);
    const reader=response.body.getReader();let bytes=0,chunks=[];
    while(true){const {done,value}=await reader.read();if(done)break;bytes+=value.byteLength;if(bytes>5_000_000){await reader.cancel();throw Error('CSV 上限為 5 MB。');}chunks.push(value);}
    const all=new Uint8Array(bytes);let offset=0;for(const chunk of chunks){all.set(chunk,offset);offset+=chunk.length;}configureCSV(new TextDecoder('utf-8',{fatal:true}).decode(all));
  }catch(e){status('載入失敗：'+e.message,true);}
}
function setBusy(busy){
  for(const id of ['train-custom','csv-file','csv-url','load-url','target'])$(id).disabled=busy;
  $('predictor-options').querySelectorAll('input').forEach(input=>input.disabled=busy);
}
function stopWorker(){if(worker)worker.terminate();worker=null;clearTimeout(timer);setBusy(false);}
function trainCustom(){
  const predictors=[...$('predictor-options').querySelectorAll('input:checked')].map(i=>i.value);
  if(!predictors.length){status('請至少選擇一個預測欄位。',true);return;}
  setBusy(true);status('準備瀏覽器分析環境…');
  if(!worker){worker=new Worker('worker.js');worker.onmessage=({data})=>{
    if(data.type==='status'){if(view==='custom')status(data.text);return;}
    clearTimeout(timer);setBusy(false);
    if(data.type==='error'){status('分析失敗：'+data.text,true);return;}
    customResult=data.payload;
    if(view==='custom'){problem=customResult;method=Object.keys(problem.methods)[0];k=problem.methods[method].best_k;render();status('分析完成。模型僅保留在目前瀏覽器，沒有上傳。');}
  };worker.onerror=e=>{stopWorker();status('運算元件載入失敗，請檢查 CDN 連線後重試。',true);};}
  clearTimeout(timer);timer=setTimeout(()=>{stopWorker();status('運算超過 180 秒已停止。請縮小資料或減少特徵後重試。',true);},180000);
  worker.postMessage({csv:customCSV,target:$('target').value,predictors});
}
document.querySelectorAll('.tabs button').forEach(button=>button.onclick=()=>setView(button.dataset.view));
$('problem').onchange=()=>setView('explorer');$('eval-split').onchange=renderDiagnostics;$('compare-methods').onclick=compare;$('research-metric').onchange=researchRanking;
$('csv-file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;if(file.size>5_000_000)throw Error('CSV 上限為 5 MB。');configureCSV(new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer()));}catch(e){status(e.message,true);}};
$('load-url').onclick=loadURL;$('target').onchange=renderPredictorOptions;$('predictor-options').onchange=()=>{customResult=null;$('analysis').classList.add('hidden');};$('train-custom').onclick=trainCustom;
$('cancel-custom').onclick=()=>{stopWorker();customCSV=null;customPreview=null;customResult=null;problem=null;$('csv-file').value='';for(const id of ['preview','target','predictor-options','predict-inputs','inference','errors','comparison-table','metrics','active-features'])$(id).replaceChildren();for(const c of charts.values())c.destroy();charts.clear();$('dataset-config').classList.add('hidden');$('analysis').classList.add('hidden');status('自訂資料與模型已清除。');};
setView(['explorer','custom','research'].includes(location.hash.slice(1))?location.hash.slice(1):'explorer');
