export function predictModel(model,schema,values){
  const encoded=schema.encoded.map(spec=>{
    if(spec.kind==='category')return values[spec.field]===spec.value?1:0;
    const raw=values[spec.field];
    const value=raw===''||raw===null||raw===undefined?spec.default:Number(raw);
    if(!Number.isFinite(value))throw Error('預測輸入必須是有限數值。');
    return (value-spec.mean)/spec.scale;
  });
  return model.intercept+model.cols.reduce((sum,col,i)=>sum+model.coef[i]*encoded[col],0);
}
export function predictOriginal(model,values){
  return model.intercept+model.coef.reduce((sum,b,i)=>sum+b*Number(values[model.names[i]]),0);
}
export function parseCSV(text){
  if(new TextEncoder().encode(text).length>5_000_000)throw Error('CSV 上限為 5 MB。');
  text=text.replace(/^\uFEFF/,'');
  const rows=[];let row=[],field='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(quoted){if(c==='"'){if(text[i+1]==='"'){field+='"';i++;}else quoted=false;}else field+=c;}
    else if(c==='"'){if(field)throw Error('CSV 引號格式錯誤。');quoted=true;}
    else if(c===','){row.push(field);field='';}
    else if(c==='\n'||c==='\r'){
      if(c==='\r'&&text[i+1]==='\n')i++;
      row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);row=[];field='';
      if(rows.length>3001)throw Error('瀏覽器資料上限為 3,000 列。');
    }else field+=c;
  }
  if(quoted)throw Error('CSV 引號未關閉。');
  row.push(field);if(row.some(v=>v.trim()!==''))rows.push(row);
  const headers=rows.shift()||[];
  if(headers.length<2||headers.length>40||headers.some(h=>!h.trim())||new Set(headers).size!==headers.length)throw Error('需有 2–40 個不重複且非空的欄名。');
  if(rows.length<20||rows.length>3000||rows.some(r=>r.length!==headers.length))throw Error('需有 20–3,000 列，且每列欄位數一致。');
  const missing=v=>['','NA','N/A','NaN','nan','null','NULL'].includes(v.trim());
  const numeric=headers.filter((h,i)=>rows.some(r=>!missing(r[i]))&&rows.every(r=>missing(r[i])||Number.isFinite(Number(r[i]))));
  return {headers,rows,numeric};
}
