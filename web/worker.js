/* Custom datasets stay in this browser worker; no server receives the CSV. */
let runtime;
async function initialize(){
  if(runtime) return runtime;
  runtime=(async()=>{
    self.postMessage({type:'status',text:'首次啟動：下載瀏覽器 Python 科學運算元件…'});
    importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');
    const py=await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/'});
    await py.loadPackage(['numpy','pandas','scipy','scikit-learn']);
    py.FS.mkdirTree('/site/utils');py.FS.writeFile('/site/utils/__init__.py','');
    for(const [url,path] of [['python/regression.py','/site/utils/regression.py'],['python/pages_payload.py','/site/pages_payload.py']]){
      const res=await fetch(new URL(url,self.location.href));if(!res.ok)throw Error('無法載入分析模組');
      py.FS.writeFile(path,await res.text());
    }
    await py.runPythonAsync("import sys\nsys.path.insert(0,'/site')\nimport json\nfrom utils.regression import parse_csv\nfrom pages_payload import build_problem");
    return py;
  })();
  return runtime;
}
self.onmessage=async({data})=>{
  try{
    const py=await initialize();
    self.postMessage({type:'status',text:'在瀏覽器內訓練模型、計算驗證曲線與統計量…'});
    py.globals.set('request_json',JSON.stringify(data));
    const result=await py.runPythonAsync(`
request = json.loads(request_json)
df = parse_csv(request['csv'].encode('utf-8'))
if len(df) > 3000:
    raise ValueError('瀏覽器資料上限為 3,000 列。')
selected = request['predictors'] + [request['target']]
result = build_problem(df[selected], request['target'], False,
    ['Lasso','RFE','ElasticNet','F-test','Forward SFS','Exhaustive Search'], browser=True)
json.dumps(result, ensure_ascii=False, allow_nan=False)
`);
    self.postMessage({type:'result',payload:JSON.parse(result)});
    await py.runPythonAsync("del df, result, request, request_json\nimport gc\ngc.collect()");
  }catch(error){self.postMessage({type:'error',text:String(error.message||error)});}
};
