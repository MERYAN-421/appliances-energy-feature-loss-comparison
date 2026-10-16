"""JSON bridge shared by the offline Pages exporter and the browser Python worker."""
import json
import numpy as np
import pandas as pd
from utils.regression import prepare, fit_model, inference, metrics, diagnostics


def clean(value):
    if isinstance(value, dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value,(np.integer,)):
        return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    return value


def diagnostic_payload(y,pred,indices):
    d=diagnostics(y,pred)
    d.insert(0,'資料列位置',indices)
    counts,edges=np.histogram(d['殘差'],bins=min(40,max(5,len(d)//2)))
    sample=d.iloc[np.linspace(0,len(d)-1,min(300,len(d)),dtype=int)]
    return clean({'points':sample[['實際值','預測值','殘差']].values,
                  'top':d.sort_values('絕對誤差',ascending=False).head(25).to_dict('records'),
                  'histogram':{'x':(edges[:-1]+edges[1:])/2,'y':counts},'n':len(d)})


def schema(ctx):
    pre=ctx['pre']; train=ctx['X'].iloc[ctx['ti']]
    inputs=[]; encoded=[]
    for kind,transformer,names in pre.transformers_:
        if kind=='remainder': continue
        imp=transformer.named_steps['simpleimputer']
        if kind=='numeric':
            scaler=transformer.named_steps['standardscaler']
            for name,default,mean,scale in zip(names,imp.statistics_,scaler.mean_,scaler.scale_):
                inputs.append({'name':name,'kind':'number','default':default,'min':train[name].min(),'max':train[name].max()})
                encoded.append({'field':name,'kind':'number','mean':mean,'scale':scale,'default':default})
        else:
            encoder=transformer.named_steps['onehotencoder']
            for name,default,categories in zip(names,imp.statistics_,encoder.categories_):
                inputs.append({'name':name,'kind':'category','default':default,'options':list(categories)})
                for category in categories[1:]:
                    encoded.append({'field':name,'kind':'category','value':category})
    return clean({'inputs':inputs,'encoded':encoded})


def model_payload(ctx,result,k):
    model=result['model']; cols=result['cols']
    try:
        table,f,p=inference(ctx['train'][:,cols],ctx['y'][ctx['ti']],result['names'],model)
        stats={'table':table.to_dict('records'),'f':f,'p':p,'error':None}
    except (ValueError,np.linalg.LinAlgError) as exc:
        stats={'table':[],'f':None,'p':None,'error':str(exc)}
    return clean({'k':k,'metrics':result['metrics'],'names':result['names'],'cols':cols,
                  'intercept':model.intercept_,'coef':model.coef_,'inference':stats,
                  'diagnostics':diagnostic_payload(ctx['y'][ctx['vi']],result['predictions'],ctx['vi'])})


def build_problem(df,target,chronological,methods,browser=False):
    ctx=prepare(df,target,chronological)
    p=len(ctx['names'])
    if browser and (len(df)>3000 or p>20):
        raise ValueError('瀏覽器運算上限：3,000 列、20 個編碼後特徵。')
    ks=list(range(1,p+1)) if p<=8 else sorted(set([1,3,5,10,15,20,25,p]) & set(range(1,p+1)))
    output={'target':target,'rows':len(df),'features':p,'schema':schema(ctx),'methods':{},
            'split':'時間順序 55% / 15% / 15% Calibration / 15% Test' if chronological else '固定種子 42：Train 60% / Validation 20% / Test 20%',
            'train_n':len(ctx['ti']),'val_n':len(ctx['vi']),'test_n':len(ctx['test']),
            'preview':df.head(6).to_dict('records')}
    for method in methods:
        if method in ['Forward SFS','Exhaustive Search'] and (p>8 or len(ctx['train'])>1000):
            continue
        fits=[fit_model(ctx,method,k) for k in ks]
        winner=min(range(len(fits)),key=lambda i:fits[i]['metrics']['RMSE'])
        best=fits[winner]
        yp=best['model'].predict(ctx['pre'].transform(ctx['X'].iloc[ctx['test']])[:,best['cols']])
        output['methods'][method]={'models':[model_payload(ctx,r,k) for r,k in zip(fits,ks)],'best_k':ks[winner],
            'test':{'k':ks[winner],'metrics':metrics(ctx['y'][ctx['test']],yp,ks[winner]),'diagnostics':diagnostic_payload(ctx['y'][ctx['test']],yp,ctx['test'])}}
    return clean(output)
