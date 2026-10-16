"""Build public static artifacts without changing src/ or outputs/."""
import json
import hashlib
import sklearn
import sys
from pathlib import Path
import shutil
import joblib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from pages_payload import build_problem, clean, diagnostic_payload
from utils.regression import METHODS

OUT=ROOT/'web'

def write(name,obj):
    (OUT/'data'/name).write_text(json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False))

if __name__=='__main__':
    write('startups.json',build_problem(pd.read_csv(ROOT/'data/reference/50_Startups.csv'),'Profit',False,METHODS[:5]))
    write('appliances.json',build_problem(pd.read_csv(ROOT/'data/energydata_complete.csv'),'Appliances',True,['Lasso','RFE','ElasticNet','F-test']))
    bundle=joblib.load(ROOT/'outputs/best_model.joblib')
    pipeline=bundle['model']; reg=pipeline.named_steps['regressor']; sc=pipeline.named_steps['scaler']; imp=pipeline.named_steps['imputer']
    coef=reg.coef_/sc.scale_
    preds=pd.read_csv(ROOT/'outputs/metrics/test_predictions.csv')
    write('research.json',{'validation':pd.read_csv(ROOT/'outputs/metrics/validation_results.csv').to_dict('records'),
        'metadata':json.loads((ROOT/'outputs/metrics/best_model_metadata.json').read_text()),
        'original':{'names':bundle['selected_features'],'defaults':imp.statistics_,'coef':coef,'intercept':reg.intercept_-np.dot(coef,sc.mean_),
                    'half_width':bundle['conformal_half_width']},
        'predictions':preds.iloc[np.linspace(0,len(preds)-1,300,dtype=int)].to_dict('records'),
        'diagnostics':diagnostic_payload(preds.actual.values,preds.predicted.values,np.arange(len(preds)))})
    shutil.copyfile(ROOT/'app/utils/regression.py',OUT/'python/regression.py')
    shutil.copyfile(ROOT/'scripts/pages_payload.py',OUT/'python/pages_payload.py')
    write('provenance.json',{'baseline':'39585a0f5485b256ac2d8b7c415ec32f8843577b','scikit_learn':sklearn.__version__,'sources':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [ROOT/'outputs/best_model.joblib',ROOT/'outputs/metrics/validation_results.csv',ROOT/'outputs/metrics/test_predictions.csv',ROOT/'data/reference/50_Startups.csv',ROOT/'data/energydata_complete.csv']}})
    print('Exported verified static model payloads; research files unchanged.')
