"""Verify exported Pages inference against the actual Python pipelines."""
import json
import subprocess
import sys
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'app'))
from utils.regression import prepare,fit_model

class PagesTests(unittest.TestCase):
    def test_browser_predictions_match_python(self):
        payload=json.loads((ROOT/'web/data/startups.json').read_text())
        ctx=prepare(pd.read_csv(ROOT/'data/reference/50_Startups.csv'),'Profit')
        inputs={i['name']:i['default'] for i in payload['schema']['inputs']}
        row=pd.DataFrame([inputs],columns=ctx['X'].columns)
        expected=[]
        for method,group in payload['methods'].items():
            for model in group['models']:
                r=fit_model(ctx,method,model['k'])
                expected.append(float(r['model'].predict(ctx['pre'].transform(row)[:,r['cols']])[0]))
        script="""
import {readFileSync} from 'node:fs';
import {predictModel} from './web/core.mjs';
const p=JSON.parse(readFileSync('web/data/startups.json'));
const values=Object.fromEntries(p.schema.inputs.map(s=>[s.name,s.default]));
console.log(JSON.stringify(Object.values(p.methods).flatMap(g=>g.models.map(m=>predictModel(m,p.schema,values)))));
"""
        result=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,capture_output=True,text=True,check=True)
        np.testing.assert_allclose(json.loads(result.stdout),expected,atol=1e-7,rtol=0)

    def test_original_bundle_preserved_in_browser(self):
        payload=json.loads((ROOT/'web/data/research.json').read_text())
        bundle=joblib.load(ROOT/'outputs/best_model.joblib')
        self.assertEqual(payload['original']['names'],bundle['selected_features'])
        self.assertEqual(payload['original']['half_width'],bundle['conformal_half_width'])
        inputs=dict(zip(bundle['selected_features'],payload['original']['defaults']))
        pred=float(bundle['model'].predict(pd.DataFrame([inputs],columns=bundle['selected_features']))[0])
        script="""
import {readFileSync} from 'node:fs';
import {predictOriginal} from './web/core.mjs';
const m=JSON.parse(readFileSync('web/data/research.json')).original;
console.log(predictOriginal(m,Object.fromEntries(m.names.map((n,i)=>[n,m.defaults[i]]))));
"""
        result=subprocess.run(['node','--input-type=module','-e',script],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertAlmostEqual(float(result.stdout),pred,places=9)
        self.assertEqual(len(payload['validation']),32)
        self.assertEqual(len(set(r['loss_function'] for r in payload['validation'])),4)

    def test_browser_python_matches_source(self):
        self.assertEqual((ROOT/'web/python/regression.py').read_bytes(),(ROOT/'app/utils/regression.py').read_bytes())
        self.assertEqual((ROOT/'web/python/pages_payload.py').read_bytes(),(ROOT/'scripts/pages_payload.py').read_bytes())

    def test_static_artifact_selection_and_metrics(self):
        for problem in ['startups','appliances']:
            p=json.loads((ROOT/f'web/data/{problem}.json').read_text())
            for group in p['methods'].values():
                winner=min(group['models'],key=lambda m:m['metrics']['RMSE'])
                self.assertEqual(winner['k'],group['best_k'])
                self.assertEqual(group['test']['k'],winner['k'])
                for model in group['models']:
                    self.assertEqual(len(model['names']),model['k'])
                    self.assertTrue(np.isfinite(model['metrics']['RMSE']))
                    stats=model['inference']
                    if not stats['error']:
                        self.assertEqual([r['特徵'] for r in stats['table']][1:],model['names'])

if __name__=='__main__': unittest.main()
