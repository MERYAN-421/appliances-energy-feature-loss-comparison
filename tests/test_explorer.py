"""Run with .venv/bin/python -m unittest discover -s tests -v."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
from utils.regression import *
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]

def load_original_bundle():
    import joblib
    return joblib.load(ROOT/'outputs/best_model.joblib')

class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.df=pd.read_csv(ROOT/'data/reference/50_Startups.csv')
        self.ctx=prepare(self.df,'Profit')

    def test_five_selectors_and_k(self):
        for method in METHODS:
            for k in range(1,6):
                with self.subTest(method=method,k=k):
                    result=fit_model(self.ctx,method,k)
                    self.assertEqual(len(result['names']),k)
                    self.assertTrue(np.isfinite(result['metrics']['RMSE']))
        self.assertFalse(set(self.ctx['ti']) & set(self.ctx['test']))
        self.assertFalse(set(self.ctx['vi']) & set(self.ctx['test']))

    def test_equation_inference_and_mapping(self):
        r=fit_model(self.ctx,'Lasso',2)
        x=self.ctx['train'][:,r['cols']]
        np.testing.assert_allclose(r['model'].predict(x),x@r['model'].coef_+r['model'].intercept_)
        table,f,p=inference(x,self.ctx['y'][self.ctx['ti']],r['names'],r['model'])
        self.assertEqual(table['特徵'].tolist()[1:],r['names'].tolist())
        self.assertGreater(f,0)
        self.assertTrue(0<=p<=1)
        # Independent classical F-statistic identity.
        score=r['model'].score(x,self.ctx['y'][self.ctx['ti']])
        self.assertAlmostEqual(f,(score/2)/((1-score)/(len(x)-3)),places=7)
        pred=r['model'].predict(self.ctx['pre'].transform(self.ctx['X'].iloc[[0]])[:,r['cols']])
        self.assertEqual(pred.shape,(1,))

    def test_preprocessing_leakage(self):
        a=fit_model(self.ctx,'Lasso',2)
        changed=self.df.copy()
        changed.loc[self.ctx['test'],'Profit']=1e20
        bctx=prepare(changed,'Profit')
        b=fit_model(bctx,'Lasso',2)
        np.testing.assert_array_equal(a['cols'],b['cols'])
        np.testing.assert_allclose(a['predictions'],b['predictions'])
        np.testing.assert_allclose(self.ctx['pre'].named_transformers_['numeric'].named_steps['standardscaler'].mean_, self.ctx['X'].iloc[self.ctx['ti']].select_dtypes(include='number').mean())

    def test_csv_limits_and_security(self):
        loaded=parse_csv(self.df.to_csv(index=False).encode())
        self.assertEqual(loaded.shape,self.df.shape)
        for url in ['http://raw.githubusercontent.com/a.csv','https://127.0.0.1/a.csv','https://raw.githubusercontent.com@localhost/a.csv','https://raw.githubusercontent.com/a.txt']:
            with self.assertRaises(ValueError): fetch_csv(url)
        with self.assertRaises(ValueError): parse_csv(b'x'*(MAX_BYTES+1))
        d=diagnostics(np.array([0,2]),np.array([1,1]))
        self.assertTrue(np.isnan(d.iloc[0]['百分比誤差 (%)']))
        self.assertEqual(d.iloc[1]['百分比誤差 (%)'],50)

    def test_live_public_csv(self):
        self.assertEqual(fetch_csv(STARTUPS_URL).shape,(50,5))

    def test_explorer_interactions(self):
        at=AppTest.from_file(str(ROOT/'app/pages/explorer.py')).run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(len(at.selectbox[0].options),2)
        at.button[0].click().run(timeout=60)
        self.assertFalse(at.exception)
        at.select_slider[0].set_value(3).run(timeout=60)
        self.assertFalse(at.exception)
        next(b for b in at.button if b.label=='計算預測').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertTrue(any(m.label=='預測值' for m in at.metric))
        at.selectbox[0].select('Appliances Energy — Energy Consumption Prediction').run(timeout=60)
        self.assertFalse(at.exception)

    def test_custom_url_ui(self):
        at=AppTest.from_file(str(ROOT/'app/pages/custom_dataset.py')).run(timeout=60)
        at.radio[0].set_value('公開 CSV URL').run()
        next(b for b in at.button if b.label=='載入 URL').click().run(timeout=60)
        self.assertFalse(at.exception)
        self.assertIn('custom_df',at.session_state)
        next(b for b in at.button if b.label=='計算／更新 K 分析').click().run(timeout=60)
        self.assertFalse(at.exception)

    def test_cloud_appliances_without_local_csv(self):
        from unittest.mock import patch
        with patch('utils.common.DATA_FILE',ROOT/'data/nonexistent-cloud-fixture.csv'):
            at=AppTest.from_file(str(ROOT/'app/pages/explorer.py')).run(timeout=60)
            at.selectbox[0].select('Appliances Energy — Energy Consumption Prediction').run(timeout=60)
            self.assertFalse(at.exception)
            self.assertFalse(at.error)
            self.assertTrue(any('計算／更新 K 分析'==b.label for b in at.button))

    def test_appliances_workspace(self):
        path=ROOT/'data/energydata_complete.csv'
        if not path.exists():
            self.skipTest('Run scripts/validate.sh to provision public raw data')
        ctx=prepare(pd.read_csv(path),'Appliances',True)
        self.assertEqual(len(ctx['train']),int(19735*.55))
        self.assertGreaterEqual(ctx['test'][0],int(19735*.85))
        r=fit_model(ctx,'F-test',15)
        self.assertEqual(len(r['cols']),15)
        self.assertTrue(np.isfinite(r['metrics']['RMSE']))

    def test_original_live_page(self):
        at=AppTest.from_file(str(ROOT/'app/pages/live_model.py')).run(timeout=30)
        self.assertFalse(at.exception)
        at.button[0].click().run(timeout=30)
        self.assertFalse(at.exception)
        bundle=load_original_bundle()
        inputs={w.label:w.value for w in at.number_input}
        expected=bundle['model'].predict(pd.DataFrame([inputs],columns=bundle['selected_features']))[0]
        shown=next(m.value for m in at.metric if m.label=='預測用電量')
        self.assertEqual(shown,f'{expected:.2f} Wh')

if __name__=='__main__': unittest.main()
