"""Read-only access to the original fitted research model."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from utils.common import ROOT
from utils.regression import diagnostics
import plotly.express as px

st.title('⚡ 原始最佳模型：即時預測與解釋')
st.caption('Pearson × OLS｜直接載入研究模型，不重新訓練。')
@st.cache_resource
def load_bundle():
    return joblib.load(ROOT/'outputs/best_model.joblib')
bundle = load_bundle()
model = bundle['model']
names = bundle['selected_features']
imputer, scaler, reg = model.named_steps['imputer'], model.named_steps['scaler'], model.named_steps['regressor']
coef = reg.coef_/scaler.scale_
intercept = reg.intercept_-np.dot(coef,scaler.mean_)
st.latex(r'\widehat{Appliances} = '+f'{intercept:.5g}'+''.join(f'{b:+.5g}\\,{n.replace("_",r"\_")}' for n,b in zip(names,coef)))
st.dataframe(pd.DataFrame({'特徵':names,'原始單位係數':coef}),hide_index=True)
st.info('係數已由標準化空間轉回原始單位。模型於 Train + Validation 擬合；預設輸入取自此擬合資料的中位數（已儲存在 imputer）。')
with st.form('original_prediction'):
    inputs = {}
    columns = st.columns(3)
    for i,(name,value) in enumerate(zip(names,imputer.statistics_)):
        inputs[name] = columns[i%3].number_input(name,value=float(value),key='original_'+name)
    submit = st.form_submit_button('預測用電量')
if submit:
    pred = float(model.predict(pd.DataFrame([inputs],columns=names))[0])
    w = bundle['conformal_half_width']
    st.metric('預測用電量',f'{pred:.2f} Wh')
    st.write(f'95% Conformal 預測區間：[{max(0,pred-w):.2f}, {pred+w:.2f}] Wh')
    st.caption('這是預測區間，並非平均值信賴區間。時間序列可能不符合交換性，名義 95% 並非任意新情境的覆蓋保證。')
preds = pd.read_csv(ROOT/'outputs/metrics/test_predictions.csv')
d = diagnostics(preds.actual,preds.predicted)
st.subheader('測試集最大預測誤差')
d.insert(0,'日期',preds['date'])
st.dataframe(d.sort_values('絕對誤差',ascending=False).head(25),hide_index=True)
st.plotly_chart(px.histogram(d,x='殘差',nbins=50))
