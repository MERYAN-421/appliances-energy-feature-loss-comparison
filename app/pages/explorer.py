import pandas as pd
import streamlit as st
from utils.common import DATA_FILE, ROOT
from utils.explorer_ui import render_explorer

st.title('📈 多元線性迴歸探索器')
st.caption('特徵選擇 · 簡約模型 · OLS 統計解釋 · 即時預測')
problem = st.selectbox('選擇 Problem',['50 Startups — Profit Prediction','Appliances Energy — Energy Consumption Prediction'])
if problem.startswith('50'):
    st.info('50 筆資料：固定 random_state=42，Train 60% / Validation 20% / Test 20%。小樣本結果對切分敏感。')
    render_explorer(pd.read_csv(ROOT/'data/reference/50_Startups.csv'),'Profit',False,key='startups',unit='$')
elif not DATA_FILE.exists():
    st.info('K 分析需要原始 Appliances CSV。原始研究展示與最佳模型預測不需要此檔案。')
else:
    st.info('沿用時序 Train 55% / Validation 15% / Calibration 15% / Test 15%；本頁不使用 Calibration。')
    render_explorer(pd.read_csv(DATA_FILE),'Appliances',True,key='appliances',unit='Wh')
