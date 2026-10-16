import pandas as pd
import streamlit as st
from utils.common import DATA_FILE
from utils.explorer_ui import render_explorer

st.title('📈 多元線性迴歸探索器')
st.caption('特徵選擇 · 簡約模型 · OLS 統計解釋 · 即時預測')
if not DATA_FILE.exists():
    st.info('K 分析需要原始 Appliances CSV。原始研究展示與最佳模型預測不需要此檔案。')
else:
    render_explorer(pd.read_csv(DATA_FILE),'Appliances',True,key='appliances',unit='Wh')
