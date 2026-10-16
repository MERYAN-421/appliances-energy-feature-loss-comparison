from io import BytesIO
import hashlib
import urllib.request
import pandas as pd
import streamlit as st
from utils.common import DATA_FILE, ROOT
from utils.explorer_ui import render_explorer

@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def public_appliances():
    # Trusted predefined source, independent of user-supplied URL handling.
    url = 'https://raw.githubusercontent.com/LuisM78/Appliances-energy-prediction-data/master/energydata_complete.csv'
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(15_000_001)
    if hashlib.sha256(raw).hexdigest() != '2820bf712ad0275cb18b85a05250926100d8e65ebb9f4d2d016ca91ea152a25d':
        raise ValueError('公開資料校驗失敗，未進行訓練。')
    return pd.read_csv(BytesIO(raw))

st.title('📈 多元線性迴歸探索器')
st.caption('特徵選擇 · 簡約模型 · OLS 統計解釋 · 即時預測')
problem = st.selectbox('選擇 Problem',['50 Startups — Profit Prediction','Appliances Energy — Energy Consumption Prediction'])
if problem.startswith('50'):
    st.info('50 筆資料：固定 random_state=42，Train 60% / Validation 20% / Test 20%。小樣本結果對切分敏感。')
    render_explorer(pd.read_csv(ROOT/'data/reference/50_Startups.csv'),'Profit',False,key='startups',unit='$')
else:
    st.info('沿用時序 Train 55% / Validation 15% / Calibration 15% / Test 15%；本頁不使用 Calibration。')
    try:
        with st.spinner('載入公開 Appliances 資料…'):
            df = pd.read_csv(DATA_FILE) if DATA_FILE.exists() else public_appliances()
        render_explorer(df,'Appliances',True,key='appliances',unit='Wh')
    except Exception as exc:
        st.error(f'公開資料載入失敗：{exc}')
        st.info('請確認 raw.githubusercontent.com 連線。原始研究頁面與最佳模型預測仍可使用。')
