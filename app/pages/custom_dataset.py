import hashlib
import streamlit as st
from utils.regression import parse_csv, fetch_csv, STARTUPS_URL
from utils.explorer_ui import render_explorer

st.title('📁 自訂 Dataset 工作區')
st.caption('CSV ≤5 MB、20–10,000 列、2–40 欄；編碼後最多 40 特徵。資料與模型只保留在目前工作階段，不寫入檔案或共用快取。')
source = st.radio('資料來源',['上傳 CSV','公開 CSV URL'],horizontal=True)
try:
    if source == '上傳 CSV':
        upload = st.file_uploader('選擇 CSV',type=['csv'])
        if upload is not None and st.button('載入上傳資料'):
            st.session_state['custom_df'] = parse_csv(upload.getvalue())
    else:
        url = st.text_input('HTTPS CSV URL',value=STARTUPS_URL)
        st.caption('目前核准來源：raw.githubusercontent.com；不接受重新導向、私有網路或其他網域。')
        if st.button('載入 URL'):
            st.session_state['custom_df'] = fetch_csv(url)
except Exception as exc:
    st.error(f'資料載入失敗：{exc}')
if 'custom_df' in st.session_state:
    df = st.session_state['custom_df']
    targets = df.select_dtypes(include='number').columns.tolist()
    if not targets:
        st.error('沒有可用的數值目標欄位。')
    else:
        target = st.selectbox('數值迴歸目標',targets,index=targets.index('Profit') if 'Profit' in targets else 0)
        st.warning('自訂資料使用固定隨機切分 60% / 20% / 20%；不適合時序、群組重複測量或存在目標洩漏的欄位。請先移除識別碼、日期與洩漏欄位。數值中位數補值、類別眾數補值／One-hot encoding 僅在訓練資料擬合。')
        fingerprint = hashlib.sha256(df.to_csv(index=False).encode()+target.encode()).hexdigest()[:12]
        render_explorer(df,target,key='custom_'+fingerprint)
    if st.button('清除資料與自訂模型'):
        for key in list(st.session_state):
            if key.startswith('custom'):
                del st.session_state[key]
        st.rerun()
