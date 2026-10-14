import streamlit as st
import pandas as pd
import plotly.express as px
from utils.common import DATA_FILE, DATASET_URL, FEATURES, TARGET, read_csv

st.title("📊 資料集總覽與探索性分析 (Dataset Overview & EDA)")
st.caption(f"資料來源：[Kaggle — Appliances Energy Prediction]({DATASET_URL})")

st.markdown("""
本專案直接讀入 Kaggle 原始資料中的 **全部 27 個數值候選特徵**，預測目標為家電用電量 `Appliances` (Wh)。
記錄間隔為每 10 分鐘一筆，涵蓋近 4.5 個月的連續時間序列。
""")

if not DATA_FILE.exists():
    st.error(f"找不到資料檔案：`{DATA_FILE}`。請將 Kaggle 下載的 `energydata_complete.csv` 放入 `data/` 目錄。")
    st.stop()

df = read_csv(DATA_FILE)

# KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("總樣本數", f"{len(df):,} 筆")
kpi2.metric("預測目標", TARGET)
kpi3.metric("數值候選特徵數", f"{len(FEATURES)} 個")
kpi4.metric("缺失值 (Missing)", f"{df.isna().sum().sum()} 個")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎯 目標變數分布", "📈 特徵相關性分析", "📋 資料預覽與統計"])

with tab1:
    st.subheader(f"目標變數 `{TARGET}` 分布特徵")
    st.markdown("""
    家電用電量呈現顯著的**右偏態（Right-Skewed）**分布：平時維持在低能耗基準線（50–100 Wh），
    少數烹飪、洗烘衣等高峰時段會衝高至 800+ Wh。這也是不同損失函數（如 OLS vs Huber/L1）表現產生差異的關鍵原因。
    """)
    fig_hist = px.histogram(
        df,
        x=TARGET,
        nbins=60,
        marginal="box",
        title="Appliances 用電量 (Wh) 分布與箱形圖",
        color_discrete_sequence=["#277DA1"],
    )
    fig_hist.update_layout(
        xaxis_title="Appliances (Wh)",
        yaxis_title="次數 (Frequency)",
        template="plotly_white",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("27 個候選特徵與目標變數的 Pearson 相關性")
    corr_series = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    corr_df = pd.DataFrame({"Feature": corr_series.index, "Correlation": corr_series.values})
    
    fig_corr = px.bar(
        corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu",
        title="各候選特徵與 Appliances 之線性相關係數 (Pearson r)",
    )
    fig_corr.update_layout(
        xaxis_title="Pearson 相關係數",
        yaxis_title="特徵名稱",
        template="plotly_white",
        height=700,
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption("提示：照明用電 (`lights`)、室外溫度 (`T_out`)、生活區溫度與特定濕度感測器具有較高相關性。")

with tab3:
    st.subheader("原始資料前 10 筆預覽")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("27 個特徵之敘述統計摘要")
    st.dataframe(df[FEATURES].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]], use_container_width=True)
