import streamlit as st
import pandas as pd
import plotly.express as px
from utils.common import SELECTORS, FEATURES, load_metric_file

st.title("🧩 特徵選擇策略 (Feature Selection Strategies)")
st.caption("從 27 個候選特徵中精選 15 個關鍵特徵 (K=15) | 7 種演算法 + Baseline 對照")

st.markdown("""
**核心理念：** 特徵選擇改變的是**進入模型的資訊子集**，而不是誤差的懲罰方式。
在時間序列資料中，適度刪除高度共線性的感測器特徵，有助於簡化模型、降低方差並提升泛化能力。
""")

descriptions = {
    "Pearson": "依據自變數與目標變數間的絕對線性相關係數 |r(X_i, y)| 進行單變量排序，選取 Top 15。",
    "F-test": "SelectKBest(f_regression)：以單變量迴歸的 F-檢定統計量與 p-value 進行線性相關排序。",
    "Mutual Information": "互資訊法：基於非參數 k-NN 估計目標與自變數間的統計依賴性，能捕捉非線性關聯。",
    "RFE": "遞迴特徵消除（Recursive Feature Elimination）：以普通線性迴歸權重逐步淘汰影響力最低的變數。",
    "Forward SFS": "前向逐步選擇（Forward SFS）：由空集合開始，結合 TimeSeriesSplit 交叉驗證逐步貪婪加入最有益特徵。",
    "Backward SFS": "後向逐步消除（Backward SFS）：從全部 27 個特徵出發，結合時間序列 CV 逐步剔除貢獻度最低者。",
    "LassoCV ranking": "嵌入式 LassoCV：在時間序列切分下以 L1 正則化搜尋最佳 alpha，依標準化係數絕對值排名取前 15。",
    "Baseline": "控制對照組：不進行特徵篩選，保留全部 27 個數值特徵。",
}

st.subheader("📖 7 種演算法原理速查")
cols = st.columns(2)
for idx, (name, desc) in enumerate(descriptions.items()):
    with cols[idx % 2]:
        with st.expander(f"🔹 {name}", expanded=False):
            st.write(desc)

st.markdown("---")
st.subheader("📊 特徵選擇真實實驗結果")

df = load_metric_file("selected_features.csv", ["feature_selection", "feature", "selected"])

if df is not None:
    d = df.copy()
    d["selected"] = d["selected"].astype(str).str.lower().isin(["1", "true", "yes"])
    
    # Exclude Baseline from frequency to see which features the 7 algorithms agree on
    d_algo = d[d["feature_selection"] != "Baseline"]
    freq = d_algo[d_algo["selected"]].groupby("feature").size().reindex(FEATURES, fill_value=0).sort_values(ascending=False)
    freq_df = pd.DataFrame({"Feature": freq.index, "Selection_Count": freq.values})
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.write("### 🏆 各特徵被選中頻率 (共 7 種演算法)")
        fig_freq = px.bar(
            freq_df,
            x="Selection_Count",
            y="Feature",
            orientation="h",
            color="Selection_Count",
            color_continuous_scale="Blues",
            range_x=[0, 7.5],
            title="特徵入選次數排行 (滿分 7 次)",
        )
        fig_freq.update_layout(
            yaxis={"autorange": "reversed"},
            xaxis_title="被選中次數 (次)",
            yaxis_title="特徵名稱",
            template="plotly_white",
            height=650,
        )
        st.plotly_chart(fig_freq, use_container_width=True)
    
    with col_right:
        st.write("### 🗺️ 特徵選擇矩陣 (Method × Feature)")
        # Pivot table: rows = feature_selection, cols = feature
        pivot = d.pivot(index="feature_selection", columns="feature", values="selected").astype(int)
        # Reorder columns by frequency
        pivot = pivot[freq.index]
        
        fig_mat = px.imshow(
            pivot,
            color_continuous_scale=[(0, "#F0F4F8"), (1, "#277DA1")],
            aspect="auto",
            title="特徵選擇矩陣（藍色代表入選）",
        )
        fig_mat.update_layout(
            xaxis_title="特徵名稱 (依入選頻率由高到低排序)",
            yaxis_title="特徵選擇策略",
            coloraxis_showscale=False,
            template="plotly_white",
            height=650,
        )
        st.plotly_chart(fig_mat, use_container_width=True)
    
    st.info("""
    💡 **洞察重點：**
    * 當多個演算法高度共識選中某變數（例如 `lights`, `T1`, `RH_1`, `T8`），代表該特徵具有極為穩健的預測信號。
    * 溫濕度感測器之間具高共線性（如室內各房間溫度與室外溫度），不同演算法會各自偏好不同感測器作為代表。
    """)
