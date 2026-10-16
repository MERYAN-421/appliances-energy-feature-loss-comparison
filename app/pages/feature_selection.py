import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.common import FEATURES, load_metric_file

st.title("🧩 特徵選擇實證與方法學分析 (Feature Selection Analysis)")
st.caption("從 27 個候選變數中精選 15 個關鍵特徵 (K=15) | 7 種演算法 + Baseline 對照實驗")

st.markdown("""
**核心問題：哪些環境與時間變數應該進入預測模型？**  
在物聯網（IoT）家庭能源監測中，各房間溫濕度感測器間具有極強的共線性（Multicollinearity）。  
透過特徵篩選降維，旨在消除冗餘噪訊、降低估計方差（Variance Reduction），並提升模型在新時間序列上的泛化能力。
""")

# ==============================================================================
# SECTION 1: 7 種特徵篩選策略原理
# ==============================================================================
descriptions = {
    "Pearson": "依據自變數與目標變數之絕對線性相關係數 |r(X_i, y)| 進行單變量排序，取 Top 15。",
    "F-test": "SelectKBest(f_regression)：以單變量迴歸之 F 檢定統計量與 p-value 進行線性關聯排序。",
    "Mutual Information": "互資訊法（Mutual Information）：基於非參數 k-NN 估計目標與特徵的統計依賴性，能捕捉非線性關聯。",
    "RFE": "遞迴特徵消除（Recursive Feature Elimination）：以普通線性迴歸權重逐步淘汰影響力最低的變數。",
    "Forward SFS": "前向逐步選擇（Forward SFS）：由空集合出發，在無洩漏 TimeSeriesSplit 交叉驗證下貪婪加入邊際效益最高者。",
    "Backward SFS": "後向逐步消除（Backward SFS）：由全量 27 特徵出發，在無洩漏 TimeSeriesSplit 交叉驗證下貪婪剔除貢獻度最低者。",
    "LassoCV ranking": "嵌入式 LassoCV：在無洩漏 TimeSeriesSplit 下搜尋最優 alpha，依完整訓練集標準化係數絕對值排名取前 15。",
    "Baseline": "控制基準對照組：不進行任何特徵篩選，完整保留全部 27 個數值候選特徵。",
}

with st.expander("📖 7 種特徵選擇演算法與對照組原理速查", expanded=False):
    cols = st.columns(2)
    for idx, (name, desc) in enumerate(descriptions.items()):
        with cols[idx % 2]:
            st.markdown(f"**🔹 {name}**：{desc}")

st.markdown("---")

# ==============================================================================
# SECTION 2: 選擇矩陣與頻率分析 (Task 3 Core)
# ==============================================================================
st.subheader("1. 27 個候選特徵選擇矩陣與入選頻率")

sel_df = load_metric_file("selected_features.csv", ["feature_selection", "feature", "selected"])

if sel_df is not None:
    d = sel_df.copy()
    d["selected"] = d["selected"].astype(str).str.lower().isin(["1", "true", "yes"])

    # Exclude Baseline from frequency calculation
    d_algo = d[d["feature_selection"] != "Baseline"]
    freq_series = d_algo[d_algo["selected"]].groupby("feature").size().reindex(FEATURES, fill_value=0).sort_values(ascending=False)
    freq_df = pd.DataFrame({"Feature": freq_series.index, "Selection_Count": freq_series.values})

    # Order columns by frequency
    sorted_features = freq_series.index.tolist()

    col_freq, col_mat = st.columns([1.0, 1.1])

    with col_freq:
        st.markdown("#### 🏆 各特徵被選中頻率 (共 7 種演算法)")
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
            yaxis_title="候選特徵",
            template="plotly_white",
            height=700,
            margin=dict(l=20, r=20, t=40, b=40),
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    with col_mat:
        st.markdown("#### 🗺️ 8 種方法 × 27 特徵選擇矩陣")
        # Pivot table: rows = feature_selection, cols = feature
        pivot = d.pivot(index="feature_selection", columns="feature", values="selected").astype(int)
        
        # Standardize row order
        method_order = ["Pearson", "F-test", "Mutual Information", "RFE", "Forward SFS", "Backward SFS", "LassoCV ranking", "Baseline"]
        pivot = pivot.reindex([m for m in method_order if m in pivot.index])
        pivot = pivot[sorted_features]  # Sort columns by frequency

        # Text values
        z_matrix = pivot.to_numpy()
        text_matrix = np.where(z_matrix == 1, "✓", "")

        fig_mat = go.Figure(data=go.Heatmap(
            z=z_matrix,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            colorscale=[[0, "#EDF2F4"], [1, "#1D3557"]],
            showscale=False,
            hovertemplate="策略: %{y}<br>特徵: %{x}<br>入選狀態: %{text}<extra></extra>",
        ))
        fig_mat.update_layout(
            title="特徵選擇矩陣（深藍 ✓ 代表入選，淺灰代表剔除）",
            xaxis_title="候選特徵名稱 (由高至低依共識頻率排序)",
            yaxis_title="特徵選擇策略",
            template="plotly_white",
            height=700,
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis=dict(tickangle=45),
        )
        st.plotly_chart(fig_mat, use_container_width=True)

    st.info("""
    💡 **特徵共識洞察：**
    * **極高共識特徵（選中 6–7 次）：** 照明用電 (`lights`)、一樓溫度 (`T1`)、一樓濕度 (`RH_1`)、二樓溫度 (`T2`)、三樓溫度 (`T3`)、八樓濕度 (`RH_8`)、室外溫度 (`T_out`)。這些感測器與家庭生活作息或大氣環境高度同步。
    * **極低共識特徵（選中 0–1 次）：** 隨機噪訊變數 (`rv1`, `rv2`)、能見度 (`Visibility`)、特定偏僻區域濕度 (`RH_4`, `RH_5`)。
    """)

st.markdown("---")

# ==============================================================================
# SECTION 3: 每種 Selector 選中之 15 個特徵清單
# ==============================================================================
st.subheader("2. 各特徵選擇器之 15 個入選特徵詳情")

if sel_df is not None:
    all_methods = [m for m in descriptions.keys() if m in sel_df["feature_selection"].unique()]
    chosen_method = st.selectbox("選擇要檢視的演算法：", all_methods, index=0)

    method_features = sel_df[(sel_df["feature_selection"] == chosen_method) & (sel_df["selected"])]["feature"].tolist()

    col_m_left, col_m_right = st.columns([1, 1])
    with col_m_left:
        st.metric(f"【{chosen_method}】選中特徵數", f"{len(method_features)} 個", help="K=15 (Baseline為27)")
        st.write("##### 📋 入選特徵清單：")
        
        feature_category_df = pd.DataFrame({
            "序號": range(1, len(method_features) + 1),
            "特徵名稱": method_features,
            "類別屬性": [
                "照明用電 (Lighting)" if f == "lights" else
                "室內溫度 (Indoor Temp)" if f.startswith("T") and f != "T_out" and f != "Tdewpoint" else
                "室內濕度 (Indoor Humidity)" if f.startswith("RH") and f != "RH_out" else
                "室外氣象 (Weather/Outdoor)" if f in ["T_out", "RH_out", "Windspeed", "Press_mm_hg", "Visibility", "Tdewpoint"] else
                "隨機變數 (Random Noise)" if f in ["rv1", "rv2"] else "其他"
                for f in method_features
            ]
        })
        st.dataframe(feature_category_df, use_container_width=True, hide_index=True)

    with col_m_right:
        st.write("##### 📊 入選特徵物理類別佔比：")
        cat_counts = feature_category_df["類別屬性"].value_counts().reset_index()
        cat_counts.columns = ["類別", "數量"]
        fig_pie = px.pie(
            cat_counts,
            values="數量",
            names="類別",
            color="類別",
            color_discrete_map={
                "室內溫度 (Indoor Temp)": "#2A9D8F",
                "室內濕度 (Indoor Humidity)": "#457B9D",
                "室外氣象 (Weather/Outdoor)": "#E76F51",
                "照明用電 (Lighting)": "#F4A261",
                "隨機變數 (Random Noise)": "#E63946",
            },
            hole=0.4,
            title=f"{chosen_method} 特徵組合領域分布",
        )
        fig_pie.update_layout(template="plotly_white", height=340)
        st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 4: BASELINE (27) VS. SELECTED (15) 模型比較
# ==============================================================================
st.subheader("3. Baseline (27 特徵) vs. 精選特徵 (15 特徵) 模型比較")
st.markdown("""
究竟「保留全部 27 個特徵」好，還是「篩選 15 個特徵」好？  
我們在 4 種損失函數下，比較 **Baseline** 與各 **15 特徵模型** 的驗證集 RMSE 與 MAE。
""")

val_df_comp = load_metric_file("validation_results.csv", ["feature_selection", "loss_function", "RMSE", "MAE"])

if val_df_comp is not None:
    # Under Squared Error (OLS)
    ols_df = val_df_comp[val_df_comp["loss_function"] == "Squared Error (OLS)"].sort_values("RMSE").copy()
    ols_df["Is_Baseline"] = ols_df["feature_selection"].apply(lambda x: "Baseline (27 特徵)" if x == "Baseline" else "精選 (15 特徵)")

    fig_ols_comp = px.bar(
        ols_df,
        x="feature_selection",
        y="RMSE",
        color="Is_Baseline",
        color_discrete_map={"Baseline (27 特徵)": "#F4A261", "精選 (15 特徵)": "#2A9D8F"},
        text=ols_df["RMSE"].round(2),
        title="OLS 迴歸下：各特徵選擇方法之 Validation RMSE 比較 (越低越好)",
    )
    fig_ols_comp.update_traces(textposition="outside")
    fig_ols_comp.update_layout(
        xaxis_title="特徵選擇策略",
        yaxis_title="RMSE (Wh)",
        yaxis=dict(range=[90, 102]),
        template="plotly_white",
        height=380,
    )
    st.plotly_chart(fig_ols_comp, use_container_width=True)

    baseline_ols_rmse = float(ols_df[ols_df["feature_selection"] == "Baseline"]["RMSE"].iloc[0])
    best_ols_rmse = float(ols_df["RMSE"].min())
    diff_rmse = baseline_ols_rmse - best_ols_rmse

    st.success(f"""
    📈 **實證結論：**  
    * **Baseline (全 27 特徵) RMSE：** `{baseline_ols_rmse:.2f} Wh`
    * **Pearson / F-test (精選 15 特徵) RMSE：** `{best_ols_rmse:.2f} Wh`（誤差降低了 **{diff_rmse:.2f} Wh**）  
    剔除 12 個高度共線及噪訊特徵後，模型複雜度降低，在獨立驗證集上的泛化誤差反而更小，充分證明了特徵選擇在時間序列迴歸中的**抗過擬合價值**。
    """)

st.markdown("---")

# ==============================================================================
# SECTION 5: 隨機變數 RV1、RV2 檢驗專區
# ==============================================================================
st.subheader("4. 隨機噪訊變數 `rv1`、`rv2` 檢驗專題")
st.markdown("""
Kaggle 原始資料集中人工注入了兩組均勻分佈隨機數變數：`rv1` 與 `rv2`。  
理論上，具備良好噪訊過濾能力的演算法應將其**全數剔除**。以下檢視 7 種演算法是否成功辨識純隨機噪訊：
""")

if sel_df is not None:
    rv_df = sel_df[sel_df["feature"].isin(["rv1", "rv2"])].copy()
    rv_pivot = rv_df.pivot(index="feature_selection", columns="feature", values="selected")
    rv_pivot["兩者皆剔除 (通過檢驗)"] = (~rv_pivot["rv1"]) & (~rv_pivot["rv2"])

    col_rv1, col_rv2 = st.columns([1, 1.2])

    with col_rv1:
        st.write("##### 📋 各演算法對 rv1 / rv2 篩選狀態：")
        
        display_rv = rv_pivot.reset_index().copy()
        display_rv["rv1 入選"] = display_rv["rv1"].apply(lambda x: "❌ 入選 (誤採)" if x else "✅ 剔除 (正確)")
        display_rv["rv2 入選"] = display_rv["rv2"].apply(lambda x: "❌ 入選 (誤採)" if x else "✅ 剔除 (正確)")
        display_rv["檢驗結果"] = display_rv["兩者皆剔除 (通過檢驗)"].apply(lambda x: "🟢 成功過濾噪訊" if x else "🔴 誤採隨機噪訊")
        
        st.dataframe(
            display_rv[["feature_selection", "rv1 入選", "rv2 入選", "檢驗結果"]],
            use_container_width=True,
            hide_index=True,
        )

    with col_rv2:
        st.warning("""
        🔍 **深層機器學習洞察：為什麼 SFS 會誤採 rv1 與 rv2？**
        
        * **過濾法（Filter Methods）與正則化（LassoCV）全數過關：**  
          Pearson、F-test、互資訊（MI）、RFE 以及 LassoCV 皆正確地將 `rv1` 與 `rv2` 排除在 Top 15 之外，因為它們在統計檢定或 L1 懲罰下係數均趨近於 0。
        * **逐步選擇（Forward / Backward SFS）誤採隨機噪訊：**  
          SFS 屬於**貪婪搜尋（Greedy Search）**策略。在時間序列切分（TimeSeriesSplit）有限的訓練折疊中，純隨機數可能碰巧在特定時段與目標殘差產生微弱的偶然相關（Spurious Correlation），導致 SFS 誤將其作為邊際改善而納入，最終陷入局部最優解。  
          這也是學界公認**逐步回歸在共線性與噪訊特徵存在時易產生假陽性（False Positive）**的經典案例！
        """)

st.markdown("---")

# ==============================================================================
# SECTION 6: 警示：不可將選中頻率直接解釋為因果重要性
# ==============================================================================
st.error("""
### 🚨 統計學與因果推論嚴正警示：特徵入選頻率 ≠ 因果重要性 (Selection Frequency ≠ Causal Importance)

在解讀特徵選擇結果時，絕對**不可**將「某特徵被選中 7 次」直接等同於「該特徵是造成能耗變化的最主要因果因素」：

1. **相關不等於因果（Correlation is not Causation）：**  
   特徵選擇演算法僅評估統計上的關聯強度（Association）或預測力（Predictive Utility），完全無法證明環境變數對用電量的物理因果關係（例如開燈 `lights` 耗電常與人員在家活動同步，但燈光本身可能只是作息的代理變數）。
2. **感測器多重共線性遮蔽（Collinearity Masking）：**  
   室內 9 個房間的溫度感測器（`T1` 至 `T9`）之間彼此相關性高達 $0.85 \\sim 0.98$。當某個演算法（如 Lasso 或 SFS）選中了 `T1` 與 `T8` 後，與它們高度重疊的 `T7` 就會因邊際資訊遞減而被捨棄。**這絕不代表 `T7` 在物理上對溫度或耗電不重要**，而僅代表它的資訊已被 `T1` 所代理。
3. **特徵工程的目的：**  
   本階段特徵選擇的純粹目標是**建構最少變異、最高預測精度且防範過擬合的工程子集**，任何物理因果解釋均需仰賴受控實驗（Randomized Experiments）或結構因果模型（SCM）。
""")
