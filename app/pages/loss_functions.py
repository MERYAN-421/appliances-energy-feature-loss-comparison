import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.common import load_metric_file

st.title("📉 損失函數實證與理論剖析 (Loss Functions Analysis)")
st.caption("探討 4 種線性模型優化目標對殘差懲罰機制、實證指標表現與抗偏態能力之影響")

# ==============================================================================
# SECTION 1: 理論原理與互動式損失曲線
# ==============================================================================
st.subheader("1. 四大損失函數數學原理與優化目標")
st.markdown("""
損失函數定義了模型在訓練階段（`fit`）如何對預測殘差 $e = y - \\hat{y}$ 進行懲罰。
不同的損失目標會引導模型估計不同的條件統計量（如條件均值 vs. 條件中位數）。
""")

loss_specs = [
    {
        "損失目標 (Objective)": "Squared Error (OLS)",
        "數學公式 L(e)": "e²",
        "極小化對應統計量": "條件均值 E[Y|X]",
        "抗離群值能力": "低 (對大殘差施加二次方重罰)",
        "scikit-learn 類別": "LinearRegression",
    },
    {
        "損失目標 (Objective)": "Huber Loss",
        "數學公式 L(e)": "0.5e² (|e| ≤ δ) ; δ(|e| − 0.5δ) (|e| > δ)",
        "極小化對應統計量": "平滑折衷估計量",
        "抗離群值能力": "中高 (近零二次、遠處線性)",
        "scikit-learn 類別": "HuberRegressor",
    },
    {
        "損失目標 (Objective)": "Absolute Error (Median)",
        "數學公式 L(e)": "|e|",
        "極小化對應統計量": "條件中位數 Median(Y|X)",
        "抗離群值能力": "高 (對長尾極值施加恆定線性懲罰)",
        "scikit-learn 類別": "QuantileRegressor(quantile=0.5)",
    },
    {
        "損失目標 (Objective)": "Epsilon-insensitive",
        "數學公式 L(e)": "max(0, |e| − ε)",
        "極小化對應統計量": "不敏感帶外中位數折衷",
        "抗離群值能力": "高 (帶內誤差不予懲罰，專注顯著偏離)",
        "scikit-learn 類別": "SGDRegressor(loss='epsilon_insensitive')",
    },
]

st.dataframe(pd.DataFrame(loss_specs), use_container_width=True, hide_index=True)

with st.expander("🎛️ 互動式探索損失函數幾何形狀 (Illustration Only)", expanded=True):
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        delta = st.slider("Huber 轉折閥值 δ (Delta)", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
    with col_ctrl2:
        eps = st.slider("Epsilon 不敏感管徑寬度 ε (Epsilon)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    e = np.linspace(-3.5, 3.5, 300)
    loss_squared = e ** 2
    loss_huber = np.where(np.abs(e) <= delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))
    loss_abs = np.abs(e)
    loss_eps = np.maximum(0, np.abs(e) - eps)

    fig_curves = go.Figure()
    fig_curves.add_trace(go.Scatter(x=e, y=loss_squared, mode="lines", name="Squared Error (e²)", line=dict(color="#E63946", width=2)))
    fig_curves.add_trace(go.Scatter(x=e, y=loss_huber, mode="lines", name=f"Huber (δ={delta})", line=dict(color="#F4A261", width=2)))
    fig_curves.add_trace(go.Scatter(x=e, y=loss_abs, mode="lines", name="Absolute Error (|e|)", line=dict(color="#2A9D8F", width=2)))
    fig_curves.add_trace(go.Scatter(x=e, y=loss_eps, mode="lines", name=f"Epsilon-insensitive (ε={eps})", line=dict(color="#264653", width=2, dash="dash")))

    fig_curves.update_layout(
        title="殘差 e 與損失值 L(e) 幾何關係圖",
        xaxis_title="殘差 e = Actual − Predicted",
        yaxis_title="懲罰損失值 Loss L(e)",
        template="plotly_white",
        height=380,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    st.plotly_chart(fig_curves, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 2: 真實實驗數據對比 (Empirical Comparison)
# ==============================================================================
st.subheader("2. 32 組真實實驗中四大損失函數之表現對決")
val_df = load_metric_file("validation_results.csv", ["feature_selection", "loss_function", "MAE", "RMSE", "R2"])

if val_df is not None:
    # 1. Best RMSE & Best MAE for each loss function
    best_summary = val_df.groupby("loss_function").agg(
        Best_RMSE=("RMSE", "min"),
        Best_MAE=("MAE", "min"),
        Best_R2=("R2", "max"),
    ).reset_index()

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown("#### 🏆 各損失函數之「最佳驗證 RMSE」")
        fig_rmse = px.bar(
            best_summary.sort_values("Best_RMSE"),
            x="loss_function",
            y="Best_RMSE",
            color="loss_function",
            text="Best_RMSE",
            color_discrete_sequence=["#277DA1", "#4D908E", "#F9C74F", "#F94144"],
            title="最低 RMSE 比較 (越低越好)",
        )
        fig_rmse.update_traces(texttemplate="%{text:.2f} Wh", textposition="outside")
        fig_rmse.update_layout(
            xaxis_title="",
            yaxis_title="RMSE (Wh)",
            yaxis=dict(range=[80, 105]),
            template="plotly_white",
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col_b2:
        st.markdown("#### 🎯 各損失函數之「最佳驗證 MAE」")
        fig_mae = px.bar(
            best_summary.sort_values("Best_MAE"),
            x="loss_function",
            y="Best_MAE",
            color="loss_function",
            text="Best_MAE",
            color_discrete_sequence=["#43AA8B", "#90BE6D", "#F9C74F", "#F94144"],
            title="最低 MAE 比較 (越低越好)",
        )
        fig_mae.update_traces(texttemplate="%{text:.2f} Wh", textposition="outside")
        fig_mae.update_layout(
            xaxis_title="",
            yaxis_title="MAE (Wh)",
            yaxis=dict(range=[30, 65]),
            template="plotly_white",
            showlegend=False,
            height=380,
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    # Controlled Feature Selection Comparison
    st.markdown("#### 🔬 固定特徵選擇條件下：4 種損失函數直面對決")
    available_selectors = val_df["feature_selection"].unique().tolist()
    chosen_sel = st.selectbox(
        "選擇要固定的特徵篩選策略 (Feature Selection Subset):",
        available_selectors,
        index=0,
    )

    subset = val_df[val_df["feature_selection"] == chosen_sel].copy()

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        x=subset["loss_function"],
        y=subset["RMSE"],
        name="RMSE (Wh)",
        marker_color="#277DA1",
        text=subset["RMSE"].round(2),
        textposition="outside",
    ))
    fig_comp.add_trace(go.Bar(
        x=subset["loss_function"],
        y=subset["MAE"],
        name="MAE (Wh)",
        marker_color="#F8961E",
        text=subset["MAE"].round(2),
        textposition="outside",
    ))

    fig_comp.update_layout(
        barmode="group",
        title=f"【{chosen_sel}】特徵子集下，四種損失函數之 RMSE 與 MAE 對照",
        xaxis_title="損失函數 (Loss Function)",
        yaxis_title="誤差值 (Wh)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ==============================================================================
# SECTION 3: 深度機制探討與方法學注意事項
# ==============================================================================
st.subheader("3. 深度機制解析與機器學習方法學澄清")

col_th1, col_th2 = st.columns(2)

with col_th1:
    st.info("""
    ### 💡 核心成因：為什麼 OLS 的 RMSE 最低，而 Median/Huber 的 MAE 最低？
    
    1. **數學目標函數的一致性：**
       * **OLS (Squared Error)**：在數學理論上，二次方誤差的經驗極小化求解對應於**條件均值（Conditional Mean $\\mathbb{E}[Y|X]$）**。由於 RMSE 本身就是二次方殘差均值的平方根，OLS 本質上就是在直接極小化 RMSE！
       * **Absolute Error (Median)**：$L_1$ 損失函數的經驗極小化求解對應於**條件中位數（Conditional Median $\\text{Median}(Y|X)$）**。MAE 亦為 $L_1$ 距離，因此中位數迴歸在理論上必然保證 MAE 極小。
    
    2. **家電用電量之厚尾偏態（Right-Skewed）：**
       * 日常用電在 90% 的時間處於 50–60 Wh 的極低基準負載，僅在少數烹調或洗烘時段產生 800+ Wh 突波。
       * **條件中位數模型**（如 Quantile / Epsilon / Huber）不受極端突波影響，將預測值錨定在 50–60 Wh 附近，使得絕大多數常規時段的誤差極小，因此 **MAE 顯著降低至約 43 Wh**！但在極端突波發生時會產生極大正殘差，計算 RMSE 時被平方放大，故 **RMSE 偏高 (~96 Wh)**。
       * **OLS 模型** 為了避免 800 Wh 突波產生的超大二次方懲罰，被迫將平常的預測基準調高至 80–100 Wh，犧牲了平時的絕對精度（**MAE 上升至 56–58 Wh**），但成功壓低了極端突波的平方懲罰，因此 **RMSE 最低 (~94.4 Wh)**。
    """)

with col_th2:
    st.warning("""
    ### ⚠️ 關鍵觀念澄清：Training Loss ≠ Evaluation Metric
    
    在評估機器學習模型時，必須將**訓練損失（Training Loss）**與**驗證評估指標（Evaluation Metric）**明確拆解：
    
    1. **訓練損失（Training Loss）是優化路徑**：
       它決定了梯度下降或演算法求解器在參數空間中尋找最優解的方向。
    2. **評估指標（Evaluation Metric）是業務驗收標準**：
       如果業務情境關心的是「一般典型耗電的日常預測精確度」，應採用 **MAE** 作為驗收指標，此時 Median / Huber 表現更優。  
       如果業務情境極度忌憚「電網調度過載的大規模未預期突波」，應採用 **RMSE**，此時 OLS 才是首選。
    
    ---
    
    ### ⚙️ scikit-learn 估計器實作之隱含差異提醒
    
    不同損失函數在 scikit-learn 中的類別並非僅僅是 Loss Function 的替換，還涉及以下關鍵差異：
    * **`LinearRegression`**：採用奇異值分解（SVD）之無約束解析解，無正則化。
    * **`HuberRegressor`**：採用 L-BFGS-B 優化器，且預設帶有 $L_2$ 正則化懲罰項（`alpha=1.0`）。
    * **`QuantileRegressor`**：轉化為線性規劃問題（Linear Programming，HiGHS 求解器），預設帶有 $L_1$ 正則化項（`alpha=1.0`）。
    * **`SGDRegressor`**：隨機梯度下降迭代求解，對特徵縮放、學習率（Learning Rate）與早停機制（Early Stopping）敏感。
    
    因此，實務評估時應理解指標差異同時包含了**損失函數本質**與**求解器正則化強度**的複合效應。
    """)
