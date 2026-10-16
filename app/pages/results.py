import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils.common import METRICS, load_metric_file

st.title("🏆 實驗成果與模型比較 (Results & Model Evaluation)")
st.caption("基於 32 組真實實驗數據 (7 Selectors + Baseline × 4 Loss Functions) 之完整評估儀表板")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 模型評估比較 (Model Comparison)",
    "🎯 預測曲線與區間 (Prediction Analysis)",
    "🔍 殘差診斷分析 (Residual Diagnostics)",
    "⭐ 最佳模型中繼檔案 (Best Model Details)",
])

# ==============================================================================
# TAB 1: MODEL COMPARISON
# ==============================================================================
with tab1:
    st.subheader("32 組實驗交叉矩陣評估")
    st.markdown("""
    在 **Validation Set (15%)** 上系統性評估 8 種特徵選擇配置與 4 種損失函數組合。
    請切換下方指標以觀察不同評估準則下的模型排序與矩陣熱圖。
    """)

    val_df = load_metric_file("validation_results.csv", ["feature_selection", "loss_function", "n_features", "MAE", "RMSE", "R2"])

    if val_df is not None:
        metric_choice = st.radio(
            "📌 選擇評估指標 (Evaluation Metric):",
            ["RMSE", "MAE", "R2"],
            format_func=lambda x: {
                "RMSE": "RMSE (均方根誤差, Wh) — 懲罰大誤差，越低越好",
                "MAE": "MAE (平均絕對誤差, Wh) — 穩健抗離群，越低越好",
                "R2": "R² (決定係數) — 解釋變異量比例，越高越好",
            }[x],
            horizontal=True,
            index=0,
        )

        higher_is_better = (metric_choice == "R2")
        unit = " Wh" if metric_choice in ["RMSE", "MAE"] else ""
        metric_label = f"{metric_choice}{unit}"

        # 8 x 4 Heatmap Pivot Table
        pivot = val_df.pivot_table(
            index="feature_selection",
            columns="loss_function",
            values=metric_choice,
            aggfunc="mean",
        )

        # Standardize row order so Baseline is clearly visible
        selector_order = [
            "Pearson", "F-test", "Mutual Information", "RFE",
            "Forward SFS", "Backward SFS", "LassoCV ranking", "Baseline",
        ]
        present_selectors = [s for s in selector_order if s in pivot.index]
        pivot = pivot.reindex(present_selectors)

        col_heat, col_rank = st.columns([1.1, 1.0])

        with col_heat:
            st.markdown(f"#### 8 × 4 配置熱圖 ({metric_label})")
            colorscale = "Blues" if higher_is_better else "Blues_r"
            z_values = pivot.to_numpy()

            # Format cell annotations
            text_values = np.empty_like(z_values, dtype=object)
            for i in range(z_values.shape[0]):
                for j in range(z_values.shape[1]):
                    val = z_values[i, j]
                    text_values[i, j] = f"{val:.2f}" if metric_choice != "R2" else f"{val:.3f}"

            fig_heat = go.Figure(data=go.Heatmap(
                z=z_values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                text=text_values,
                texttemplate="%{text}",
                textfont={"size": 12},
                colorscale=colorscale,
                colorbar=dict(title=metric_label),
                hoverongaps=False,
                hovertemplate="特徵選擇: %{y}<br>損失目標: %{x}<br>" + f"{metric_choice}: %{{z:.3f}}{unit}<extra></extra>",
            ))
            fig_heat.update_layout(
                xaxis_title="損失函數 (Loss Function)",
                yaxis_title="特徵選擇策略 (Feature Selector)",
                template="plotly_white",
                height=480,
                margin=dict(l=40, r=40, t=30, b=50),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with col_rank:
            st.markdown(f"#### Top 10 模型排名 ({metric_label})")
            val_sorted = val_df.sort_values(by=metric_choice, ascending=not higher_is_better).copy()
            val_sorted["Combination"] = val_sorted["feature_selection"] + " × " + val_sorted["loss_function"]

            top10 = val_sorted.head(10).copy()
            best_combo = top10.iloc[0]["Combination"]

            # Assign color categories: Best = Gold, Baseline = Orange, Others = Blue
            def get_color_tag(row):
                if row["Combination"] == best_combo:
                    return "⭐ 最佳模型 (Best)"
                elif "Baseline" in row["feature_selection"]:
                    return "📍 基準對照 (Baseline)"
                return "🔹 候選模型"

            top10["Type"] = top10.apply(get_color_tag, axis=1)

            fig_rank = px.bar(
                top10,
                x=metric_choice,
                y="Combination",
                orientation="h",
                color="Type",
                color_discrete_map={
                    "⭐ 最佳模型 (Best)": "#E76F51",
                    "📍 基準對照 (Baseline)": "#F4A261",
                    "🔹 候選模型": "#2A9D8F",
                },
                title=f"Top 10 最優模型組合 (依 {metric_choice} 排序)",
            )
            fig_rank.update_layout(
                yaxis={"autorange": "reversed"},
                xaxis_title=metric_label,
                yaxis_title="",
                template="plotly_white",
                height=480,
                margin=dict(l=20, r=20, t=40, b=40),
                legend_title="模型類別",
            )
            st.plotly_chart(fig_rank, use_container_width=True)

        # Baseline vs Best Callout
        baseline_rows = val_df[val_df["feature_selection"] == "Baseline"]
        best_row = val_sorted.iloc[0]
        st.info(f"""
        🏆 **當前指標最優解：** `{best_row['Combination']}` （{metric_choice} = **{best_row[metric_choice]:.3f}{unit}**，特徵數：{best_row['n_features']}）  
        📍 **基準對照組 (Baseline)：** 最佳 Baseline 模型為 `{baseline_rows.sort_values(metric_choice, ascending=not higher_is_better).iloc[0]['loss_function']}` 組合（{metric_choice} = **{baseline_rows[metric_choice].min() if not higher_is_better else baseline_rows[metric_choice].max():.3f}{unit}**，全量 27 特徵）。  
        💡 **結論：** 從 27 個特徵中精選 15 個關鍵變數，在驗證指標上不僅未造成損失，反而消除冗餘噪訊，在多項指標上優於保留全部特徵的 Baseline。
        """)

        # Full 32 Experiments Table
        with st.expander("📋 檢視完整 32 組實驗數據表格", expanded=False):
            display_cols = ["feature_selection", "loss_function", "n_features", "RMSE", "MAE", "R2"]
            formatted_df = val_sorted[display_cols].copy()
            st.dataframe(
                formatted_df.style.format({
                    "RMSE": "{:.2f} Wh",
                    "MAE": "{:.2f} Wh",
                    "R2": "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

# ==============================================================================
# TAB 2: PREDICTION ANALYSIS
# ==============================================================================
with tab2:
    st.subheader("真實預測 vs. 實際觀測與 95% Conformal 預測區間")
    st.markdown("""
    冠軍模型以 **Train + Validation (70%)** 擬合，在獨立的 **Calibration Set (15%)** 上計算保形半寬（Conformal Half-width），
    最終在完全隔離的 **Test Set (15%, 2,961 筆時序樣本)** 評估區間覆蓋率與點預測效果。
    """)

    preds_df = load_metric_file("test_predictions.csv", ["date", "actual", "predicted", "lower_95PI", "upper_95PI"])

    if preds_df is not None:
        p_df = preds_df.copy()
        p_df["lower_95PI"] = np.clip(p_df["lower_95PI"], 0.0, None)  # Physical energy >= 0

        # Interactive slider for time window
        max_pts = len(p_df)
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            n_display = st.slider(
                "選擇時間序列顯示樣本數 (按時間順序展開)",
                min_value=50,
                max_value=min(1000, max_pts),
                value=200,
                step=50,
            )
        with col_s2:
            st.metric("測試集總樣本數", f"{max_pts:,} 筆")

        sample_p = p_df.head(n_display).copy()

        # Time Series with 95% Conformal Prediction Interval Band
        fig_ts = go.Figure()

        # Interval Upper Bound
        fig_ts.add_trace(go.Scatter(
            x=sample_p["date"],
            y=sample_p["upper_95PI"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="95% PI 上界",
        ))

        # Interval Lower Bound with fill to upper
        fig_ts.add_trace(go.Scatter(
            x=sample_p["date"],
            y=sample_p["lower_95PI"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(39, 125, 161, 0.2)",
            name="95% Conformal 預測區間 (下界 ≥ 0 Wh)",
        ))

        # Actual series
        fig_ts.add_trace(go.Scatter(
            x=sample_p["date"],
            y=sample_p["actual"],
            mode="lines",
            line=dict(color="#1D2A44", width=1.5),
            name="實際用電量 (Actual)",
        ))

        # Predicted series
        fig_ts.add_trace(go.Scatter(
            x=sample_p["date"],
            y=sample_p["predicted"],
            mode="lines",
            line=dict(color="#E76F51", width=1.5, dash="solid"),
            name="模型點預測 (Predicted)",
        ))

        fig_ts.update_layout(
            title=f"測試集時間序列預測與 95% Conformal 區間 (前 {n_display} 筆樣本)",
            xaxis_title="觀測時間 (Date / Time)",
            yaxis_title="家電用電量 Appliances (Wh)",
            template="plotly_white",
            height=450,
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        st.markdown("---")

        # Actual vs Predicted Scatter Plot
        col_sc, col_text = st.columns([1.2, 0.8])

        with col_sc:
            st.markdown("#### Actual vs. Predicted 散布圖")
            fig_sc = go.Figure()

            # Scatter points
            fig_sc.add_trace(go.Scatter(
                x=p_df["actual"],
                y=p_df["predicted"],
                mode="markers",
                marker=dict(size=4, color="#2A9D8F", opacity=0.45),
                name="觀測值點",
                hovertemplate="實際用電: %{x} Wh<br>模型預測: %{y:.1f} Wh<extra></extra>",
            ))

            # 45-degree ideal prediction line
            max_val = max(p_df["actual"].max(), p_df["predicted"].max()) + 20
            fig_sc.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="#E63946", width=2, dash="dash"),
                name="理想預測線 (y = x)",
            ))

            fig_sc.update_layout(
                xaxis_title="實際用電量 Actual (Wh)",
                yaxis_title="模型預測用電量 Predicted (Wh)",
                template="plotly_white",
                height=420,
                xaxis=dict(range=[0, max_val]),
                yaxis=dict(range=[0, max_val]),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with col_text:
            st.markdown("#### 散布特徵與預測瓶頸解讀")
            # Interval Coverage metric
            in_interval = (p_df["actual"] >= p_df["lower_95PI"]) & (p_df["actual"] <= p_df["upper_95PI"])
            cov = float(in_interval.mean())

            st.metric("測試集 95% 預測區間經驗涵蓋率", f"{cov * 100:.2f}%", help="名義目標為 95.00%")
            st.warning("""
            ⚠️ **高能耗區間預測壓縮效應：**  
            觀察散布圖可發現，當實際耗電量超過 **250 Wh** 時，預測值大多被壓縮在 **100–250 Wh** 之間，點嚴重偏離 45 度理想線。  
            這是由於家庭高耗電行為（如烹飪或烘衣機瞬間啟動）頻率極低且缺乏專屬特徵，使得線性模型傾向給出回歸均值的穩健預測。
            """)

# ==============================================================================
# TAB 3: RESIDUAL DIAGNOSTICS
# ==============================================================================
with tab3:
    st.subheader("殘差診斷與統計特性 (Residual Diagnostics)")
    st.markdown("""
    殘差定義為 $e_i = y_{\\text{actual}, i} - \\hat{y}_{\\text{pred}, i}$。透過檢驗殘差分布與異質變異數，剖析線性模型之結構性限制。
    """)

    preds_res = load_metric_file("test_predictions.csv", ["actual", "predicted"])

    if preds_res is not None:
        residuals = preds_res["actual"] - preds_res["predicted"]
        res_mean = float(residuals.mean())
        res_median = float(residuals.median())
        res_std = float(residuals.std())
        res_max_pos = float(residuals.max())  # Under-predicted spike
        res_max_neg = float(residuals.min())  # Over-predicted base

        # Summary KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("殘差平均值 (Mean)", f"{res_mean:.2f} Wh")
        k2.metric("殘差中位數 (Median)", f"{res_median:.2f} Wh")
        k3.metric("標準差 (Std Dev)", f"{res_std:.2f} Wh")
        k4.metric("最大正殘差 (嚴重低估尖峰)", f"+{res_max_pos:.2f} Wh")
        k5.metric("最大負殘差 (高估離峰)", f"{res_max_neg:.2f} Wh")

        st.markdown("---")

        r_col1, r_col2 = st.columns(2)

        with r_col1:
            st.markdown("#### 殘差 vs. 預測值 (Residual vs. Predicted)")
            fig_rp = go.Figure()

            fig_rp.add_trace(go.Scatter(
                x=preds_res["predicted"],
                y=residuals,
                mode="markers",
                marker=dict(size=4, color="#264653", opacity=0.4),
                name="殘差觀測點",
                hovertemplate="預測值: %{x:.1f} Wh<br>殘差: %{y:.1f} Wh<extra></extra>",
            ))

            # Horizontal zero line
            fig_rp.add_hline(y=0, line_dash="dash", line_color="#E76F51", line_width=2)

            fig_rp.update_layout(
                xaxis_title="預測用電量 Predicted (Wh)",
                yaxis_title="殘差 Residual = Actual − Predicted (Wh)",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig_rp, use_container_width=True)

        with r_col2:
            st.markdown("#### 殘差機率分布 (Residual Distribution)")
            fig_hist = px.histogram(
                residuals,
                nbins=60,
                marginal="box",
                title="殘差直方圖與箱形圖 (呈現厚右尾分布)",
                color_discrete_sequence=["#2A9D8F"],
            )
            fig_hist.add_vline(x=0, line_dash="dash", line_color="#E76F51", annotation_text="0 Wh")
            fig_hist.add_vline(x=res_mean, line_dash="dot", line_color="#F4A261", annotation_text="Mean")
            fig_hist.update_layout(
                xaxis_title="殘差 (Wh)",
                yaxis_title="次數 (Frequency)",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.error(f"""
        🔍 **統計診斷與物理現象解讀：**
        1. **異質變異數現象 (Heteroscedasticity)：**  
           觀察「殘差 vs. 預測值」散布圖，殘差隨著預測電量增加而顯著向外擴散（呈現右開口漏斗形狀）。這違反了經典 OLS 的同質變異數（Homoscedasticity）假設。
        2. **極端非對稱厚尾 (Heavy Right Tail)：**  
           最大正殘差高達 **{res_max_pos:+.2f} Wh**，而最大負殘差僅 **{res_max_neg:+.2f} Wh**。在一般離峰時段，模型預測略微高估（中位數殘差為 {res_median:.2f} Wh）；但在瞬間大功率用電爆發時，點預測無法及時反應，產生巨大的正殘差。這也是採用 **Conformal 預測區間** 提供不確定性覆蓋保護的必要性所在。
        """)

# ==============================================================================
# TAB 4: BEST MODEL DETAILS
# ==============================================================================
with tab4:
    st.subheader("⭐ 最佳模型中繼檔案 (Best Model Metadata)")
    st.caption("直接載入 outputs/metrics/best_model_metadata.json 內容展示，無需在 Streamlit 重新擬合。")

    metadata_path = METRICS / "best_model_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as exc:
            st.error(f"無法讀取 best_model_metadata.json: {exc}")
            meta = None
    else:
        st.warning("尚無 best_model_metadata.json。請先執行訓練取得完整中繼檔案。")
        meta = None

    if meta is not None:
        # Top Cards
        c1, c2, c3 = st.columns(3)
        c1.metric("最佳模型名稱", meta.get("best_model_name", "N/A"))
        c2.metric("最佳特徵選擇", meta.get("feature_selection", "N/A"))
        c3.metric("最佳損失函數", meta.get("loss_function", "N/A"))

        st.markdown("---")

        # Metrics Breakdown
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🎯 驗證集表現 (Validation Metrics)")
            vm = meta.get("validation_metrics", {})
            st.write(f"- **RMSE:** `{vm.get('RMSE', 0):.3f} Wh`")
            st.write(f"- **MAE:** `{vm.get('MAE', 0):.3f} Wh`")
            st.write(f"- **R²:** `{vm.get('R2', 0):.4f}`")

        with col_m2:
            st.markdown("#### 🏆 測試集泛化表現 (Test Metrics)")
            tm = meta.get("test_metrics", {})
            st.write(f"- **RMSE:** `{tm.get('RMSE', 0):.3f} Wh`")
            st.write(f"- **MAE:** `{tm.get('MAE', 0):.3f} Wh`")
            st.write(f"- **R²:** `{tm.get('R2', 0):.4f}`")
            st.write(f"- **95% 預測區間經驗涵蓋率:** `{tm.get('interval_coverage', 0) * 100:.2f}%`")
            st.write(f"- **區間平均寬度:** `{tm.get('mean_interval_width', 0):.2f} Wh`")

        st.markdown("---")

        # Conformal interval info
        hw = meta.get("conformal_half_width", 0)
        st.success(f"""
        🛡️ **95% Conformal Prediction Interval 半寬度：** `±{hw:.4f} Wh`  
        總預測帶寬度為 `{hw * 2:.4f} Wh`。在測試集 2,961 筆實測中，達成 **{tm.get('interval_coverage', 0) * 100:.2f}%** 的實際覆蓋水準，接近名義 95% 預測覆蓋目標；時間序列可能違反交換性，因此不保證新情境的覆蓋率。
        """)

        # 15 Selected features ordered
        st.markdown("#### 📋 最終選中特徵清單 (保留訓練時嚴格排序，共 15 個)")
        sel_features = meta.get("selected_features", [])
        
        # Display as an indexed table
        feat_df = pd.DataFrame({
            "順序 (Order)": range(1, len(sel_features) + 1),
            "特徵名稱 (Feature Name)": sel_features,
            "特徵物理涵義說明": [
                "照明用電功率 (Wh)" if f == "lights" else
                "室內/外溫度感測器 (°C)" if f.startswith("T") else
                "室內/外濕度感測器 (%)" if f.startswith("RH") else
                "室外風速 (m/s)" if f == "Windspeed" else
                "大氣壓力 (mm Hg)" if f == "Press_mm_hg" else
                "能見度 (km)" if f == "Visibility" else
                "露點溫度 (°C)" if f == "Tdewpoint" else f
                for f in sel_features
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

        # Raw metadata expander
        with st.expander("🔍 檢視完整 JSON Metadata 結構", expanded=False):
            st.json(meta)
