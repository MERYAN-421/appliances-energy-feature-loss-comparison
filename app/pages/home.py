import streamlit as st
from utils.common import DATASET_URL, METRICS

st.title("⚡ 家電能源消耗預測實驗室 (Appliances Energy Lab)")
st.caption("Feature Selection × Loss Functions | CRISP-DM 機器學習全流程 | 互動式研究專案")

st.markdown("""
**核心研究問題 (Research Question)：**
在多元線性迴歸框架下，**7 種特徵選擇策略（各挑選 15 個特徵）** 與 **4 種不同損失目標（OLS、Huber、L1 中位數、Epsilon-insensitive）** 的交叉組合（共 32 種配置），如何影響家電用電量（Wh）的預測準確度與穩健性？
""")

col1, col2, col3, col4 = st.columns(4)
col1.metric("數值候選特徵", "27 個")
col2.metric("特徵選擇維度", "7 種 + Baseline")
col3.metric("損失目標維度", "4 種 Loss")
col4.metric("交叉實驗矩陣", "32 組模型")

st.markdown("---")

st.subheader("💡 兩大核心設計維度")
c1, c2 = st.columns(2)
with c1:
    st.info("""
    **維度一：特徵選擇（Feature Selection）**
    * **核心提問：** *哪些資訊該進入模型？*
    * **操作方式：** 從原始 27 個數值特徵中，篩選最具代表性的 15 個特徵（降低共線性與維度災難）。
    * **7 種演算法：** Pearson 相關係數、F-test、互資訊（Mutual Information）、遞迴特徵消除（RFE）、前向 SFS、後向 SFS、LassoCV 係數排序。
    """)
with c2:
    st.success("""
    **維度二：損失函數（Loss Functions）**
    * **核心提問：** *預測殘差如何受到懲罰？*
    * **操作方式：** 固定特徵子集，比較 4 種對誤差不同敏感度的線性優化目標。
    * **4 種目標：**
      1. **Squared Error (OLS)**：二次方殘差，對大誤差懲罰重。
      2. **Huber Loss**：近零二次、遠處線性，兼顧平滑與抗離群值。
      3. **Absolute Error (Median)**：L1 損失，估計中位數，抗偏態。
      4. **Epsilon-insensitive**：小誤差不懲罰，專注管外顯著殘差。
    """)

st.subheader("🔄 CRISP-DM 嚴謹實驗流程")
st.markdown("""
1. **Business Understanding：** 預測家庭用電量高峰，找出關鍵氣象與室內溫濕度驅動因子。
2. **Data Understanding：** 探勘 Kaggle 19,735 筆時間序列記錄（27 個數值特徵）。
3. **Data Preparation：** 依時間順序切分 **Train (55%) / Val (15%) / Cal (15%) / Test (15%)**，杜絕未來資訊洩漏（No Data Leakage）。
4. **Modeling：** 8 種選擇器在 Train 上擬合，產生 32 組模型並於 Validation 評估。
5. **Evaluation：** 依最低 Validation RMSE 決定冠軍，以 Train+Val 重新擬合特徵與模型，在 Calibration 集校準 **95% Conformal 預測區間**，最後在完全隔離的 Test 集評估 Coverage 與 MAE/RMSE/$R^2$。
6. **Deployment：** 輸出指標 CSV 與 Streamlit 互動視覺化平台。
""")

# Check experiment status
val_file = METRICS / "validation_results.csv"
if val_file.exists():
    st.success("✅ **實驗已完成！** 真實 32 組模型運算數據已儲存於 `outputs/metrics/`，請透過左側選單瀏覽各分析頁面。")
else:
    st.warning("⚠️ 尚未執行實驗。請在終端機執行 `python -m src.run_experiments` 產生真實結果。")

st.caption(f"資料來源：[Kaggle — Appliances Energy Prediction Dataset]({DATASET_URL})")
