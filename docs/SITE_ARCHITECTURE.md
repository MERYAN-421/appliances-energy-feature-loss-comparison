# Streamlit 網站架構

入口：`app/streamlit_app.py`，以 `st.Page` / `st.navigation` 註冊以下 5 頁：

1. `pages/home.py`：研究目的、CRISP-DM、Feature Selection 與 Loss 差異、32 組實驗設計。
2. `pages/dataset.py`：資料來源與候選特徵，載入本機 CSV 後顯示真實分布、相關性。
3. `pages/feature_selection.py`：七種方法介紹，讀 `selected_features.csv` 展示特徵頻率。
4. `pages/loss_functions.py`：四種 loss 公式、效果與互動式損失曲線（僅教學示意）。
5. `pages/results.py`：從真實 `validation_results.csv` 生成 RMSE heatmap / top-10；從 final test CSV 繪製 95% 區間。

分析演算法與網站分離：網站只讀資料與已輸出的驗證結果，避免每次訪問重訓昂貴的 32 個模型。正式網站需把輸出的 CSV push 到 GitHub，才會有可見評估圖。若未提供檔案，顯示明確說明，不產生假成績。

部署入口：`app/streamlit_app.py`；相依套件：根目錄 `requirements.txt`。網站頁面以中文導覽、英文技術術語介紹，適合課堂 demo。
