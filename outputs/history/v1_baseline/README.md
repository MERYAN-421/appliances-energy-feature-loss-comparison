# 歷史實驗版本 v1 (Baseline Run)

- **審查 Commit SHA**：`9751c2776e9fe192cd145da81e538d83f6776ad7`
- **產生時間**：Phase 1 修正前（未標準化特徵選擇器）
- **說明**：
  - 本目錄封存專案第一版 32 組模型實驗數據與圖表。
  - 特點：RFE 與 LassoCV 特徵選擇未先進行尺度標準化，SGDRegressor 採用 epsilon=0.1。
  - 作為與 Phase 1 改進後之 v2 實驗結果之對照基準。
