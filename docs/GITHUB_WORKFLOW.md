# GitHub 工作流與 Streamlit 部署指南

## 一、建立專案（Windows PowerShell）

```powershell
cd <你的專案所在位置>
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

成功後可在本機瀏覽器開啟 Streamlit 顯示的網址（常見為 `http://localhost:8501`）。

## 二、下載資料

從 https://www.kaggle.com/datasets/loveall/appliances-energy-prediction 取得 `energydata_complete.csv`，放至 `data/`；原始資料預設不進 Git。

資料集使用全部 27 個數值候選特徵，Target 為 Appliances，date 用於時間排序。各特徵選擇方法從 27 個特徵中選出 15 個特徵，Baseline 則保留全部 27 個特徵。

## 三、建模工作流程與提交節點

| 階段 | Git 分支 / commit 建議 | 驗收條件 |
|---|---|---|
| 專案初始化 | `chore: initialize project skeleton` | Streamlit 5 頁可以啟動 |
| EDA 與切分 | `feat: add chronological data pipeline` | 欄位、缺失值、切分比例可核對 |
| 7 種 Feature Selection | `feat: implement seven feature selectors` | 訓練資料選 15 特徵且無洩漏 |
| 4 種 Loss 模型 | `feat: compare four linear objectives` | 32 組驗證結果可重現 |
| 圖表輸出 | `feat: export experiment charts and metrics` | RMSE heatmap、ranking、interval 圖 |
| 網站整合 | `feat: connect experimental results to app` | 頁面顯示真實 CSV 結果 |
| 文件與發佈 | `docs: finalize methodology and deployment` | README、來源、網址皆有效 |

## 四、初始化 Git

```bash
git init
git add .
git commit -m "chore: initialize CRISP-DM website scaffold"
git branch -M main
```

先到 GitHub 建立空白 repository，建議名稱：
`appliances-energy-feature-loss-comparison`

```bash
git remote add origin https://github.com/YOUR_USERNAME/appliances-energy-feature-loss-comparison.git
git push -u origin main
```

若 remote 已存在，可用 `git remote -v` 確認；不要重複新增。

## 五、日常修改與版本管理

```bash
git status
git switch -c feature/experiment-engine
# 修改程式與測試
git add .
git commit -m "feat: implement experiment matrix"
git push -u origin feature/experiment-engine
```

在 GitHub 建立 Pull Request，檢查程式、預測圖、資料洩漏風險，再合併到 `main`。

## 六、運算結果如何進網站

訓練程式輸出以下檔案（格式參考 README）：

- `outputs/metrics/validation_results.csv`：32 組驗證指標
- `outputs/metrics/selected_features.csv`：方法與被選取特徵
- `outputs/metrics/test_results.csv`：最佳模型最後測試
- `outputs/metrics/test_predictions.csv`：真實值 / 預測值 / 95% 區間

網站會讀取檔案並渲染圖表；未輸出時會提示「尚未提供實驗結果」。

## 七、部署 Streamlit Community Cloud

1. 將程式與欲展示的結果推到公開或受支援的 GitHub repository。
2. 前往 https://share.streamlit.io，連線 GitHub，選擇 Create app。
3. 選 `main`、檔案路徑 `app/streamlit_app.py`。
4. 確認根目錄 `requirements.txt` 存在。
5. 部署完成後開啟網址，逐頁測試。
6. 把公開網址補回 README 的 Demo 區塊。

Streamlit 文件：
- https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

**注意：** 推送 GitHub 不等於網站自動上線；仍需完成 Streamlit Cloud 的部署操作。GitHub Pages 不會直接執行 Python Streamlit 伺服器。

## 八、交付檢查

- [ ] `README.md` 包含問題、Kaggle 來源、方法、安裝、輸出格式與限制
- [ ] 7 selectors + baseline × 4 objectives = 32 組驗證結果
- [ ] 訓練、驗證、校準、測試切分沒有資料洩漏
- [ ] 有 RMSE、MAE、R² 和 95% prediction interval + coverage
- [ ] 網站的結果來自真正 CSV，不能是假數據
- [ ] GitHub 原始碼可讀、可安裝、可重現
- [ ] 網站 URL 可以開啟、五頁都正常
- [ ] 報告說明使用全部 27 個候選特徵、特徵選擇方法篩選出 15 個特徵
