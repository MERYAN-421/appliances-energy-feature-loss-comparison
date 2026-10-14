# Appliance Energy Prediction: Feature Selection × Loss Functions

> A reproducible CRISP-DM project comparing **7 feature-selection strategies**, **4 linear regression objectives**, and an unselected baseline on Kaggle's *Appliances Energy Prediction* dataset.

**Status:** Website scaffold ready · Numerical experiments **not run** · Results and best model **not yet available**.

## Project goals

1. Compare how feature selection changes the information supplied to a linear model.
2. Compare how different objectives penalize prediction errors.
3. Evaluate **32 combinations** (7 selectors + baseline, crossed with 4 objectives) on a consistent chronological validation split.
4. Present validation results, selected features, and final test predictions with a 95% prediction interval in a Streamlit website.

## Dataset and citation

- Kaggle: https://www.kaggle.com/datasets/loveall/appliances-energy-prediction
- Original UCI repository: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
- File expected locally: `data/energydata_complete.csv`
- Target: `Appliances` (Wh)
- Source dataset: ~19,735 rows, 29 columns including target/date.
- This project uses **16 preselected candidate features**; this is **not** a claim that the original Kaggle dataset contains only 10–20 features. Verify that subset-based compliance is accepted by your instructor.

Selected candidate variables:
`T1, T2, T3, T4, T5, RH_1, RH_2, RH_3, RH_4, RH_5, T_out, RH_out, Windspeed, Press_mm_hg, Visibility, lights`

## Experimental design

| Dimension | Methods |
|---|---|
| Feature selection | Pearson correlation, F-test / SelectKBest, mutual information, RFE, forward SFS, backward SFS, LassoCV coefficient ranking |
| Control group | No feature selection (16 predictors) |
| Linear training objectives | Ordinary least squares (squared error), Huber, median quantile regression (absolute error), epsilon-insensitive SGD |
| Subset size | 10 features per selection method; 16 for baseline |
| Model selection | Lowest validation RMSE |
| Evaluation | MAE, RMSE, R², final held-out test prediction interval coverage |

Note: the objective comparison changes the estimator/optimization settings, so differences cannot be attributed to loss alone without further controls. Fix random seeds, scaling, split, feature subsets and tune any regularization within training data.

## CRISP-DM workflow

1. **Business Understanding:** Predict energy consumption and compare which inputs/objectives help.
2. **Data Understanding:** Inspect data, missingness, distributions, correlated sensors.
3. **Data Preparation:** Choose 16 candidates, split chronologically, build leakage-safe preprocessing pipelines.
4. **Modeling:** Fit selectors using training data; train each of four objectives on identical selected columns. Keep a 16-feature baseline for each objective.
5. **Evaluation:** Select configurations on validation RMSE; refit the winner using train + validation; calibrate a 95% prediction interval; evaluate once on untouched test data.
6. **Deployment / Reporting:** Save CSV/PNG outputs, publish Streamlit explanation and link to version-controlled GitHub code.

**Suggested chronological split**: train 55% / validation 15% / calibration 15% / test 15%. For SFS/CV use time-aware folds. Classical exchangeability assumptions for simple conformal prediction intervals may not hold for time-series data; report observed coverage and interval width.

## Quick start (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The **website scaffold runs without the CSV or experiment outputs**. Result pages display a clear missing-results notice until the real CSVs and figures are produced. No fabricated scores are displayed.

## Project structure

```text
app/
  streamlit_app.py       # multipage navigation entrypoint
  pages/                # Home, Dataset, Features, Losses, Results
  utils/common.py       # shared file checks and loaders
data/                    # locally downloaded data (not committed)
docs/GITHUB_WORKFLOW.md  # version-control, deployment and reproducibility guide
outputs/metrics/         # experiment outputs (not yet produced)
outputs/figures/         # plots (not yet produced)
requirements.txt
.gitignore
.streamlit/config.toml
```

## Expected analysis output schema

`outputs/metrics/validation_results.csv` should have columns:

`feature_selection,loss_function,n_features,MAE,RMSE,R2`

There should be one row for each of the 32 configurations. Exact field names are case-sensitive.

`outputs/metrics/selected_features.csv` should have columns:

`feature_selection,feature,selected` (one row per method and candidate feature; `selected` boolean).

`outputs/metrics/test_results.csv` should have columns:

`feature_selection,loss_function,MAE,RMSE,R2,interval_coverage,mean_interval_width`

For unbiased reporting, this file should typically contain **only the selected winning model**, evaluated once on held-out test observations.

`outputs/metrics/test_predictions.csv` should have columns:

`date,actual,predicted,lower_95PI,upper_95PI`

Optionally save `outputs/figures/*.png` for static report reuse. The Results page renders from CSVs when available.

## Deployment

1. Push the repository to GitHub (see [GitHub workflow](docs/GITHUB_WORKFLOW.md)).
2. Open https://share.streamlit.io and create an app from your repository.
3. Select branch `main` and entrypoint `app/streamlit_app.py`.
4. Configure a public URL, deploy and test every page. If the actual experiment CSVs are not committed, the site deliberately remains in **scaffold / awaiting results** state.

Official Streamlit docs: https://docs.streamlit.io/develop/concepts/multipage-apps/page-and-navigation and https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Reproducibility and research safeguards

- Never fit a scaler/selector using validation, calibration, or test information.
- Choose the best combination on validation, **not** test scores.
- Align the four loss models to the same 10-feature subset when comparing objective effects.
- Control preprocessing and regularization; some estimators differ in optimization as well as loss.
- Document hyperparameters, random seeds, software versions, and exact dataset source.
- Check Kaggle/UCI license terms before redistributing the dataset; this skeleton ignores the raw CSV by default.
- Never commit API secrets or private credentials.

## Current limitations

This delivery is the README, Git workflow, and Streamlit **website architecture**. It does not include the complete 32-experiment training engine or verified empirical results. Add the analysis engine and run it to populate the website.

## License and credits

Software license: select a license before distributing reused code. Credit the dataset's original authors and Kaggle/UCI source in your submitted report.
