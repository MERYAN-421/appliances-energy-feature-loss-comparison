# Regression explorer integration

Baseline: 39585a0f5485b256ac2d8b7c415ec32f8843577b.
Preserve src's original 32-model engine and all outputs. Add a separate explorer,
training-only selection, validation K sweeps, OLS interpretation, saved-model live
prediction, a second problem, and a bounded custom CSV workspace.

The research pages remain available. New models never inherit the original
model's conformal calibration. Test data never choose selectors or K.
Milestones: interpretation; K sweep; Startups; custom data; integration tests.

## Implemented navigation

- Regression Explorer (default): dropdown for 50 Startups and Appliances;
  training-only selection, validation K curves, five-method comparison for small
  problems, equations, OLS inference, prediction and diagnostics.
- Original Model Live Prediction: original fitted research bundle, original
  feature ordering, training-derived imputer medians, calibrated interval.
- Custom Dataset: CSV upload or approved raw.githubusercontent.com HTTPS URL;
  target and predictor selection; session-only preprocessing, models and data.
- Original five research pages remain available.

## Statistical and computational choices

Startups uses random_state=42 with 60/20/20 train/validation/test. Appliances
keeps chronological 55/15/15/15; the explorer does not use calibration rows or
inherit the original model's interval. K is the number of transformed columns,
including individual State dummy columns. Selection never sees validation or test
targets. SFS/exhaustive CV refits preprocessing inside each training fold and
rejects folds with inconsistent categorical feature mappings.

Lasso and ElasticNet are coefficient rankings with fixed alpha=.001 on scaled
training targets; the final reported model is OLS on the selected columns.
They do not claim to reproduce a separate site's tuning or numerical results.
Exhaustive means minimum training-CV MSE across all size-K subsets; SFS is greedy.
Both are limited to 1,000 training rows and 8 transformed features. Large problems
use Lasso, ElasticNet, RFE or F-test and a sparse K grid. Public contexts/results
are bounded cached objects; custom data never enter shared caches.

OLS inference is on the actual training fit in transformed coordinates. It uses
classical covariance, residual degrees of freedom, t intervals and a training
F-test, with explicit assumptions and post-selection limitations. Rank-deficient
fits suppress classical inference. Validation adjusted R² is descriptive, with
N/A when its denominator is nonpositive. Test metrics are available only for the
K selected by validation RMSE. They are not used to change the selected K.

## Reproduce and deploy

Use Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
bash scripts/validate.sh
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Validation provisions the ignored Appliances CSV from the author's public data
repository with a pinned SHA-256, because the legacy test's fallback URL is 404.
It runs the new unittest suite, original reproducibility script and original
five-page/standalone tests. Do not run the old page suite concurrently: its
standalone test temporarily moves the CSV to a fixed /tmp path.

Optional browser verification (requires Chromium and `pip install playwright`):
start Streamlit on port 8502, then run `.venv/bin/python scripts/browser_smoke.py`.
EXPLORER_URL and CHROMIUM_PATH override the defaults. It exercises an actual CSV
upload, navigation, training and live prediction. Test screenshots go to /tmp.

Streamlit Community Cloud: select this repository, branch
`feat/final-regression-explorer`, entrypoint `app/streamlit_app.py`, Python 3.12.
No API keys are required. The bundled Startups example and original research
pages/live model work without raw Appliances CSV. Appliances K analysis requires
the raw CSV; it shows an explicit message if unavailable. For full Appliances K
analysis on Community Cloud, provision that public CSV through the deployment's
supported setup process; do not commit user-uploaded datasets or run the full
research engine during startup. Public URL loading requires HTTPS access to
raw.githubusercontent.com. The upload limit is 5 MB at both server and parser.

The saved model requires scikit-learn 1.9.0, pinned in requirements. No source,
model bundle, figure, or metric in src/ and outputs/ is changed. Deployment must
be completed by an authenticated Streamlit Cloud account; local HTTP readiness
and repository submission are distinct from a live deployment.
