"""Model Reproducibility Verification Test.

Task 4 verification:
1. Load best_model.joblib and extract bundle metadata.
2. Confirm ordered selected features match pipeline input expectations.
3. Re-run inference on test partition with identical feature ordering.
4. Verify predictions match outputs/metrics/test_predictions.csv.
5. Reconstruct 95% Conformal Prediction Intervals and verify consistency.
6. Report maximum absolute discrepancies for point predictions and intervals.
"""

import json
from pathlib import Path
import urllib.request
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "outputs" / "metrics" / "best_model_metadata.json"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "metrics" / "test_predictions.csv"
DATA_PATH = PROJECT_ROOT / "data" / "energydata_complete.csv"
RAW_DATA_URL = (
    "https://raw.githubusercontent.com/LuisM78/"
    "Appliances-energy-prediction-for-regression-analysis/master/energydata_complete.csv"
)


def ensure_dataset():
    """Ensure energydata_complete.csv is available locally."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Dataset not found locally. Downloading from {RAW_DATA_URL} ...")
        urllib.request.urlretrieve(RAW_DATA_URL, DATA_PATH)
        print("Download complete.")
    return pd.read_csv(DATA_PATH)


def run_reproducibility_test():
    print("=" * 70)
    print("Starting Model Reproducibility Verification Test (Task 4)")
    print("=" * 70)

    # 1. Load model bundle
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    print(f"[1/5] Loading bundle from: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)

    required_keys = [
        "model",
        "selected_features",
        "candidate_features",
        "feature_selection",
        "loss_function",
        "conformal_half_width",
    ]
    for key in required_keys:
        assert key in bundle, f"Missing key in bundle: '{key}'"

    model = bundle["model"]
    selected_features = bundle["selected_features"]
    half_width = bundle["conformal_half_width"]
    winner_selector = bundle["feature_selection"]
    winner_loss = bundle["loss_function"]

    print(f"      Best Configuration : {winner_selector} × {winner_loss}")
    print(f"      Selected Features  : {len(selected_features)} features")
    print(f"      Ordered List       : {selected_features}")
    print(f"      Conformal Half-W   : ±{half_width:.4f} Wh")

    # 2. Check metadata consistency
    print("\n[2/5] Validating against best_model_metadata.json...")
    assert METADATA_PATH.exists(), f"Metadata not found: {METADATA_PATH}"
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["selected_features"] == selected_features, (
        "Selected features order in metadata does not match bundle!"
    )
    assert np.isclose(metadata["conformal_half_width"], half_width), (
        "Conformal half-width does not match metadata!"
    )
    print("      Metadata consistency verified.")

    # 3. Load dataset and reproduce test split
    print(f"\n[3/5] Loading dataset from: {DATA_PATH}")
    df = ensure_dataset()
    n = len(df)
    n_train = int(0.55 * n)
    n_val = int(0.15 * n)
    n_cal = int(0.15 * n)
    split_test = n_train + n_val + n_cal

    df_test = df.iloc[split_test:].copy()
    y_test = df_test["Appliances"].to_numpy(dtype=float)
    X_test = df_test[selected_features]

    print(f"      Total samples: {n:,}")
    print(f"      Test partition: {len(df_test):,} samples (rows {split_test} to {n})")
    print(f"      Feature matrix shape: {X_test.shape}")

    # 4. Generate predictions with identical feature ordering
    print("\n[4/5] Generating predictions with serialized pipeline...")
    y_pred_reproduced = model.predict(X_test)

    # Reconstruct 95% prediction interval
    lower_raw = y_pred_reproduced - half_width
    upper_raw = y_pred_reproduced + half_width
    lower_95pi = np.clip(lower_raw, 0, None)
    upper_95pi = upper_raw

    # 5. Compare with test_predictions.csv
    print(f"\n[5/5] Comparing against recorded: {PREDICTIONS_PATH}")
    assert PREDICTIONS_PATH.exists(), f"Predictions file not found: {PREDICTIONS_PATH}"
    recorded_preds = pd.read_csv(PREDICTIONS_PATH)

    assert len(recorded_preds) == len(y_pred_reproduced), (
        f"Row count mismatch: recorded {len(recorded_preds)} vs reproduced {len(y_pred_reproduced)}"
    )

    pred_abs_diff = np.abs(recorded_preds["predicted"].to_numpy() - y_pred_reproduced)
    max_pred_diff = float(np.max(pred_abs_diff))
    mean_pred_diff = float(np.mean(pred_abs_diff))

    lower_abs_diff = np.abs(recorded_preds["lower_95PI"].to_numpy() - lower_95pi)
    max_lower_diff = float(np.max(lower_abs_diff))

    upper_abs_diff = np.abs(recorded_preds["upper_95PI"].to_numpy() - upper_95pi)
    max_upper_diff = float(np.max(upper_abs_diff))

    coverage_reproduced = float(np.mean((y_test >= lower_95pi) & (y_test <= upper_95pi)))

    print("=" * 70)
    print("VERIFICATION RESULTS & ACCEPTANCE METRICS")
    print("=" * 70)
    print(f"  Maximum Absolute Discrepancy (Predicted y) : {max_pred_diff:.2e} Wh")
    print(f"  Mean Absolute Discrepancy (Predicted y)    : {mean_pred_diff:.2e} Wh")
    print(f"  Maximum Discrepancy (Lower 95% PI)         : {max_lower_diff:.2e} Wh")
    print(f"  Maximum Discrepancy (Upper 95% PI)         : {max_upper_diff:.2e} Wh")
    print(f"  Reconstructed 95% PI Empirical Coverage   : {coverage_reproduced * 100:.2f}%")
    print(f"  Recorded Test 95% PI Coverage in Metadata : {metadata['test_metrics']['interval_coverage'] * 100:.2f}%")

    tolerance = 1e-9
    assert max_pred_diff < tolerance, f"Prediction discrepancy {max_pred_diff} exceeds tolerance {tolerance}!"
    assert max_lower_diff < tolerance, f"Lower PI discrepancy {max_lower_diff} exceeds tolerance {tolerance}!"
    assert max_upper_diff < tolerance, f"Upper PI discrepancy {max_upper_diff} exceeds tolerance {tolerance}!"
    assert np.isclose(coverage_reproduced, metadata["test_metrics"]["interval_coverage"]), "Coverage mismatch!"

    print("-" * 70)
    print(">>> ALL REPRODUCIBILITY CHECKS PASSED (discrepancy < 1e-9). <<<")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_reproducibility_test()
