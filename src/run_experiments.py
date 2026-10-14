"""CRISP-DM Full Experiment Engine: 32 Model Configurations (8 Selectors × 4 Losses).

Workflow:
1. Business & Data Understanding: Load 27 numeric features from Kaggle Appliances Energy dataset.
2. Data Preparation: Chronological split into Train (55%), Val (15%), Cal (15%), Test (15%).
3. Modeling: Fit 8 feature selectors on Train (K=15, Baseline=27); fit 4 loss models on selected features.
4. Validation: Compare 32 combinations using Validation RMSE, MAE, R².
5. Model Selection: Select winning combination based on lowest Validation RMSE.
6. Re-training: Refit winning selector and model on Train + Validation.
7. Calibration: Estimate 95% conformal prediction interval half-width on Calibration set.
8. Evaluation: One-time evaluation on untouched Test set for MAE, RMSE, R², Coverage, Width.
9. Output: Save CSVs to outputs/metrics/ and static plots to outputs/figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.feature_selection import get_selectors, extract_selected_features
from src.loss_models import get_loss_models, build_loss_pipeline
from src.evaluation import compute_metrics, calibrate_conformal_half_width, evaluate_intervals

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "energydata_complete.csv"
METRICS_DIR = ROOT / "outputs" / "metrics"
FIGURES_DIR = ROOT / "outputs" / "figures"
TARGET = "Appliances"
K = 15
RANDOM_STATE = 42


def run(data_path: Path = DATA_PATH):
    print("=" * 70)
    print("🚀 Appliances Energy Prediction: 32 Experiment Execution Engine")
    print("=" * 70)

    # 1. Check data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place energydata_complete.csv there.")

    print(f"\n[1/6] Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])
    
    # Extract candidate features: all numeric columns except TARGET
    features = df.select_dtypes(include="number").columns.drop(TARGET).tolist()
    print(f"Total observations: {len(df):,}")
    print(f"Target: {TARGET}")
    print(f"Total numeric candidate features ({len(features)}): {', '.join(features)}")

    # 2. Chronological data split (55% / 15% / 15% / 15%)
    n = len(df)
    i1, i2, i3 = int(0.55 * n), int(0.70 * n), int(0.85 * n)
    X = df[features]
    y = df[TARGET]

    X_train, y_train = X.iloc[:i1], y.iloc[:i1]
    X_val, y_val = X.iloc[i1:i2], y.iloc[i1:i2]
    X_cal, y_cal = X.iloc[i2:i3], y.iloc[i2:i3]
    X_test, y_test = X.iloc[i3:], y.iloc[i3:]
    dates_test = df["date"].iloc[i3:].values

    print("\n[2/6] Chronological data split (Leakage-safe):")
    print(f"  • Training set:    {len(X_train):>5} samples ({len(X_train)/n*100:.1f}%)")
    print(f"  • Validation set:  {len(X_val):>5} samples ({len(X_val)/n*100:.1f}%)")
    print(f"  • Calibration set: {len(X_cal):>5} samples ({len(X_cal)/n*100:.1f}%)")
    print(f"  • Test set:        {len(X_test):>5} samples ({len(X_test)/n*100:.1f}%)")

    # 3. Fit 8 selectors on Training data
    print(f"\n[3/6] Fitting 8 Feature Selection strategies on Train (K={K})...")
    selectors = get_selectors(k=K, random_state=RANDOM_STATE)
    selected_features_dict = {}
    selected_features_rows = []

    for sel_name, selector in selectors.items():
        print(f"  → Fitting selector: {sel_name:20s}", end="", flush=True)
        selector.fit(X_train, y_train)
        sel_cols = extract_selected_features(selector, features)
        selected_features_dict[sel_name] = sel_cols
        print(f"Done ({len(sel_cols)} features selected)")
        
        for feat in features:
            selected_features_rows.append({
                "feature_selection": sel_name,
                "feature": feat,
                "selected": (feat in sel_cols),
            })

    # 4. Train 32 configurations and evaluate on Validation
    print("\n[4/6] Training 32 model combinations and evaluating on Validation...")
    loss_models = get_loss_models(random_state=RANDOM_STATE)
    validation_rows = []

    for sel_name, sel_cols in selected_features_dict.items():
        X_train_sub = X_train[sel_cols]
        X_val_sub = X_val[sel_cols]

        for loss_name, regressor in loss_models.items():
            print(f"  • [{sel_name:18s}] × [{loss_name:24s}] ... ", end="", flush=True)
            pipeline = build_loss_pipeline(regressor)
            pipeline.fit(X_train_sub, y_train)
            y_val_pred = pipeline.predict(X_val_sub)

            m = compute_metrics(y_val, y_val_pred)
            validation_rows.append({
                "feature_selection": sel_name,
                "loss_function": loss_name,
                "n_features": len(sel_cols),
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
            })
            print(f"Val RMSE: {m['RMSE']:.3f} | MAE: {m['MAE']:.3f} | R2: {m['R2']:.4f}")

    val_df = pd.DataFrame(validation_rows).sort_values("RMSE", ascending=True).reset_index(drop=True)
    selected_df = pd.DataFrame(selected_features_rows)

    # 5. Identify the winner
    best_row = val_df.iloc[0]
    winner_selector = best_row["feature_selection"]
    winner_loss = best_row["loss_function"]
    print("\n" + "=" * 70)
    print(f"🏆 Winning Combination (Lowest Validation RMSE):")
    print(f"   Selector:      {winner_selector}")
    print(f"   Loss Function: {winner_loss}")
    print(f"   Val RMSE:      {best_row['RMSE']:.3f}")
    print(f"   Val MAE:       {best_row['MAE']:.3f}")
    print(f"   Val R²:        {best_row['R2']:.4f}")
    print("=" * 70)

    # 6. Re-fit winner on Train + Validation, calibrate 95% PI, evaluate on Test
    print("\n[5/6] Refitting winning configuration on Train + Validation...")
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    # Re-run feature selection on Train + Validation
    refit_selectors = get_selectors(k=K, random_state=RANDOM_STATE)
    final_selector = refit_selectors[winner_selector]
    final_selector.fit(X_train_val, y_train_val)
    final_features = extract_selected_features(final_selector, features)
    print(f"Final selected features ({len(final_features)}): {', '.join(final_features)}")

    # Refit winning loss model on Train + Validation
    final_regressor = get_loss_models(random_state=RANDOM_STATE)[winner_loss]
    final_pipeline = build_loss_pipeline(final_regressor)
    final_pipeline.fit(X_train_val[final_features], y_train_val)

    # Calibrate on Calibration set
    print("\n[6/6] Calibrating 95% Conformal Prediction Interval on Calibration set...")
    y_cal_pred = final_pipeline.predict(X_cal[final_features])
    half_width = calibrate_conformal_half_width(y_cal, y_cal_pred, alpha=0.05)
    print(f"Calibrated 95% Prediction Interval Half-Width: ±{half_width:.2f} Wh")

    # One-time untouched Test evaluation
    y_test_pred = final_pipeline.predict(X_test[final_features])
    test_metrics = compute_metrics(y_test, y_test_pred)
    coverage, mean_width, lower_95, upper_95 = evaluate_intervals(y_test, y_test_pred, half_width)

    print("\n" + "=" * 70)
    print("🎯 Untouched Final Test Set Performance:")
    print(f"   Test RMSE:               {test_metrics['RMSE']:.3f}")
    print(f"   Test MAE:                {test_metrics['MAE']:.3f}")
    print(f"   Test R²:                 {test_metrics['R2']:.4f}")
    print(f"   95% Interval Coverage:   {coverage:.1%}")
    print(f"   Mean Interval Width:     {mean_width:.2f} Wh")
    print("=" * 70)

    # Save metrics to CSV
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    val_csv_path = METRICS_DIR / "validation_results.csv"
    val_df.to_csv(val_csv_path, index=False)
    print(f"\nSaved: {val_csv_path}")

    sel_csv_path = METRICS_DIR / "selected_features.csv"
    selected_df.to_csv(sel_csv_path, index=False)
    print(f"Saved: {sel_csv_path}")

    test_res_df = pd.DataFrame([{
        "feature_selection": winner_selector,
        "loss_function": winner_loss,
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "R2": test_metrics["R2"],
        "interval_coverage": coverage,
        "mean_interval_width": mean_width,
    }])
    test_res_path = METRICS_DIR / "test_results.csv"
    test_res_df.to_csv(test_res_path, index=False)
    print(f"Saved: {test_res_path}")

    test_pred_df = pd.DataFrame({
        "date": pd.to_datetime(dates_test).strftime("%Y-%m-%d %H:%M:%S"),
        "actual": y_test.values,
        "predicted": y_test_pred,
        "lower_95PI": lower_95,
        "upper_95PI": upper_95,
        "lower_95": lower_95,
        "upper_95": upper_95,
    })
    test_pred_path = METRICS_DIR / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved: {test_pred_path}")

    # Generate static figures for reports
    _generate_figures(val_df, test_pred_df, winner_selector, winner_loss, coverage)
    print("\nAll experiments successfully completed and artifacts generated!")


def _generate_figures(val_df: pd.DataFrame, test_pred_df: pd.DataFrame, winner_sel: str, winner_loss: str, coverage: float):
    """Generates report-ready static figures."""
    # 1. Validation RMSE Heatmap
    pivot = val_df.pivot_table(index="feature_selection", columns="loss_function", values="RMSE")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Validation RMSE Matrix (8 Selectors × 4 Losses)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="RMSE (Wh)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rmse_heatmap.png", dpi=200)
    plt.close(fig)

    # 2. Top-10 RMSE Bar Chart
    top10 = val_df.nsmallest(10, "RMSE").copy()
    top10["label"] = top10["feature_selection"] + " × " + top10["loss_function"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(range(len(top10)), top10["RMSE"], color="#277DA1", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Validation RMSE (Wh)")
    ax.set_title("Top 10 Configurations by Validation RMSE", fontsize=12, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top10_rmse.png", dpi=200)
    plt.close(fig)

    # 3. Test Predictions with 95% Prediction Interval
    sample = test_pred_df.head(200)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = range(len(sample))
    ax.plot(x, sample["actual"], label="Actual", color="#264653", lw=1.2)
    ax.plot(x, sample["predicted"], label="Predicted", color="#E76F51", lw=1.2)
    ax.fill_between(x, sample["lower_95PI"], sample["upper_95PI"], color="#E76F51", alpha=0.2, label=f"95% Prediction Interval (Observed: {coverage:.1%})")
    ax.set_xlabel("Chronological Test Sample Index")
    ax.set_ylabel("Appliances Energy (Wh)")
    ax.set_title(f"Test Predictions & Calibrated 95% PI | {winner_sel} × {winner_loss}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_intervals.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
