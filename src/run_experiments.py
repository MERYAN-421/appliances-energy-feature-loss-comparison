"""CRISP-DM Full Experiment Engine: 32 Model Configurations (8 Selectors × 4 Losses).

Workflow:
1. Business & Data Understanding: Load 27 numeric features from Kaggle Appliances Energy dataset.
2. Data Preparation: Missing value verification, chronological split into Train (55%), Val (15%), Cal (15%), Test (15%).
3. Precomputing EDA Artifacts: Export summary statistics, correlations, and sample data for offline/cloud display.
4. Modeling: Fit 8 feature selectors on Train (K=15, Baseline=27); fit 4 loss models on selected features.
5. Validation: Compare 32 combinations using Validation RMSE, MAE, R².
6. Model Selection: Select winning combination based on lowest Validation RMSE.
7. Re-training: Refit winning selector and model pipeline on Train + Validation.
8. Serialization: Save winning model to outputs/best_model.joblib.
9. Calibration: Estimate 95% conformal prediction interval half-width on Calibration set.
10. Evaluation: One-time evaluation on untouched Test set for MAE, RMSE, R², Coverage, Width.
11. Output: Save CSVs to outputs/metrics/ and all 8 static plots to outputs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
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
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"
BEST_MODEL_PATH = OUTPUTS_DIR / "best_model.joblib"
TARGET = "Appliances"
K = 15
RANDOM_STATE = 42


def run(data_path: Path = DATA_PATH):
    print("=" * 75)
    print("🚀 Appliances Energy Prediction: 32 Experiment Execution Engine (Phase 1 Refined)")
    print("=" * 75)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check data
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place energydata_complete.csv there.")

    print(f"\n[1/7] Loading data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=[TARGET])

    # Extract candidate features: all numeric columns except TARGET
    features = df.select_dtypes(include="number").columns.drop(TARGET).tolist()
    print(f"  • Total observations: {len(df):,}")
    print(f"  • Target: {TARGET}")
    print(f"  • Total numeric candidate features ({len(features)}): {', '.join(features)}")

    # Check missing values
    missing_counts = df[features].isna().sum()
    total_missing = int(missing_counts.sum())
    print(f"  • Total missing values across features: {total_missing}")

    # Export EDA artifacts for standalone Streamlit Cloud deployment
    _export_eda_artifacts(df, features)

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

    print("\n[2/7] Chronological data split (Leakage-safe):")
    print(f"  • Training set:    {len(X_train):>5} samples ({len(X_train)/n*100:.1f}%)")
    print(f"  • Validation set:  {len(X_val):>5} samples ({len(X_val)/n*100:.1f}%)")
    print(f"  • Calibration set: {len(X_cal):>5} samples ({len(X_cal)/n*100:.1f}%)")
    print(f"  • Test set:        {len(X_test):>5} samples ({len(X_test)/n*100:.1f}%)")

    # 3. Fit 8 selectors on Training data
    print(f"\n[3/7] Fitting 8 Feature Selection strategies on Train (K={K}, Baseline=27)...")
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
    print("\n[4/7] Training 32 model combinations and evaluating on Validation...")
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
    print("\n" + "=" * 75)
    print(f"🏆 Winning Combination (Lowest Validation RMSE):")
    print(f"   Selector:      {winner_selector}")
    print(f"   Loss Function: {winner_loss}")
    print(f"   Val RMSE:      {best_row['RMSE']:.3f}")
    print(f"   Val MAE:       {best_row['MAE']:.3f}")
    print(f"   Val R²:        {best_row['R2']:.4f}")
    print("=" * 75)

    # 6. Re-fit winner on Train + Validation, calibrate 95% PI, evaluate on Test
    print("\n[5/7] Refitting winning configuration on Train + Validation (70%)...")
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
    print("\n[6/7] Calibrating 95% Conformal Prediction Interval on Calibration set...")
    y_cal_pred = final_pipeline.predict(X_cal[final_features])
    half_width = calibrate_conformal_half_width(y_cal, y_cal_pred, alpha=0.05)
    print(f"Calibrated 95% Prediction Interval Half-Width: ±{half_width:.2f} Wh")

    # One-time untouched Test evaluation
    y_test_pred = final_pipeline.predict(X_test[final_features])
    test_metrics = compute_metrics(y_test, y_test_pred)
    coverage, mean_width, lower_95, upper_95 = evaluate_intervals(y_test, y_test_pred, half_width)
    lower_95_clipped = np.maximum(0.0, lower_95)  # Physical energy >= 0

    # Task 2: Complete Model Bundle (Must be created after calibration)
    bundle = {
        "model": final_pipeline,
        "selected_features": list(final_features),
        "candidate_features": list(features),
        "feature_selection": winner_selector,
        "loss_function": winner_loss,
        "conformal_half_width": float(half_width),
        "target": TARGET,
        "validation_metrics": {
            "RMSE": float(best_row["RMSE"]),
            "MAE": float(best_row["MAE"]),
            "R2": float(best_row["R2"]),
        },
        "test_metrics": {
            "RMSE": float(test_metrics["RMSE"]),
            "MAE": float(test_metrics["MAE"]),
            "R2": float(test_metrics["R2"]),
            "interval_coverage": float(coverage),
            "mean_interval_width": float(mean_width),
        },
    }
    joblib.dump(bundle, BEST_MODEL_PATH)
    print(f"Saved complete serialized best model bundle: {BEST_MODEL_PATH}")

    # Output best_model_metadata.json
    metadata = {
        "best_model_name": f"{winner_selector} × {winner_loss}",
        "feature_selection": winner_selector,
        "loss_function": winner_loss,
        "selected_features": list(final_features),
        "candidate_features": list(features),
        "conformal_half_width": float(half_width),
        "target": TARGET,
        "validation_metrics": {
            "RMSE": float(best_row["RMSE"]),
            "MAE": float(best_row["MAE"]),
            "R2": float(best_row["R2"]),
        },
        "test_metrics": {
            "RMSE": float(test_metrics["RMSE"]),
            "MAE": float(test_metrics["MAE"]),
            "R2": float(test_metrics["R2"]),
            "interval_coverage": float(coverage),
            "mean_interval_width": float(mean_width),
        },
    }
    metadata_json_path = METRICS_DIR / "best_model_metadata.json"
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved best model metadata JSON: {metadata_json_path}")

    print("\n" + "=" * 75)
    print("🎯 Untouched Final Test Set Performance:")
    print(f"   Test RMSE:               {test_metrics['RMSE']:.3f}")
    print(f"   Test MAE:                {test_metrics['MAE']:.3f}")
    print(f"   Test R²:                 {test_metrics['R2']:.4f}")
    print(f"   95% Interval Coverage:   {coverage:.1%}")
    print(f"   Mean Interval Width:     {mean_width:.2f} Wh")
    print("=" * 75)

    # Save metrics to CSV
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
        "lower_95PI": lower_95_clipped,
        "upper_95PI": upper_95,
        "lower_95_raw": lower_95,
        "upper_95_raw": upper_95,
    })
    test_pred_path = METRICS_DIR / "test_predictions.csv"
    test_pred_df.to_csv(test_pred_path, index=False)
    print(f"Saved: {test_pred_path}")

    # 7. Generate all 8 required static figures
    print("\n[7/7] Generating all 8 required static figures...")
    _generate_all_figures(df, features, val_df, selected_df, test_pred_df, winner_selector, winner_loss, coverage)
    print("\n🎉 All 32 experiments completed and all artifacts successfully generated!")


def _export_eda_artifacts(df: pd.DataFrame, features: list[str]):
    """Precomputes and saves EDA tables and summaries for standalone Streamlit Cloud display."""
    # 1. Descriptive stats
    stats_df = df[features].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    stats_df.to_csv(METRICS_DIR / "eda_stats.csv", index=True)

    # 2. Pearson correlations with target
    corr_series = df[features + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()
    corr_df = pd.DataFrame({"Feature": corr_series.index, "Correlation": corr_series.values})
    corr_df.to_csv(METRICS_DIR / "eda_correlations.csv", index=False)

    # 3. Missing values summary
    missing_df = pd.DataFrame({
        "Feature": features,
        "Missing_Count": [df[c].isna().sum() for c in features],
        "Missing_Pct": [df[c].isna().mean() * 100 for c in features],
    })
    missing_df.to_csv(METRICS_DIR / "eda_missing.csv", index=False)

    # 4. First 15 sample rows
    df.head(15).to_csv(METRICS_DIR / "sample_data.csv", index=False)

    # 5. Target sample (1000 points) for cloud distribution rendering
    target_sample = df[TARGET].sample(min(1000, len(df)), random_state=RANDOM_STATE)
    target_sample.to_frame(name=TARGET).to_csv(METRICS_DIR / "target_samples.csv", index=False)

    # 6. Overall summary json
    summary_info = {
        "n_samples": len(df),
        "n_features": len(features),
        "target_name": TARGET,
        "target_mean": float(df[TARGET].mean()),
        "target_std": float(df[TARGET].std()),
        "target_median": float(df[TARGET].median()),
        "target_min": float(df[TARGET].min()),
        "target_max": float(df[TARGET].max()),
        "total_missing": int(df[features].isna().sum().sum()),
    }
    with open(METRICS_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_info, f, indent=2, ensure_ascii=False)


def _generate_all_figures(
    df: pd.DataFrame,
    features: list[str],
    val_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    test_pred_df: pd.DataFrame,
    winner_sel: str,
    winner_loss: str,
    coverage: float,
):
    """Generates all 8 static PNG figures required by specifications."""
    # 1. target_distribution.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    y = df[TARGET].values
    ax1.hist(y, bins=50, color="#277DA1", edgecolor="black", alpha=0.8)
    ax1.set_title("Appliances Energy (Wh) Histogram", fontweight="bold")
    ax1.set_xlabel("Appliances (Wh)")
    ax1.set_ylabel("Frequency")
    ax1.axvline(np.mean(y), color="red", linestyle="--", label=f"Mean: {np.mean(y):.1f}")
    ax1.axvline(np.median(y), color="orange", linestyle="-", label=f"Median: {np.median(y):.1f}")
    ax1.legend()

    ax2.boxplot(y, orientation="horizontal", patch_artist=True, boxprops=dict(facecolor="#F4A261", color="black"))
    ax2.set_title("Appliances Energy Boxplot (Right-Skewed Outliers)", fontweight="bold")
    ax2.set_xlabel("Appliances (Wh)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=200)
    plt.close(fig)

    # 2. correlation_heatmap.png
    corr_mat = df[features + [TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_mat.columns)))
    ax.set_xticklabels(corr_mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(corr_mat.index)))
    ax.set_yticklabels(corr_mat.index, fontsize=8)
    ax.set_title("Correlation Matrix (27 Features + Appliances)", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 3. feature_frequency_heatmap.png
    algo_sel = selected_df[selected_df["feature_selection"] != "Baseline"].copy()
    algo_sel["selected"] = algo_sel["selected"].astype(bool)
    freq = algo_sel[algo_sel["selected"]].groupby("feature").size().reindex(features, fill_value=0).sort_values(ascending=False)
    pivot_sel = selected_df.pivot(index="feature_selection", columns="feature", values="selected").astype(int)
    pivot_sel = pivot_sel[freq.index]  # order by frequency

    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(pivot_sel.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot_sel.columns)))
    ax.set_xticklabels(pivot_sel.columns, rotation=90, fontsize=9)
    ax.set_yticks(range(len(pivot_sel.index)))
    ax.set_yticklabels(pivot_sel.index, fontsize=10)
    ax.set_title("Feature Selection Frequency Heatmap (7 Selectors × 27 Features)", fontsize=12, fontweight="bold")
    for i in range(len(pivot_sel.index)):
        for j in range(len(pivot_sel.columns)):
            v = pivot_sel.iloc[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v else "#666", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_frequency_heatmap.png", dpi=200)
    plt.close(fig)

    # 4. rmse_heatmap.png
    pivot_rmse = val_df.pivot_table(index="feature_selection", columns="loss_function", values="RMSE")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    im = ax.imshow(pivot_rmse.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot_rmse.columns)))
    ax.set_xticklabels(pivot_rmse.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot_rmse.index)))
    ax.set_yticklabels(pivot_rmse.index, fontsize=9)
    for i in range(len(pivot_rmse.index)):
        for j in range(len(pivot_rmse.columns)):
            val = pivot_rmse.iloc[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Validation RMSE Matrix (8 Selectors × 4 Losses)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="RMSE (Wh)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rmse_heatmap.png", dpi=200)
    plt.close(fig)

    # 5. model_ranking.png
    top10 = val_df.nsmallest(10, "RMSE").copy()
    top10["label"] = top10["feature_selection"] + " × " + top10["loss_function"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(top10)), top10["RMSE"], color="#277DA1", edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["label"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Validation RMSE (Wh)")
    ax.set_title("Top 10 Model Configurations Ranked by Validation RMSE", fontsize=12, fontweight="bold")
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.3, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_ranking.png", dpi=200)
    plt.close(fig)

    # 6. actual_vs_predicted.png
    sample = test_pred_df.head(200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.scatter(test_pred_df["actual"], test_pred_df["predicted"], alpha=0.25, color="#277DA1", s=15)
    max_val = max(test_pred_df["actual"].max(), test_pred_df["predicted"].max())
    ax1.plot([0, max_val], [0, max_val], "r--", label="Ideal 1:1 line")
    ax1.set_xlabel("Actual Energy (Wh)")
    ax1.set_ylabel("Predicted Energy (Wh)")
    ax1.set_title("Actual vs. Predicted Scatter (Test Set)", fontweight="bold")
    ax1.legend()

    x = range(len(sample))
    ax2.plot(x, sample["actual"], label="Actual", color="#264653", lw=1.2)
    ax2.plot(x, sample["predicted"], label="Predicted", color="#E76F51", lw=1.2)
    ax2.set_xlabel("Chronological Test Index (First 200)")
    ax2.set_ylabel("Energy (Wh)")
    ax2.set_title("Time-Series Actual vs. Predicted", fontweight="bold")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    # 7. residual_plot.png
    residuals = test_pred_df["actual"] - test_pred_df["predicted"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.scatter(test_pred_df["predicted"], residuals, alpha=0.25, color="#457B9D", s=15)
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Predicted Energy (Wh)")
    ax1.set_ylabel("Residual (Actual - Predicted)")
    ax1.set_title("Residuals vs. Fitted Values", fontweight="bold")

    ax2.hist(residuals, bins=50, color="#E63946", edgecolor="black", alpha=0.75)
    ax2.axvline(0, color="black", linestyle="--")
    ax2.set_xlabel("Residual (Wh)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual Distribution (Mean={residuals.mean():.2f})", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_plot.png", dpi=200)
    plt.close(fig)

    # 8. prediction_interval.png
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = range(len(sample))
    ax.plot(x, sample["actual"], label="Actual", color="#264653", lw=1.2)
    ax.plot(x, sample["predicted"], label="Predicted", color="#E76F51", lw=1.2)
    ax.fill_between(
        x,
        sample["lower_95PI"],
        sample["upper_95PI"],
        color="#E76F51",
        alpha=0.2,
        label=f"95% Conformal Prediction Interval (Coverage: {coverage:.1%})",
    )
    ax.set_xlabel("Chronological Test Sample Index (First 200)")
    ax.set_ylabel("Appliances Energy (Wh)")
    ax.set_title(f"Test Predictions with 95% PI | {winner_sel} × {winner_loss}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "prediction_interval.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run()
