from pathlib import Path
from typing import List, Optional
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
DATA_FILE = ROOT / "data" / "energydata_complete.csv"
DATASET_URL = "https://www.kaggle.com/datasets/loveall/appliances-energy-prediction"

# 27 candidate numeric features
DEFAULT_FEATURES = [
    "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4",
    "T5", "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9",
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "rv1", "rv2"
]
TARGET = "Appliances"
SELECTORS = ["Pearson", "F-test", "Mutual Information", "RFE", "Forward SFS", "Backward SFS", "LassoCV ranking", "Baseline"]
LOSSES = ["Squared Error (OLS)", "Huber", "Absolute Error (Median)", "Epsilon-insensitive"]


@st.cache_data(show_spinner=False)
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def get_feature_list() -> List[str]:
    """Gets the candidate features list from dataset if available, otherwise fallback."""
    if DATA_FILE.exists():
        try:
            df_head = pd.read_csv(DATA_FILE, nrows=5)
            num_cols = df_head.select_dtypes(include="number").columns.drop(TARGET, errors="ignore").tolist()
            if num_cols:
                return num_cols
        except Exception:
            pass
    return DEFAULT_FEATURES


FEATURES = get_feature_list()


def load_metric_file(filename: str, required: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Loads a metric CSV from outputs/metrics/ and verifies required columns."""
    path = METRICS / filename
    if not path.exists():
        st.info(f"尚無實驗檔案：`outputs/metrics/{filename}`。請先執行 `python -m src.run_experiments` 完成實際建模並輸出 CSV。")
        return None
    try:
        df = read_csv(path)
    except Exception as exc:
        st.error(f"無法讀取 {filename}: {exc}")
        return None
    if required:
        missing = sorted(set(required) - set(df.columns))
        if missing:
            st.error(f"{filename} 缺少必要欄位：{', '.join(missing)}")
            return None
    return df
