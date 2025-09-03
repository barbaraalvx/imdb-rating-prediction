from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import joblib

from src.preprocess import basic_clean, add_log_transforms, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "imdb_rf.pkl"

def _to_df(record: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([record])

def _prepare(record: Dict[str, Any]) -> Tuple[pd.DataFrame, list, list]:
    df = _to_df(record)
    df = basic_clean(df)
    df = add_log_transforms(df)

    _, num_cols, cat_cols = build_preprocessor(df)
    return df, num_cols, cat_cols

def load_pipeline(model_path: Optional[Path | str] = None):
    model_path = Path(model_path) if model_path else DEFAULT_MODEL
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
    return joblib.load(model_path)

def predict_one(record: Dict[str, Any], model_path: Optional[Path | str] = None) -> Tuple[float, Dict[str, Any]]:

    df, num_cols, cat_cols = _prepare(record)
    X = df[num_cols + cat_cols]

    pipe = load_pipeline(model_path)
    y_pred = float(pipe.predict(X)[0])

    info = {
        "model_path": str(model_path or DEFAULT_MODEL),
        "used_num_cols": num_cols,
        "used_cat_cols": cat_cols,
        "clean_shape": df.shape,
    }
    return y_pred, info