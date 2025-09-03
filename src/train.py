from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from .preprocess import basic_clean, add_log_transforms, build_preprocessor

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def train_and_eval(df: pd.DataFrame, model_name: str = "rf") -> dict:
    df = basic_clean(df)
    df = add_log_transforms(df)

    target = "IMDB_Rating"
    y = df[target]

    # Constrói o pré-processador, normalizando e codificando as variáveis
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X = df[num_cols + cat_cols]

    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Aplica os modelos
    if model_name == "lin":
        model = LinearRegression()
        model_filename = "imdb_linear.pkl"
    else:
        model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
        model_filename = "imdb_rf.pkl"

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    # Avalia o modelo
    preds = pipe.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    r2   = float(r2_score(y_test, preds))

    # Salva em .pkl
    outpath = MODELS_DIR / model_filename
    joblib.dump(pipe, outpath)

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "path": str(outpath),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }