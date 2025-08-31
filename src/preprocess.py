from pathlib import Path
from typing import Optional, List

import pandas as pd

# Imputar valor "Unknown" em Certificate
def clean_certificate(df: pd.DataFrame, col: str = "Certificate") -> pd.DataFrame:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype("category")
    return df

# Eliminar colunas (primeiro a "Unnamed: 0", mas também permite eliminar uma lista de colunas) 
def drop_redundant_columns(df: pd.DataFrame, extra_to_drop: Optional[List[str]] = None) -> pd.DataFrame:
    
    cols_to_drop = []
    
    if "Unnamed: 0" in df.columns:
        cols_to_drop.append("Unnamed: 0")
    
    if extra_to_drop:
        cols_to_drop.extend([c for c in extra_to_drop if c in df.columns])

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df

# Converter Gross para numérico
def parse_gross(df: pd.DataFrame, col: str = "Gross") -> pd.DataFrame:
    
    if col in df.columns:
        df[col] = (
            df[col].astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"nan": None, "None": None, "": None})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df

# Converte ano (object) para numérico
def to_numeric_year(df: pd.DataFrame, col: str = "Released_Year") -> pd.DataFrame:
    
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[col])
    
    return df

# Meta_score (imputar média)
def impute_meta_score(df: pd.DataFrame, col: str = "Meta_score") -> pd.DataFrame:

    if col in df.columns:
        mean = df[col].mean(skipna=True)
        df[col] = df[col].fillna(mean)

    return df

# Gross (remover linhas Gross = NaN)
def drop_missing_gross(df: pd.DataFrame, col: str = "Gross") -> pd.DataFrame:

    if col in df.columns:
        df = df.dropna(subset=[col])

    return df

# Converte runtime para numérico
def parse_runtime_minutes(df: pd.DataFrame, src_col: str = "Runtime") -> pd.DataFrame:
    if src_col in df.columns:
        mins = (
            df[src_col].astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        mins = pd.to_numeric(mins, errors="coerce").astype("Int64")
        df[src_col] = mins
    return df


# Remoção de duplicatas
def drop_dupes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=None, keep="first", ignore_index=True)

    return df

# Pipeline de Limpeza
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_redundant_columns(df)
    df = drop_dupes(df)
    df = clean_certificate(df)
    df = parse_gross(df)
    df = to_numeric_year(df)
    df = impute_meta_score(df)
    df = drop_missing_gross(df)
    df = parse_runtime_minutes(df)

    return df