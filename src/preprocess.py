from pathlib import Path
import re
from typing import Optional, List, Iterable, Sequence, Set, Dict

import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

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

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    
    s = s.lower()
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _build_stopwords(extra: Optional[Iterable[str]] = None) -> Set[str]:
    sw = set(STOPWORDS)

    sw.update({
        "film", "movie", "story", "life", "man", "woman", "years", "one",
        "young", "new", "must", "find", "world", "family", "together",
        "based", "true", "team", "set"
    })
    if extra:
        sw.update({e.lower() for e in extra})
    return sw

def generate_wordcloud_from_series(text_series, outfile: Path | str, width: int = 1600, height: int = 900, max_words: int = 300, background_color: str = "white", extra_stopwords: Optional[Iterable[str]] = None) -> Path:
    sw = _build_stopwords(extra_stopwords)

    joined = " ".join(_normalize_text(t) for t in text_series.dropna().astype(str))
    if not joined.strip():
        raise ValueError("Série de texto vazia após normalização.")
    wc = WordCloud(
        width=width,
        height=height,
        background_color=background_color,
        stopwords=sw,
        collocations=True,
        max_words=max_words
    ).generate(joined)

    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(width/100, height/100))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(outfile, dpi=150)
    plt.close()
    return outfile

def generate_wordclouds_by_category(df: pd.DataFrame, text_col: str = "Overview", cat_col: str = "Genre", top_k: int = 5, min_docs: int = 10, extra_stopwords: Optional[Iterable[str]] = None) -> Dict[str, Path]:

    vc = df[cat_col].value_counts()
    cats = [c for c in vc.index[:top_k] if vc[c] >= min_docs]

    outputs: Dict[str, Path] = {}
    for c in cats:
        subset = df.loc[df[cat_col] == c, text_col]
        safe_name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(c))[:40]
        outfile = REPORTS_DIR / f"wordcloud_{cat_col.lower()}_{safe_name}.png"
        try:
            p = generate_wordcloud_from_series(
                subset,
                outfile=outfile,
                extra_stopwords=extra_stopwords
            )
            outputs[str(c)] = p
        except ValueError:
            continue
    return outputs