"""Dataset loading, column standardisation and validation (pure pandas, no Streamlit)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "airbnb_listings.csv"
SAMPLE_PATH = ROOT / "data" / "sample_listings.csv"

# Inside Airbnb "detailed" listings use different names; map them to ours.
ALIASES = {"neighbourhood_cleansed": "neighbourhood", "neighborhood": "neighbourhood",
           "neighborhood_cleansed": "neighbourhood", "accommodate": "accommodates",
           "bedroom": "bedrooms", "bed": "beds", "bathroom": "bathrooms"}
MONEY_COLS = ["price", "weekday_price", "weekend_price"]


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _money_to_float(s: pd.Series) -> pd.Series:
    if is_numeric(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace("", np.nan)
    return pd.to_numeric(cleaned, errors="coerce")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case names, apply aliases, convert '$1,234.00' prices to floats, parse bathrooms_text."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for old, new in ALIASES.items():
        if old in df.columns:
            if new in df.columns and old != new:
                df = df.drop(columns=[new])
            df = df.rename(columns={old: new})
    if "bathrooms" not in df.columns and "bathrooms_text" in df.columns:
        txt = df["bathrooms_text"].astype(str).str.lower()
        num = pd.to_numeric(txt.str.extract(r"(\d+\.?\d*)")[0], errors="coerce")
        df["bathrooms"] = num.where(~txt.str.contains("half"), 0.5)
    for c in MONEY_COLS:
        if c in df.columns:
            df[c] = _money_to_float(df[c])
    return df


def validate_dataset(df: pd.DataFrame, min_rows: int = 50):
    """Return (ok, messages). A usable dataset needs price, >=min_rows rows and >=1 other feature."""
    msgs = []
    if "price" not in df.columns:
        return False, ["Missing required column 'price'. Columns found: " + ", ".join(map(str, df.columns[:15]))]
    valid = df["price"].notna() & (df["price"] > 0)
    if valid.sum() < min_rows:
        msgs.append(f"Only {int(valid.sum())} rows have a positive price; at least {min_rows} are required.")
    feats = [c for c in df.columns if c not in ("price", "id", "name", "url") and df[c].nunique(dropna=True) > 1]
    if not feats:
        msgs.append("No usable feature columns (e.g. room_type, neighbourhood, accommodates) were found.")
    return (not msgs), msgs


def load_csv(path_or_buffer) -> pd.DataFrame:
    return standardise_columns(pd.read_csv(path_or_buffer))


def load_default() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from scripts.generate_dataset import generate  # fallback: rebuild on the fly
        generate().to_csv(DATA_PATH, index=False)
    return load_csv(DATA_PATH)


def categorical_columns(df: pd.DataFrame, max_unique: int = 60):
    """Grouping-friendly columns: non-numeric, not an identifier, and every level has a sensible number of rows."""
    out = []
    for c in df.columns:
        if is_numeric(df[c]) or c in ("id", "name", "url", "listing_id", "bathrooms_text"):
            continue
        n = df[c].nunique()
        if 1 < n <= max_unique and len(df) / n >= 3:       # avg >= 3 rows per level (rules out ID-like columns)
            out.append(c)
    return out


def numeric_columns(df: pd.DataFrame):
    return [c for c in df.columns if is_numeric(df[c]) and c != "id"]
