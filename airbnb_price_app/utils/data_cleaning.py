"""Cleaning and preprocessing helpers."""
import pandas as pd
from .data_io import is_numeric


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    rep = pd.DataFrame({"missing": df.isna().sum(), "missing_%": (df.isna().mean() * 100).round(2)})
    return rep[rep["missing"] > 0].sort_values("missing", ascending=False)


def iqr_bounds(s: pd.Series, k: float = 1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def handle_outliers(df: pd.DataFrame, col: str = "price", method: str = "none", k: float = 1.5) -> pd.DataFrame:
    """method: 'none' | 'remove' (drop rows outside IQR fences) | 'cap' (winsorise to the fences)."""
    if method == "none" or col not in df.columns:
        return df.copy()
    lo, hi = iqr_bounds(df[col].dropna(), k)
    if method == "remove":
        return df[(df[col].isna()) | df[col].between(lo, hi)].copy()
    if method == "cap":
        out = df.copy()
        out[col] = out[col].clip(lo, hi)
        return out
    raise ValueError(f"Unknown outlier method: {method}")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is safe BEFORE a train/test split (no statistics learned from the data):
    remove exact duplicates and rows with a missing / non-positive price."""
    out = df.drop_duplicates().copy()
    out = out[out["price"].notna() & (out["price"] > 0)]
    return out.reset_index(drop=True)


def clean_listings(df: pd.DataFrame, impute: bool = True, outliers: str = "none", k: float = 1.5):
    """Full cleaning for EDA. Returns (clean_df, report dict).
    NOTE: imputation here uses whole-data statistics, which is fine for descriptive work but is NOT used
    for model training (training imputes inside the sklearn Pipeline after the split to avoid leakage)."""
    rep = {"rows_before": len(df)}
    rep["duplicates_removed"] = int(df.duplicated().sum())
    out = df.drop_duplicates().copy()
    bad = out["price"].isna() | (out["price"] <= 0)
    rep["invalid_price_removed"] = int(bad.sum())
    out = out[~bad]
    rep["missing_before_imputation"] = int(out.isna().sum().sum())
    if impute:
        for c in out.columns:
            if c in ("id", "name", "price"):
                continue
            if is_numeric(out[c]):
                out[c] = out[c].fillna(out[c].median())
            elif out[c].isna().any():
                m = out[c].mode()
                out[c] = out[c].fillna(m.iloc[0] if len(m) else "Unknown")
    n0 = len(out)
    out = handle_outliers(out, "price", outliers, k)
    rep["outlier_rows_removed"] = n0 - len(out) if outliers == "remove" else 0
    rep["outlier_method"] = outliers
    rep["rows_after"] = len(out)
    return out.reset_index(drop=True), rep
