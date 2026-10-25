"""Descriptive statistics."""
import numpy as np
import pandas as pd


def inspect_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Equivalent of df.info() as a table."""
    return pd.DataFrame({"dtype": df.dtypes.astype(str), "non_null": df.notna().sum(),
                         "null": df.isna().sum(), "unique": df.nunique()})


def summary_stats(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    mode = s.mode()
    q1, q3 = s.quantile([0.25, 0.75])
    return {"count": int(len(s)), "mean": s.mean(), "median": s.median(),
            "mode": mode.iloc[0] if len(mode) else np.nan, "variance": s.var(), "std_dev": s.std(),
            "min": s.min(), "Q1 (25%)": q1, "Q3 (75%)": q3, "max": s.max(), "IQR": q3 - q1,
            "range": s.max() - s.min(), "skewness": s.skew(), "kurtosis": s.kurt(),
            "coef_of_variation": s.std() / s.mean() if s.mean() else np.nan}


def grouped_summary(df: pd.DataFrame, by: str, col: str = "price") -> pd.DataFrame:
    g = df.groupby(by)[col].agg(["count", "mean", "median", "std", "min", "max"]).round(2)
    return g.sort_values("mean", ascending=False)


def numeric_describe(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").describe().T.round(3)
