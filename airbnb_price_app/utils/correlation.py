"""Covariance and correlation."""
import numpy as np
import pandas as pd
from scipy import stats


def _pair(x, y):
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    return d["x"].astype(float), d["y"].astype(float)


def strength_label(r: float) -> str:
    a = abs(r)
    s = "very weak" if a < 0.2 else "weak" if a < 0.4 else "moderate" if a < 0.6 else "strong" if a < 0.8 else "very strong"
    return f"{s} {'positive' if r > 0 else 'negative' if r < 0 else ''}".strip()


def covariance(x, y) -> float:
    x, y = _pair(x, y)
    return float(np.cov(x, y, ddof=1)[0, 1])


def pearson(x, y) -> dict:
    x, y = _pair(x, y)
    r = stats.pearsonr(x, y)
    return {"r": r.statistic, "p_value": r.pvalue, "n": len(x), "strength": strength_label(r.statistic)}


def spearman(x, y) -> dict:
    x, y = _pair(x, y)
    r = stats.spearmanr(x, y)
    return {"rho": r.statistic, "p_value": r.pvalue, "n": len(x), "strength": strength_label(r.statistic)}


def correlation_matrix(df: pd.DataFrame, cols, method: str = "pearson") -> pd.DataFrame:
    return df[list(cols)].corr(method=method)


def covariance_matrix(df: pd.DataFrame, cols) -> pd.DataFrame:
    return df[list(cols)].cov()


def correlation_with_target(df: pd.DataFrame, target: str, cols) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c == target:
            continue
        x, y = _pair(df[c], df[target])
        if len(x) < 3 or x.nunique() < 2:
            continue
        p, s = pearson(x, y), spearman(x, y)
        rows.append({"variable": c, "n": p["n"], "covariance": covariance(x, y), "pearson_r": p["r"],
                     "pearson_p": p["p_value"], "spearman_rho": s["rho"], "spearman_p": s["p_value"],
                     "strength (pearson)": p["strength"]})
    return pd.DataFrame(rows).sort_values("pearson_r", key=abs, ascending=False).reset_index(drop=True)
