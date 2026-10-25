"""Hypothesis tests (t-tests, ANOVA, chi-square, non-parametric alternatives)."""
import numpy as np
import pandas as pd
from scipy import stats


def interpret(p: float, alpha: float) -> str:
    if p < alpha:
        return (f"p-value ({p:.4g}) < α ({alpha}) → reject H0: the result is statistically significant at the "
                f"{alpha:.0%} level.")
    return (f"p-value ({p:.4g}) ≥ α ({alpha}) → fail to reject H0: there is not enough evidence of a difference "
            f"(this does not prove H0 is true).")


def _clean(x):
    return pd.Series(x).dropna().astype(float).to_numpy()


def cohens_d(a, b) -> float:
    a, b = _clean(a), _clean(b)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def effect_label(d: float) -> str:
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def normality_check(x, max_n: int = 5000, seed: int = 0) -> dict:
    x = _clean(x)
    if len(x) < 3:
        return {"statistic": np.nan, "p_value": np.nan, "n": len(x)}
    if len(x) > max_n:
        x = np.random.default_rng(seed).choice(x, max_n, replace=False)
    w = stats.shapiro(x)
    return {"statistic": w.statistic, "p_value": w.pvalue, "n": len(x)}


def one_sample_ttest(x, mu0: float, alpha: float = 0.05, alternative: str = "two-sided") -> dict:
    x = _clean(x)
    r = stats.ttest_1samp(x, mu0, alternative=alternative)
    return {"test": "One-sample t-test", "statistic": r.statistic, "p_value": r.pvalue, "df": len(x) - 1,
            "sample_mean": x.mean(), "mu0": mu0, "n": len(x), "reject": bool(r.pvalue < alpha),
            "conclusion": interpret(r.pvalue, alpha)}


def two_sample_ttest(a, b, alpha: float = 0.05, equal_var: bool = False, alternative: str = "two-sided") -> dict:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group needs at least 2 observations.")
    r = stats.ttest_ind(a, b, equal_var=equal_var, alternative=alternative)
    lev = stats.levene(a, b)
    d = cohens_d(a, b)
    out = {"test": "Student t-test (equal variances)" if equal_var else "Welch t-test (unequal variances)",
           "statistic": r.statistic, "p_value": r.pvalue, "df": getattr(r, "df", np.nan),
           "mean_a": a.mean(), "mean_b": b.mean(), "diff": a.mean() - b.mean(), "n_a": len(a), "n_b": len(b),
           "cohens_d": d, "effect_size": effect_label(d), "levene_p": lev.pvalue,
           "equal_variance_ok": bool(lev.pvalue >= alpha), "reject": bool(r.pvalue < alpha),
           "conclusion": interpret(r.pvalue, alpha)}
    try:
        ci = r.confidence_interval(1 - alpha)
        out["ci_diff"] = (ci.low, ci.high)
    except Exception:
        out["ci_diff"] = None
    return out


def mann_whitney(a, b, alpha: float = 0.05) -> dict:
    r = stats.mannwhitneyu(_clean(a), _clean(b), alternative="two-sided")
    return {"test": "Mann-Whitney U (non-parametric)", "statistic": r.statistic, "p_value": r.pvalue,
            "reject": bool(r.pvalue < alpha), "conclusion": interpret(r.pvalue, alpha)}


def paired_ttest(x, y, alpha: float = 0.05, alternative: str = "two-sided") -> dict:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 2:
        raise ValueError("Need at least 2 complete pairs.")
    r = stats.ttest_rel(d["x"], d["y"], alternative=alternative)
    diff = d["x"] - d["y"]
    dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
    return {"test": "Paired t-test", "statistic": r.statistic, "p_value": r.pvalue, "df": len(d) - 1,
            "mean_x": d["x"].mean(), "mean_y": d["y"].mean(), "mean_diff": diff.mean(), "n_pairs": len(d),
            "cohens_dz": dz, "reject": bool(r.pvalue < alpha), "conclusion": interpret(r.pvalue, alpha)}


def one_way_anova(groups: dict, alpha: float = 0.05, post_hoc: bool = True) -> dict:
    groups = {k: _clean(v) for k, v in groups.items() if len(_clean(v)) >= 2}
    if len(groups) < 2:
        raise ValueError("ANOVA needs at least two groups with 2+ observations each.")
    arrs = list(groups.values())
    f = stats.f_oneway(*arrs)
    allv = np.concatenate(arrs)
    ss_b = sum(len(g) * (g.mean() - allv.mean()) ** 2 for g in arrs)
    ss_t = ((allv - allv.mean()) ** 2).sum()
    out = {"test": "One-way ANOVA", "statistic": f.statistic, "p_value": f.pvalue, "k_groups": len(arrs),
           "df_between": len(arrs) - 1, "df_within": len(allv) - len(arrs), "eta_squared": ss_b / ss_t,
           "levene_p": stats.levene(*arrs).pvalue, "kruskal_p": stats.kruskal(*arrs).pvalue,
           "reject": bool(f.pvalue < alpha), "conclusion": interpret(f.pvalue, alpha),
           "group_means": pd.Series({k: v.mean() for k, v in groups.items()}).round(2)}
    if post_hoc and out["reject"] and len(arrs) > 2:
        try:
            res = stats.tukey_hsd(*arrs)
            names, rows = list(groups), []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    rows.append({"group_1": names[i], "group_2": names[j], "mean_diff": arrs[i].mean() - arrs[j].mean(),
                                 "p_adj": res.pvalue[i, j], "significant": bool(res.pvalue[i, j] < alpha)})
            out["tukey"] = pd.DataFrame(rows)
        except Exception:
            out["tukey"] = None
    return out


def chi_square_independence(df: pd.DataFrame, col1: str, col2: str, alpha: float = 0.05) -> dict:
    ct = pd.crosstab(df[col1], df[col2])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Both variables need at least 2 categories.")
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    exp_df = pd.DataFrame(exp, index=ct.index, columns=ct.columns)
    return {"test": "Chi-square test of independence", "statistic": chi2, "p_value": p, "dof": dof,
            "cramers_v": v, "observed": ct, "expected": exp_df.round(2),
            "low_expected_cells": int((exp < 5).sum()), "reject": bool(p < alpha), "conclusion": interpret(p, alpha)}
