"""Probability distributions, sampling, Central Limit Theorem and estimation."""
import numpy as np
import pandas as pd
from scipy import stats

DISTRIBUTIONS = ["Normal", "Uniform", "Binomial", "Poisson"]


def sample_distribution(name: str, size: int, seed: int | None = None, **p) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "Normal":
        return rng.normal(p.get("loc", 0.0), p.get("scale", 1.0), size)
    if name == "Uniform":
        return rng.uniform(p.get("low", 0.0), p.get("high", 1.0), size)
    if name == "Binomial":
        return rng.binomial(int(p.get("n", 10)), p.get("p", 0.5), size)
    if name == "Poisson":
        return rng.poisson(p.get("lam", 3.0), size)
    raise ValueError(f"Unknown distribution {name}")


def theoretical_curve(name: str, **p):
    """Return (x, y, kind) where kind is 'pdf' or 'pmf'."""
    if name == "Normal":
        loc, sc = p.get("loc", 0.0), p.get("scale", 1.0)
        x = np.linspace(loc - 4 * sc, loc + 4 * sc, 300)
        return x, stats.norm.pdf(x, loc, sc), "pdf"
    if name == "Uniform":
        lo, hi = p.get("low", 0.0), p.get("high", 1.0)
        x = np.linspace(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo), 300)
        return x, stats.uniform.pdf(x, lo, hi - lo), "pdf"
    if name == "Binomial":
        n, pr = int(p.get("n", 10)), p.get("p", 0.5)
        x = np.arange(0, n + 1)
        return x, stats.binom.pmf(x, n, pr), "pmf"
    if name == "Poisson":
        lam = p.get("lam", 3.0)
        x = np.arange(0, int(lam + 5 * np.sqrt(lam) + 5))
        return x, stats.poisson.pmf(x, lam), "pmf"
    raise ValueError(name)


def theoretical_moments(name: str, **p) -> dict:
    if name == "Normal":
        return {"mean": p.get("loc", 0.0), "variance": p.get("scale", 1.0) ** 2}
    if name == "Uniform":
        lo, hi = p.get("low", 0.0), p.get("high", 1.0)
        return {"mean": (lo + hi) / 2, "variance": (hi - lo) ** 2 / 12}
    if name == "Binomial":
        n, pr = int(p.get("n", 10)), p.get("p", 0.5)
        return {"mean": n * pr, "variance": n * pr * (1 - pr)}
    if name == "Poisson":
        return {"mean": p.get("lam", 3.0), "variance": p.get("lam", 3.0)}
    raise ValueError(name)


def fit_price_distributions(prices: pd.Series) -> dict:
    """Fit Normal and Log-normal to prices and run Kolmogorov-Smirnov goodness-of-fit tests."""
    x = pd.Series(prices).dropna()
    x = x[x > 0].to_numpy()
    mu, sd = x.mean(), x.std(ddof=1)
    shape, loc, scale = stats.lognorm.fit(x, floc=0)
    ks_n = stats.kstest(
    x,
    lambda value: stats.norm.cdf(value, loc=mu, scale=sd)
)
    ks_l = stats.kstest(x, "lognorm", args=(shape, loc, scale))
    return {"normal": {"mu": mu, "sigma": sd, "ks_stat": ks_n.statistic, "p_value": ks_n.pvalue},
            "lognormal": {"shape": shape, "loc": loc, "scale": scale, "ks_stat": ks_l.statistic, "p_value": ks_l.pvalue},
            "skewness": stats.skew(x), "n": len(x)}


# ---------------------------- CLT & estimation ----------------------------
def sampling_means(population, n: int, n_samples: int, seed: int | None = None) -> np.ndarray:
    """Draw `n_samples` random samples of size n (with replacement) and return their means."""
    pop = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.choice(pop, size=(n_samples, n), replace=True).mean(axis=1)


def clt_summary(population, means: np.ndarray, n: int) -> dict:
    pop = np.asarray(population, dtype=float)
    sigma = pop.std(ddof=0)
    return {"population_mean": pop.mean(), "population_std": sigma, "mean_of_sample_means": means.mean(),
            "empirical_SE": means.std(ddof=1), "theoretical_SE": sigma / np.sqrt(n),
            "skew_population": stats.skew(pop), "skew_sample_means": stats.skew(means)}


def standard_error(sample) -> float:
    s = np.asarray(sample, dtype=float)
    return s.std(ddof=1) / np.sqrt(len(s))


def confidence_interval(sample, confidence: float = 0.95) -> dict:
    s = np.asarray(sample, dtype=float)
    n = len(s)
    if n < 2:
        raise ValueError("Need at least 2 observations for a confidence interval.")
    m, se = s.mean(), standard_error(s)
    t = stats.t.ppf((1 + confidence) / 2, n - 1)
    return {"mean": m, "se": se, "t_crit": t, "margin": t * se, "lower": m - t * se, "upper": m + t * se, "n": n}


def ci_coverage(population, n: int, trials: int, confidence: float, seed: int | None = None) -> dict:
    """Fraction of CIs that contain the true population mean."""
    pop = np.asarray(population, dtype=float)
    rng = np.random.default_rng(seed)
    mu = pop.mean()
    hits, rows = 0, []
    for i in range(trials):
        ci = confidence_interval(rng.choice(pop, n, replace=True), confidence)
        ok = ci["lower"] <= mu <= ci["upper"]
        hits += ok
        if i < 50:
            rows.append({"trial": i + 1, "lower": ci["lower"], "upper": ci["upper"], "mean": ci["mean"], "covers": ok})
    return {"coverage": hits / trials, "true_mean": mu, "examples": pd.DataFrame(rows)}


def se_vs_n(population, sizes, reps: int = 500, seed: int | None = None) -> pd.DataFrame:
    pop = np.asarray(population, dtype=float)
    sigma = pop.std(ddof=0)
    rows = []
    for k, n in enumerate(sizes):
        m = sampling_means(pop, int(n), reps, None if seed is None else seed + k)
        rows.append({"sample_size": int(n), "empirical_SE": m.std(ddof=1), "theoretical_SE": sigma / np.sqrt(n),
                     "skewness_of_means": stats.skew(m)})
    return pd.DataFrame(rows)
