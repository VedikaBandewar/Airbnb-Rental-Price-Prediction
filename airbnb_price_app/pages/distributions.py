import numpy as np
import plotly.graph_objects as go
import scipy.stats as sps
import streamlit as st
from utils import ui
from utils.distributions import (DISTRIBUTIONS, fit_price_distributions, sample_distribution, theoretical_curve,
                                 theoretical_moments)

ui.page_header("Probability Distributions", "Populations, samples and random variables — and how prices compare to theory.")
t1, t2, t3 = st.tabs(["Concepts", "Distribution lab", "Airbnb prices vs theory"])

with t1:
    st.markdown("""
- **Population** – all the listings we care about; **sample** – the subset we actually observe.
- **Random variable** – a numeric outcome of a random process (e.g. a listing's price; the number of reviews).
- **Discrete** (Binomial, Poisson) use a *probability mass function*; **continuous** (Normal, Uniform) use a *density*.
- **Normal(μ, σ)** – symmetric bell curve. **Uniform(a, b)** – all values equally likely.
- **Binomial(n, p)** – number of successes in *n* trials (e.g. how many of 20 listings are 'Private room').
- **Poisson(λ)** – count of events in a fixed interval (e.g. reviews per month).
""")

with t2:
    name = st.selectbox("Distribution", DISTRIBUTIONS)
    c1, c2, c3 = st.columns(3)
    size = c1.slider("Sample size", 10, 20000, 1000, 10)
    seed = c2.number_input("Random seed", 0, 99999, 42)
    p = {}
    if name == "Normal":
        p["loc"] = c3.slider("Mean μ", -50.0, 200.0, 100.0); p["scale"] = c3.slider("Std dev σ", 0.5, 60.0, 20.0)
    elif name == "Uniform":
        lo = c3.slider("Low a", 0.0, 200.0, 20.0); hi = c3.slider("High b", 1.0, 500.0, 200.0)
        if hi <= lo:
            st.error("High must be greater than low."); st.stop()
        p["low"], p["high"] = lo, hi
    elif name == "Binomial":
        p["n"] = c3.slider("Trials n", 1, 200, 20); p["p"] = c3.slider("Success probability p", 0.01, 0.99, 0.3)
    else:
        p["lam"] = c3.slider("Rate λ", 0.1, 50.0, 4.0)
    x = sample_distribution(name, size, int(seed), **p)
    gx, gy, kind = theoretical_curve(name, **p)
    fig = go.Figure()
    fig.add_histogram(x=x, histnorm="probability density", nbinsx=50 if kind == "pdf" else None, name="Random sample",
                      opacity=0.6, xbins=None if kind == "pdf" else dict(size=1))
    if kind == "pdf":
        fig.add_scatter(x=gx, y=gy, mode="lines", name="Theoretical PDF", line=dict(width=3))
    else:
        fig.add_scatter(x=gx, y=gy, mode="markers+lines", name="Theoretical PMF")
    fig.update_layout(bargap=0.05, xaxis_title="value", yaxis_title="density / probability")
    ui.show_fig(fig)
    m = theoretical_moments(name, **p)
    ui.kpi_row([("Sample mean", f"{x.mean():.3f}", f"theory {m['mean']:.3f}"),
                ("Sample variance", f"{x.var(ddof=1):.3f}", f"theory {m['variance']:.3f}"),
                ("Sample size", f"{size:,}", "bigger → closer to theory")])

with t3:
    df, _ = ui.get_clean()
    prices = df["price"]
    fit = fit_price_distributions(prices)
    ui.kpi_row([("Skewness", f"{fit['skewness']:.2f}", "0 = symmetric"),
                ("Normal fit KS p", f"{fit['normal']['p_value']:.2g}", "small → poor fit"),
                ("Log-normal fit KS p", f"{fit['lognormal']['p_value']:.2g}", "larger → better fit"),
                ("KS statistic (N vs LN)", f"{fit['normal']['ks_stat']:.3f} vs {fit['lognormal']['ks_stat']:.3f}", "lower is better")])
    cap = prices.quantile(0.99)
    xs = np.linspace(0.5, cap, 400)
    fig = go.Figure()
    fig.add_histogram(x=prices[prices <= cap], histnorm="probability density", nbinsx=60, name="Observed prices", opacity=0.55)
    fig.add_scatter(x=xs, y=sps.norm.pdf(xs, fit["normal"]["mu"], fit["normal"]["sigma"]), name="Normal fit", mode="lines")
    ln = fit["lognormal"]
    fig.add_scatter(x=xs, y=sps.lognorm.pdf(xs, ln["shape"], ln["loc"], ln["scale"]), name="Log-normal fit", mode="lines")
    fig.update_layout(xaxis_title="Nightly price ($)", yaxis_title="density")
    ui.show_fig(fig)
    q = np.linspace(0.01, 0.99, 99)
    qq = go.Figure()
    qq.add_scatter(x=sps.norm.ppf(q), y=np.log(prices.quantile(q).to_numpy()), mode="markers", name="log(price) quantiles")
    zs = sps.norm.ppf([0.01, 0.99]); lp = np.log(prices)
    qq.add_scatter(x=zs, y=lp.mean() + lp.std() * zs, mode="lines", name="Normal reference")
    qq.update_layout(title="Q–Q plot of log(price) vs Normal", xaxis_title="theoretical quantile", yaxis_title="log(price) quantile")
    ui.show_fig(qq)
    st.markdown("Raw prices are **right-skewed** and badly described by a Normal curve, while **log(price)** is close to "
                "normal (log-normal prices). This is why the regression module can model `log(price)`.")
