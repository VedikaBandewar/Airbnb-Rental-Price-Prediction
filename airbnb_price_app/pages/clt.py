import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as sps
import streamlit as st
from utils import ui
from utils.distributions import ci_coverage, clt_summary, confidence_interval, sampling_means, se_vs_n

ui.page_header("Central Limit Theorem and Estimation", "Sampling distributions, standard error and confidence intervals from Airbnb prices.")
df, _ = ui.get_clean()
pop = df["price"].to_numpy(float)
ui.note("<b>CLT:</b> for large enough <i>n</i>, the sample mean x̄ is approximately Normal with mean μ and standard error "
        "σ/√n — <i>even if the population itself is skewed</i> (like prices).")
t1, t2, t3 = st.tabs(["Sampling experiment", "Effect of sample size", "Confidence intervals"])

with t1:
    c1, c2, c3 = st.columns(3)
    n = c1.slider("Sample size n", 2, 500, 30)
    reps = c2.slider("Number of samples", 50, 5000, 1000, 50)
    seed = c3.number_input("Seed", 0, 99999, 7)
    means = sampling_means(pop, n, reps, int(seed))
    s = clt_summary(pop, means, n)
    ui.kpi_row([("Population mean μ", ui.money(s["population_mean"]), f"σ = {s['population_std']:.1f}"),
                ("Mean of sample means", ui.money(s["mean_of_sample_means"]), "≈ μ (unbiased)"),
                ("Empirical SE", f"{s['empirical_SE']:.2f}", f"theory σ/√n = {s['theoretical_SE']:.2f}"),
                ("Skewness", f"{s['skew_sample_means']:.2f}", f"population {s['skew_population']:.2f}")])
    a, b = st.columns(2)
    with a:
        cap = np.quantile(pop, 0.99)
        ui.show_fig(px.histogram(x=pop[pop <= cap], nbins=50, title="Population of prices (skewed)", labels={"x": "price"}))
    with b:
        xs = np.linspace(means.min(), means.max(), 200)
        fig = go.Figure()
        fig.add_histogram(x=means, histnorm="probability density", nbinsx=50, name="Sample means", opacity=0.6)
        fig.add_scatter(x=xs, y=sps.norm.pdf(xs, s["population_mean"], s["theoretical_SE"]), mode="lines", name="Normal(μ, σ/√n)")
        fig.add_vline(x=s["population_mean"], line_dash="dash")
        fig.update_layout(title=f"Distribution of {reps:,} sample means (n = {n})", xaxis_title="sample mean")
        ui.show_fig(fig)

with t2:
    sizes = [2, 5, 10, 20, 30, 50, 100, 200, 500]
    tab = se_vs_n(pop, sizes, reps=500, seed=11)
    fig = go.Figure()
    fig.add_scatter(x=tab["sample_size"], y=tab["empirical_SE"], mode="markers+lines", name="Empirical SE")
    fig.add_scatter(x=tab["sample_size"], y=tab["theoretical_SE"], mode="lines", name="σ/√n", line=dict(dash="dash"))
    fig.update_layout(xaxis_type="log", xaxis_title="sample size n (log scale)", yaxis_title="standard error ($)")
    ui.show_fig(fig)
    ui.show_df(tab.round(3), hide_index=True)
    st.caption("Quadrupling the sample size halves the standard error. Skewness of the sample means shrinks toward 0 as n grows.")

with t3:
    c1, c2, c3 = st.columns(3)
    n2 = c1.slider("Sample size", 5, 1000, 50, key="ci_n")
    conf = c2.select_slider("Confidence level", [0.80, 0.90, 0.95, 0.99], 0.95)
    seed2 = c3.number_input("Seed", 0, 99999, 3, key="ci_seed")
    sample = np.random.default_rng(int(seed2)).choice(pop, n2, replace=False if n2 <= len(pop) else True)
    ci = confidence_interval(sample, conf)
    ui.kpi_row([("Sample mean", ui.money(ci["mean"]), f"population μ = {ui.money(pop.mean())}"),
                ("Standard error", f"{ci['se']:.2f}", "s/√n"), ("Margin of error", f"±{ci['margin']:.2f}", f"t* = {ci['t_crit']:.3f}"),
                (f"{conf:.0%} CI", f"{ci['lower']:.1f} – {ci['upper']:.1f}", "contains μ ✅" if ci["lower"] <= pop.mean() <= ci["upper"] else "misses μ ❌")])
    st.markdown("**Coverage simulation** — repeat the experiment many times:")
    trials = st.slider("Number of repeated samples", 100, 3000, 500, 100)
    cov = ci_coverage(pop, n2, trials, conf, int(seed2))
    st.metric("Share of intervals containing the true mean", f"{cov['coverage']:.1%}", f"target {conf:.0%}")
    ex = cov["examples"]
    fig = go.Figure()
    for _, r in ex.iterrows():
        fig.add_scatter(x=[r["lower"], r["upper"]], y=[r["trial"]] * 2, mode="lines",
                        line=dict(color="#2a9d8f" if r["covers"] else "#e63946"), showlegend=False)
    fig.add_vline(x=cov["true_mean"], line_dash="dash")
    fig.update_layout(title="First 50 intervals (red = missed the true mean)", yaxis_title="trial", xaxis_title="price ($)")
    ui.show_fig(fig)
