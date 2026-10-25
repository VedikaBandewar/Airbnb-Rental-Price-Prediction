import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from utils import ui
from utils.correlation import correlation_matrix, correlation_with_target, covariance, covariance_matrix, pearson, spearman
from utils.data_io import categorical_columns, numeric_columns
from utils.hypothesis_testing import chi_square_independence, one_way_anova, paired_ttest, two_sample_ttest

ui.page_header("Correlation and Statistical Tests", "t-tests, paired t-test, ANOVA, chi-square, covariance and correlation.")
df, _ = ui.get_clean()
num, cat = numeric_columns(df), categorical_columns(df)
ui.note("Choosing a test: <b>numeric vs 2 groups</b> → t-test · <b>same units measured twice</b> → paired t-test · "
        "<b>numeric vs 3+ groups</b> → ANOVA · <b>two categorical variables</b> → chi-square · "
        "<b>two numeric variables</b> → correlation. <b>Correlation ≠ causation.</b>")
tabs = st.tabs(["Independent t-test", "Paired t-test", "One-way ANOVA", "Chi-square", "Covariance & correlation", "Heatmap"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    var = c1.selectbox("Numeric variable", ["price"] + [c for c in num if c != "price"])
    by = c2.selectbox("Split by", cat)
    lv = sorted(df[by].dropna().unique().tolist())
    ga, gb = c3.selectbox("Group A", lv, 0), c3.selectbox("Group B", lv, min(1, len(lv) - 1))
    alpha = st.select_slider("α", [0.01, 0.05, 0.10], 0.05, key="a_t")
    if ga == gb:
        st.error("Choose two different groups.")
    else:
        try:
            r = two_sample_ttest(df.loc[df[by] == ga, var], df.loc[df[by] == gb, var], alpha)
            ui.kpi_row([("t statistic", f"{r['statistic']:.3f}", f"df ≈ {r['df']:.0f}"), ("p-value", f"{r['p_value']:.4g}", r["test"].split(" (")[0]),
                        ("Mean A / B", f"{r['mean_a']:.1f} / {r['mean_b']:.1f}", f"n = {r['n_a']} / {r['n_b']}"),
                        ("Cohen's d", f"{r['cohens_d']:.2f}", r["effect_size"])])
            (st.success if r["reject"] else st.info)(r["conclusion"])
        except ValueError as e:
            st.error(str(e))

with tabs[1]:
    if {"weekday_price", "weekend_price"} <= set(df.columns):
        st.markdown("**Paired design:** each listing has a weekday and a weekend price → one pair per listing.  \n"
                    "**H0:** mean(weekend − weekday) = 0 &nbsp; **H1:** ≠ 0")
        r = paired_ttest(df["weekend_price"], df["weekday_price"])
        ui.kpi_row([("t statistic", f"{r['statistic']:.2f}", f"df = {r['df']}"), ("p-value", f"{r['p_value']:.3g}", ""),
                    ("Mean difference", f"${r['mean_diff']:.2f}", "weekend − weekday"), ("Pairs", f"{r['n_pairs']:,}", "")])
        (st.success if r["reject"] else st.info)(r["conclusion"])
        diff = df["weekend_price"] - df["weekday_price"]
        ui.show_fig(px.histogram(diff, nbins=50, labels={"value": "weekend − weekday price ($)"}).update_layout(showlegend=False))
    else:
        st.warning("This dataset has no paired columns (weekday_price / weekend_price), so a paired t-test is not applicable. "
                   "Never pair unrelated observations.")

with tabs[2]:
    by = st.selectbox("Group by", cat, key="an_by")
    alpha = st.select_slider("α", [0.01, 0.05, 0.10], 0.05, key="a_an")
    log = st.checkbox("Use log(price)", True, key="an_log")
    groups = {str(k): (np.log(v["price"]) if log else v["price"]) for k, v in df.groupby(by)}
    try:
        r = one_way_anova(groups, alpha)
        ui.kpi_row([("F statistic", f"{r['statistic']:.2f}", f"df = ({r['df_between']}, {r['df_within']})"), ("p-value", f"{r['p_value']:.3g}", ""),
                    ("η²", f"{r['eta_squared']:.3f}", "share of variance explained"), ("Levene p", f"{r['levene_p']:.3g}", "equal variances?")])
        (st.success if r["reject"] else st.info)(r["conclusion"])
        if r.get("tukey") is not None:
            st.markdown("**Tukey HSD post-hoc** (which pairs differ):")
            ui.show_df(r["tukey"].round(4), hide_index=True)
    except ValueError as e:
        st.error(str(e))

with tabs[3]:
    d2 = df.copy()
    d2["price_tier"] = pd.qcut(d2["price"], 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    options = cat + ["price_tier"]
    c1, c2 = st.columns(2)
    a = c1.selectbox("Variable 1", options, 0)
    b = c2.selectbox("Variable 2", options, min(1, len(options) - 1))
    if a == b:
        st.error("Choose two different variables.")
    else:
        r = chi_square_independence(d2.assign(**{a: d2[a].astype(str), b: d2[b].astype(str)}), a, b)
        ui.kpi_row([("χ²", f"{r['statistic']:.1f}", f"df = {r['dof']}"), ("p-value", f"{r['p_value']:.3g}", ""),
                    ("Cramér's V", f"{r['cramers_v']:.3f}", "strength of association"), ("Cells expected < 5", f"{r['low_expected_cells']}", "should be 0")])
        (st.success if r["reject"] else st.info)(r["conclusion"])
        if r["low_expected_cells"]:
            st.warning("Some expected counts are below 5 — the chi-square approximation may be unreliable.")
        ui.show_df(r["observed"])
        ui.show_fig(px.imshow(r["observed"], text_auto=True, aspect="auto", title="Observed counts"))

with tabs[4]:
    cands = [c for c in ["number_of_reviews", "minimum_nights", "availability_365", "accommodates", "bedrooms", "beds", "bathrooms"] if c in df.columns]
    st.markdown("**Price versus other variables**")
    tab = correlation_with_target(df, "price", cands)
    ui.show_df(tab.round(4), hide_index=True)
    x = st.selectbox("Inspect pair: price vs …", cands)
    p, s = pearson(df[x], df["price"]), spearman(df[x], df["price"])
    ui.kpi_row([("Covariance", f"{covariance(df[x], df['price']):,.2f}", "units depend on scale"), ("Pearson r", f"{p['r']:.3f}", f"p = {p['p_value']:.3g}"),
                ("Spearman ρ", f"{s['rho']:.3f}", f"p = {s['p_value']:.3g}"), ("Strength", p["strength"], "")])
    sm = df.sample(min(len(df), 2500), random_state=0)
    fig = px.scatter(sm, x=x, y="price", opacity=0.4, log_y=True)
    k, b0 = np.polyfit(df[x], np.log(df["price"]), 1)
    xs = np.linspace(df[x].min(), df[x].max(), 50)
    fig.add_scatter(x=xs, y=np.exp(b0 + k * xs), mode="lines", name="log-linear trend", line=dict(width=3))
    ui.show_fig(fig)
    st.caption("Pearson measures *linear* association and is sensitive to outliers; Spearman measures *monotonic* association on ranks. "
               "A strong correlation does not mean one variable causes the other (e.g. bigger places have more bedrooms *and* cost more).")
    st.markdown("**Price versus categorical variables** is covered by the t-test / ANOVA tabs (room type, neighbourhood).")

with tabs[5]:
    method = st.radio("Method", ["pearson", "spearman"], horizontal=True)
    cols = st.multiselect("Variables", num, default=[c for c in num if c not in ("weekday_price", "weekend_price")])
    if len(cols) >= 2:
        cm = correlation_matrix(df, cols, method)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
        ui.show_mpl(fig)
        with st.expander("Covariance matrix"):
            ui.show_df(covariance_matrix(df, cols).round(2))
    else:
        st.info("Select at least two variables.")
