import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as sps
import streamlit as st
from utils import ui
from utils.data_io import categorical_columns
from utils.hypothesis_testing import mann_whitney, normality_check, one_sample_ttest, one_way_anova, two_sample_ttest

ui.page_header("Hypothesis Testing", "Null vs alternative, significance level, test statistic, p-value and conclusion.")
df, _ = ui.get_clean()
cat = categorical_columns(df)
with st.expander("📘 Recap: the logic of a hypothesis test", expanded=False):
    st.markdown("""
1. State **H0** (no effect / no difference) and **H1** (what we suspect).  2. Choose significance level **α** (usually 0.05).
3. Compute a **test statistic** from the data.  4. The **p-value** is the probability of a result at least this extreme *if H0 were true*.
5. If p < α reject H0, otherwise fail to reject. A small p-value is not proof; statistical significance ≠ practical importance (check the effect size).""")
alpha = st.select_slider("Significance level α", [0.01, 0.05, 0.10], 0.05)
t1, t2, t3 = st.tabs(["Two-group comparison", "One-sample test", "Several categories (ANOVA)"])


def t_plot(stat, dfree, alpha, alt):
    x = np.linspace(-5, 5, 400)
    y = sps.t.pdf(x, dfree)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", name="t distribution under H0"))
    if alt == "two-sided":
        crit = sps.t.ppf(1 - alpha / 2, dfree)
        regions = [(x <= -crit), (x >= crit)]
    elif alt == "greater":
        regions = [x >= sps.t.ppf(1 - alpha, dfree)]
    else:
        regions = [x <= sps.t.ppf(alpha, dfree)]
    for m in regions:
        fig.add_scatter(x=x[m], y=y[m], fill="tozeroy", mode="none", fillcolor="rgba(226,80,63,.45)", name="Rejection region")
    fig.add_vline(x=float(np.clip(stat, -5, 5)), line_dash="dash", annotation_text=f"observed t = {stat:.2f}")
    fig.update_layout(showlegend=False, xaxis_title="t", yaxis_title="density")
    return fig


def show_result(r, alpha):
    ui.kpi_row([("Test", r["test"].split(" (")[0], ""), ("Statistic", f"{r['statistic']:.3f}", f"df ≈ {r.get('df', float('nan')):.1f}"),
                ("p-value", f"{r['p_value']:.4g}", f"α = {alpha}"), ("Decision", "Reject H0" if r["reject"] else "Fail to reject", "")])
    (st.success if r["reject"] else st.info)(r["conclusion"])


with t1:
    if not cat:
        st.warning("No categorical column available for grouping.")
    else:
        by = st.selectbox("Grouping variable", cat)
        levels = sorted(df[by].dropna().unique().tolist())
        c1, c2, c3 = st.columns(3)
        ga = c1.selectbox("Group A", levels, index=0)
        gb = c2.selectbox("Group B", levels, index=min(1, len(levels) - 1))
        alt = c3.selectbox("Alternative H1", ["two-sided", "greater", "less"],
                           help="'greater' tests μ_A > μ_B; 'less' tests μ_A < μ_B")
        use_log = st.checkbox("Test log(price) (recommended: prices are right-skewed)", True)
        if ga == gb:
            st.error("Choose two different groups.")
        else:
            col = np.log(df["price"]) if use_log else df["price"]
            a, b = col[df[by] == ga], col[df[by] == gb]
            sym = {"two-sided": "≠", "greater": ">", "less": "<"}[alt]
            st.markdown(f"**H0:** μ({ga}) = μ({gb})  &nbsp;&nbsp; **H1:** μ({ga}) {sym} μ({gb})")
            eq = st.checkbox("Assume equal variances (Student) — otherwise Welch", False)
            try:
                r = two_sample_ttest(a, b, alpha, equal_var=eq, alternative=alt)
                show_result(r, alpha)
                ui.show_fig(t_plot(r["statistic"], r["df"], alpha, alt))
                st.write(f"Mean A = {r['mean_a']:.3f}, mean B = {r['mean_b']:.3f}, difference = {r['diff']:.3f}"
                         + (f", {(1 - alpha):.0%} CI of difference: ({r['ci_diff'][0]:.3f}, {r['ci_diff'][1]:.3f})" if r["ci_diff"] else ""))
                st.write(f"Effect size Cohen's d = **{r['cohens_d']:.2f}** ({r['effect_size']}). Levene equal-variance p = {r['levene_p']:.3g}.")
                mw = mann_whitney(a, b, alpha)
                st.caption(f"Robustness check — Mann-Whitney U (no normality assumption): p = {mw['p_value']:.4g} → "
                           f"{'significant' if mw['reject'] else 'not significant'}.")
                ui.show_fig(px.box(df[df[by].isin([ga, gb])], x=by, y="price", points=False, log_y=use_log))
            except ValueError as e:
                st.error(str(e))

with t2:
    mu0 = st.number_input("Hypothesised mean nightly price μ0 ($)", 1.0, 5000.0, float(round(df["price"].median())))
    alt = st.selectbox("Alternative H1", ["two-sided", "greater", "less"], key="alt1")
    sample_n = st.slider("Sample size drawn from listings", 10, len(df), min(200, len(df)))
    seed = st.number_input("Seed", 0, 9999, 1, key="s1")
    sample = df["price"].sample(sample_n, random_state=int(seed))
    sym = {"two-sided": "≠", "greater": ">", "less": "<"}[alt]
    st.markdown(f"**H0:** μ = {mu0:g} &nbsp;&nbsp; **H1:** μ {sym} {mu0:g}")
    r = one_sample_ttest(sample, mu0, alpha, alt)
    show_result(r, alpha)
    ui.show_fig(t_plot(r["statistic"], r["df"], alpha, alt))
    st.caption(f"Sample mean = {r['sample_mean']:.2f} (n = {r['n']}). Shapiro-Wilk normality p = {normality_check(sample)['p_value']:.3g} "
               "— with skewed prices and small n, treat the t-test cautiously; large n relies on the CLT.")

with t3:
    if cat:
        by = st.selectbox("Category", cat, key="anova_by")
        groups = {str(k): v["price"].apply(np.log) for k, v in df.groupby(by)}
        st.markdown(f"**H0:** all {by} groups have the same mean log-price &nbsp; **H1:** at least one differs")
        try:
            r = one_way_anova(groups, alpha)
            show_result({**r, "df": r["df_within"]}, alpha)
            st.write(f"Effect size η² = **{r['eta_squared']:.3f}** · Levene p = {r['levene_p']:.3g} · Kruskal-Wallis p = {r['kruskal_p']:.3g}")
            ui.show_df(np.exp(r["group_means"]).round(2).rename("geometric-mean price ($)").to_frame())
        except ValueError as e:
            st.error(str(e))
