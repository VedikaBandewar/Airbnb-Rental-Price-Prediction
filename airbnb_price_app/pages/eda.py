import io
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import ui
from utils.data_cleaning import clean_listings, iqr_bounds, missing_report
from utils.data_io import categorical_columns, numeric_columns
from utils.descriptive_stats import grouped_summary, inspect_frame, numeric_describe, summary_stats

ui.page_header("Data Cleaning and EDA", "Inspect, clean, summarise and visualise the listings.")
raw = ui.get_raw()
t1, t2, t3, t4, t5 = st.tabs(["1 · Inspect", "2 · Clean", "3 · Descriptive stats", "4 · Visualise", "5 · Location & room type"])

with t1:
    ui.kpi_row([("shape()", f"{raw.shape[0]:,} × {raw.shape[1]}", "rows × columns"),
                ("Missing cells", f"{int(raw.isna().sum().sum()):,}", f"{raw.isna().mean().mean():.1%} of all cells"),
                ("Duplicate rows", f"{int(raw.duplicated().sum()):,}", "exact copies"),
                ("Price ≤ 0 / missing", f"{int((raw['price'].isna() | (raw['price'] <= 0)).sum()):,}", "invalid target")])
    n = st.slider("Rows for head() / tail()", 3, 20, 5)
    a, b = st.columns(2)
    a.markdown("**`head()`**"); ui.show_df(raw.head(n))
    b.markdown("**`tail()`**"); ui.show_df(raw.tail(n))
    st.markdown("**`info()`** (as a table)"); ui.show_df(inspect_frame(raw))
    st.markdown("**`describe()`**"); ui.show_df(numeric_describe(raw))

with t2:
    c1, c2, c3 = st.columns(3)
    impute = c1.checkbox("Impute missing values (median / mode)", True)
    method = c2.selectbox("Outlier handling (price)", ["none", "remove", "cap"],
                          help="IQR rule: values outside [Q1 − k·IQR, Q3 + k·IQR].")
    k = c3.slider("IQR multiplier k", 1.0, 4.0, 1.5, 0.5)
    clean, rep = clean_listings(raw, impute=impute, outliers=method, k=k)
    ui.kpi_row([("Rows before", f"{rep['rows_before']:,}", ""), ("Duplicates removed", f"{rep['duplicates_removed']:,}", ""),
                ("Invalid prices removed", f"{rep['invalid_price_removed']:,}", ""),
                ("Outlier rows removed", f"{rep['outlier_rows_removed']:,}", f"method: {method}"),
                ("Rows after", f"{rep['rows_after']:,}", "")])
    st.markdown("**Missing values in the raw data**")
    mr = missing_report(raw)
    ui.show_df(mr) if len(mr) else st.success("No missing values.")
    lo, hi = iqr_bounds(raw["price"].dropna(), k)
    st.info(f"IQR fences for price with k={k}: **{lo:,.1f} – {hi:,.1f}**. "
            f"{int((raw['price'] > hi).sum()):,} listings are above the upper fence.")
    st.markdown("**Cleaned data preview**"); ui.show_df(clean.head(15))
    ui.download_csv(clean, "⬇️ Download cleaned CSV", "cleaned_listings.csv", key="eda_dl")
    ui.note("For <b>model training</b>, imputation and scaling are performed inside a scikit-learn Pipeline <i>after</i> "
            "the train/test split, so the test set never influences the preprocessing (no data leakage).")

df, _ = ui.get_clean()
with t3:
    num = numeric_columns(df)
    col = st.selectbox("Numeric column", num, index=num.index("price") if "price" in num else 0)
    st_ = summary_stats(df[col])
    ui.kpi_row([("Mean", f"{st_['mean']:,.2f}", ""), ("Median", f"{st_['median']:,.2f}", ""),
                ("Mode", f"{st_['mode']:,.2f}", ""), ("Std dev", f"{st_['std_dev']:,.2f}", f"variance {st_['variance']:,.1f}")])
    st.write("")
    ui.kpi_row([("Q1", f"{st_['Q1 (25%)']:,.2f}", ""), ("Q3", f"{st_['Q3 (75%)']:,.2f}", ""),
                ("IQR", f"{st_['IQR']:,.2f}", ""), ("Skewness", f"{st_['skewness']:.2f}", ">0 = right-skewed")])
    ui.show_df(pd.Series(st_, name="value").round(3).to_frame())
    if st_["mean"] > st_["median"]:
        st.caption("Mean > median: the distribution is pulled right by high values (typical for prices).")

with t4:
    num, cat = numeric_columns(df), categorical_columns(df)
    chart = st.selectbox("Chart type", ["Histogram", "Box plot", "Bar chart", "Pie chart", "Scatter plot", "Line chart"])
    cap = st.checkbox("Limit price to 99th percentile for readability", True)
    d = df[df["price"] <= df["price"].quantile(0.99)] if cap else df
    if chart == "Histogram":
        x = st.selectbox("Column", num, index=num.index("price"))
        ui.show_fig(px.histogram(d, x=x, nbins=st.slider("Bins", 10, 100, 40), marginal="box"))
    elif chart == "Box plot":
        by = st.selectbox("Group by", cat)
        ui.show_fig(px.box(d, x=by, y="price", points=False))
    elif chart == "Bar chart":
        by = st.selectbox("Group by", cat)
        g = d.groupby(by)["price"].mean().sort_values(ascending=False).reset_index()
        ui.show_fig(px.bar(g, x=by, y="price", labels={"price": "Mean price ($)"}))
    elif chart == "Pie chart":
        by = st.selectbox("Category", cat)
        ui.show_fig(px.pie(d, names=by, title=f"Share of listings by {by}", hole=0.35))
    elif chart == "Scatter plot":
        x = st.selectbox("X variable", [c for c in num if c != "price"])
        color = st.selectbox("Colour by", ["(none)"] + cat)
        s = d.sample(min(len(d), 2500), random_state=1)
        ui.show_fig(px.scatter(s, x=x, y="price", color=None if color == "(none)" else color, opacity=0.5))
    else:
        x = st.selectbox("X variable (averaged into bins)", [c for c in num if c != "price"])
        t = d.copy()
        t["_bin"] = pd.qcut(t[x], q=min(15, t[x].nunique()), duplicates="drop")
        g = t.groupby("_bin", observed=True).agg(x=(x, "mean"), price=("price", "mean")).reset_index()
        ui.show_fig(px.line(g, x="x", y="price", markers=True, labels={"x": x, "price": "Mean price ($)"}))

with t5:
    cat = categorical_columns(df)
    by = st.selectbox("Group prices by", cat, key="grp")
    ui.show_df(grouped_summary(df, by))
    c1, c2 = st.columns(2)
    d = df[df["price"] <= df["price"].quantile(0.99)]
    with c1:
        ui.show_fig(px.box(d, x=by, y="price", points=False, title=f"Price by {by}"))
    with c2:
        if len(cat) > 1:
            other = st.selectbox("Split by", [c for c in cat if c != by])
            g = d.groupby([by, other])["price"].mean().reset_index()
            ui.show_fig(px.bar(g, x=by, y="price", color=other, barmode="group", title="Mean price"))
