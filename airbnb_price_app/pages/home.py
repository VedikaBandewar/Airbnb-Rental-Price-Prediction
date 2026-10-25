import plotly.express as px
import streamlit as st
from utils import ui
from utils.descriptive_stats import summary_stats

ui.page_header("Airbnb Rental Price Prediction", "Statistical analysis + regression on Airbnb-style listings — one integrated project.")
raw = ui.get_raw()
df, rep = ui.get_clean()
bundle = ui.get_bundle()
st.markdown(f'<span class="dataset-badge">Active dataset: {ui.active_source()}</span>', unsafe_allow_html=True)
st.write("")
s = summary_stats(df["price"])
best = bundle["results"].iloc[0] if bundle else None
ui.kpi_row([("Listings (cleaned)", f"{len(df):,}", f"{rep['rows_before'] - rep['rows_after']:,} rows removed"),
            ("Average price", ui.money(s["mean"]), "per night"),
            ("Median price", ui.money(s["median"]), f"IQR {ui.money(s['IQR'])}"),
            ("Locations", f"{df['neighbourhood'].nunique() if 'neighbourhood' in df else '—'}", "neighbourhoods"),
            ("Best model", bundle["best_name"].replace(" Regression", "") if bundle else "Not trained",
             f"Test R² = {best['R2']:.2f}" if best is not None else "see Regression page")])
st.write("")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Price distribution")
    cap = df["price"].quantile(0.99)
    ui.show_fig(px.histogram(df[df["price"] <= cap], x="price", nbins=50, labels={"price": "Nightly price ($)"}))
    st.caption("Right-skewed: most listings are cheap, a few are very expensive (shown up to the 99th percentile).")
with c2:
    if "room_type" in df:
        st.subheader("Average price by room type")
        g = df.groupby("room_type")["price"].mean().sort_values().reset_index()
        ui.show_fig(px.bar(g, x="price", y="room_type", orientation="h", labels={"price": "Mean price ($)", "room_type": ""}))
st.subheader("Where each practical appears in this app")
ui.show_df(__import__("pandas").DataFrame({
    "Practical topic": ["Web scraping, BeautifulSoup, CSV export", "Pandas cleaning, descriptive stats, plots",
                        "Probability distributions, random variables", "Central Limit Theorem, standard error, CI",
                        "Hypothesis testing, p-values", "t-tests, ANOVA, chi-square, correlation",
                        "Regression models + evaluation", "Streamlit frontend & prediction"],
    "Page": ["Web Scraping", "Data Cleaning and EDA", "Probability Distributions", "Central Limit Theorem",
             "Hypothesis Testing", "Correlation and Statistical Tests", "Regression Model Comparison",
             "Airbnb Price Predictor"]}), hide_index=True)
st.subheader("Dataset preview")
ui.show_df(raw.head(10))
ui.download_csv(df, "⬇️ Download cleaned dataset (CSV)", "cleaned_listings.csv", key="home_dl")
