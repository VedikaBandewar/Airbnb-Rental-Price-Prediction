"""Airbnb Rental Price Prediction using Regression — main Streamlit entry point.
Run with:  streamlit run app.py
"""
import streamlit as st

from utils import ui
from utils.data_io import load_csv, validate_dataset

st.set_page_config(page_title="Airbnb Price Prediction", page_icon="🏠", layout="wide")
ui.load_css()

PAGES = [
    st.Page("pages/home.py", title="Home Dashboard", icon="🏠", default=True),
    st.Page("pages/web_scraping.py", title="Web Scraping", icon="🕸️"),
    st.Page("pages/eda.py", title="Data Cleaning and EDA", icon="🧹"),
    st.Page("pages/distributions.py", title="Probability Distributions", icon="🎲"),
    st.Page("pages/clt.py", title="Central Limit Theorem", icon="🔔"),
    st.Page("pages/hypothesis.py", title="Hypothesis Testing", icon="⚖️"),
    st.Page("pages/correlation_tests.py", title="Correlation and Statistical Tests", icon="🔗"),
    st.Page("pages/regression_models.py", title="Regression Model Comparison", icon="📈"),
    st.Page("pages/predictor.py", title="Airbnb Price Predictor", icon="💰"),
    st.Page("pages/about.py", title="About the Project", icon="ℹ️"),
]
nav = st.navigation(PAGES)

with st.sidebar:
    st.markdown("### Dataset")
    if "_pending_source" in st.session_state:          # set by upload / scraping pages, applied before the widget exists
        st.session_state["data_source"] = st.session_state.pop("_pending_source")
    avail = [ui.SOURCES[0]] + ([ui.SOURCES[1]] if "uploaded_df" in st.session_state else []) \
        + ([ui.SOURCES[2]] if "scraped_df" in st.session_state else [])
    if st.session_state.get("data_source") not in avail:
        st.session_state["data_source"] = ui.SOURCES[0]
    st.radio("Active dataset", avail, key="data_source")
    st.session_state["_active_source"] = st.session_state["data_source"]   # plain key: survives page switches
    with st.expander("Upload your own CSV"):
        up = st.file_uploader("Airbnb listings CSV (needs a 'price' column)", type="csv", key="uploader")
        if up is not None and st.session_state.get("_uploaded_name") != up.name:
            try:
                df_up = load_csv(up)
                ok, msgs = validate_dataset(df_up)
                if ok:
                    st.session_state["uploaded_df"] = df_up
                    st.session_state["_uploaded_name"] = up.name
                    st.session_state["_pending_source"] = ui.SOURCES[1]
                    st.rerun()
                else:
                    st.error("Dataset rejected: " + " ".join(msgs))
            except Exception as e:  # unreadable / not a CSV
                st.error(f"Could not read that file as CSV: {e}")
    st.caption("The bundled dataset is **synthetic** (simulated), generated for learning purposes.")

nav.run()
