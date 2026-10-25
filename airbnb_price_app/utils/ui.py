"""Reusable Streamlit UI components and shared (cached) data access."""
from pathlib import Path
import pandas as pd
import streamlit as st

from .data_cleaning import clean_listings
from .data_io import DATA_PATH, load_csv, load_default
from .regression import load_bundle, MODEL_DIR

CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "styles.css"
SOURCES = ["Bundled sample dataset", "Uploaded CSV", "Scraped data"]


def load_css():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="page-header"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def kpi_row(items):
    """items: list of (label, value, sub)."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>'
                     f'<div class="kpi-sub">{sub}</div></div>', unsafe_allow_html=True)


def note(text: str):
    st.markdown(f'<div class="note-box">{text}</div>', unsafe_allow_html=True)


def show_fig(fig):
    st.plotly_chart(fig, width="stretch")


def show_df(df, **kw):
    st.dataframe(df, width="stretch", **kw)


def show_mpl(fig):
    st.pyplot(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


def download_csv(df: pd.DataFrame, label: str, filename: str, key: str | None = None):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv", key=key)


@st.cache_data(show_spinner=False)
def _default_data(mtime: float) -> pd.DataFrame:
    return load_default()


def active_source() -> str:
    src = st.session_state.get("_active_source", SOURCES[0])
    ss = st.session_state
    if (src == SOURCES[1] and "uploaded_df" not in ss) or (src == SOURCES[2] and "scraped_df" not in ss):
        return SOURCES[0]
    return src


def get_raw() -> pd.DataFrame:
    src = active_source()
    if src == SOURCES[1]:
        return st.session_state["uploaded_df"]
    if src == SOURCES[2]:
        return st.session_state["scraped_df"]
    return _default_data(DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0)


@st.cache_data(show_spinner=False)
def _clean_cached(df: pd.DataFrame, outliers: str):
    return clean_listings(df, impute=True, outliers=outliers)


def get_clean(outliers: str = "none"):
    """Cleaned copy of the active dataset (duplicates + invalid prices removed, missing values imputed)."""
    return _clean_cached(get_raw(), outliers)


@st.cache_resource(show_spinner=False)
def _bundle_cached(mtime: float):
    try:
        return load_bundle(MODEL_DIR), None
    except Exception as e:  # e.g. scikit-learn version mismatch, corrupted file
        return None, f"{type(e).__name__}: {e}"


def get_bundle():
    p = MODEL_DIR / "model_bundle.joblib"
    bundle, err = _bundle_cached(p.stat().st_mtime if p.exists() else 0.0)
    if err:
        st.error("The saved model could not be loaded (it was probably trained with a different scikit-learn version). "
                 "Retrain it from the Regression Model Comparison page or run `python train_model.py`.  \nDetails: " + err)
    return bundle


def clear_model_cache():
    _bundle_cached.clear()


def money(x: float) -> str:
    return f"${x:,.0f}"
