"""Smoke tests: render every Streamlit page headlessly and fail on any exception.
Run:  python -m pytest tests/test_app_pages.py -q"""
import sys
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAGES = ["home", "web_scraping", "eda", "distributions", "clt", "hypothesis",
         "correlation_tests", "regression_models", "predictor", "about"]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    at.run()
    if page != "home":
        at.switch_page(f"pages/{page}.py").run()
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    at = _run(page)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_every_eda_chart_type_renders():
    at = _run("eda")
    charts = [s for s in at.selectbox if s.label == "Chart type"][0].options
    for chart in charts:
        [s for s in at.selectbox if s.label == "Chart type"][0].set_value(chart)
        at.run()
        assert not at.exception, (chart, [e.value for e in at.exception])


def test_every_distribution_renders():
    at = _run("distributions")
    names = at.selectbox[0].options
    for name in names:
        at.selectbox[0].set_value(name)
        at.run()
        assert not at.exception, (name, [e.value for e in at.exception])


def test_predictor_predicts_and_compares():
    at = _run("predictor")
    at.button[0].click().run()          # first form-submit = "Predict price"
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Estimated nightly price" in html and "not an actual Airbnb quote" in " ".join(w.value for w in at.warning)
    at.button[1].click().run()          # "Add to comparison"
    assert len(at.session_state["scenarios"]) == 1 and not at.exception


def test_pages_work_on_small_scraped_dataset():
    """Reduced schema (no bedrooms/beds/...), 60 rows, string ids — every page must still render.
    (Does not retrain, so the shipped models/ are untouched.)"""
    from utils.data_io import load_default
    from utils.scraping import demo_scrape
    scraped, _ = demo_scrape(load_default(), 3, 20)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.session_state["scraped_df"] = scraped
    at.session_state["_pending_source"] = "Scraped data"
    at.run()
    assert at.session_state["_active_source"] == "Scraped data" and not at.exception
    for page in PAGES[1:]:
        at.switch_page(f"pages/{page}.py").run()
        assert at.session_state["_active_source"] == "Scraped data"
        assert not at.exception, (page, [e.value for e in at.exception])
