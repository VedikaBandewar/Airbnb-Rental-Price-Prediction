"""Unit tests for every utility module. Run:  python -m pytest -q"""
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import correlation, data_cleaning, descriptive_stats, distributions, evaluation, hypothesis_testing as ht
from utils import regression, scraping
from utils.data_io import load_default, standardise_columns, validate_dataset


@pytest.fixture(scope="module")
def raw():
    return load_default()


@pytest.fixture(scope="module")
def clean(raw):
    return data_cleaning.clean_listings(raw)[0]


# ---------- data ----------
def test_dataset_has_expected_problems(raw):
    assert raw.duplicated().sum() >= 30 and raw.isna().sum().sum() > 0 and (raw["price"] <= 0).sum() >= 1


def test_standardise_inside_airbnb_style():
    df = pd.DataFrame({"Price": ["$1,200.00", "$85.00"], "neighbourhood_cleansed": ["A", "B"], "bathrooms_text": ["1.5 baths", "Half-bath"]})
    out = standardise_columns(df)
    assert out["price"].tolist() == [1200.0, 85.0] and "neighbourhood" in out and out["bathrooms"].tolist() == [1.5, 0.5]


def test_validate_dataset_rejects_bad_input():
    assert not validate_dataset(pd.DataFrame({"a": [1, 2]}))[0]
    assert not validate_dataset(pd.DataFrame({"price": [1.0] * 10, "x": range(10)}))[0]  # too few rows


def test_cleaning(raw):
    out, rep = data_cleaning.clean_listings(raw)
    assert out.duplicated().sum() == 0 and (out["price"] > 0).all() and out.drop(columns=["name"]).isna().sum().sum() == 0
    capped, _ = data_cleaning.clean_listings(raw, outliers="cap")
    removed, r2 = data_cleaning.clean_listings(raw, outliers="remove")
    assert capped["price"].max() < out["price"].max() and r2["outlier_rows_removed"] > 0 and len(removed) < len(out)


def test_descriptive_stats(clean):
    s = descriptive_stats.summary_stats(clean["price"])
    assert s["IQR"] == pytest.approx(s["Q3 (75%)"] - s["Q1 (25%)"]) and s["variance"] == pytest.approx(s["std_dev"] ** 2)
    assert len(descriptive_stats.grouped_summary(clean, "room_type")) == clean["room_type"].nunique()


# ---------- distributions / CLT ----------
@pytest.mark.parametrize("name,p", [("Normal", dict(loc=5, scale=2)), ("Uniform", dict(low=1, high=9)),
                                    ("Binomial", dict(n=20, p=0.3)), ("Poisson", dict(lam=4))])
def test_sample_moments_match_theory(name, p):
    x = distributions.sample_distribution(name, 200_000, 1, **p)
    m = distributions.theoretical_moments(name, **p)
    assert x.mean() == pytest.approx(m["mean"], rel=0.02) and x.var() == pytest.approx(m["variance"], rel=0.03)
    gx, gy, kind = distributions.theoretical_curve(name, **p)
    assert (gy >= 0).all() and (kind == "pmf") == (name in ("Binomial", "Poisson"))


def test_clt_se_and_skew(clean):
    pop = clean["price"].to_numpy()
    means = distributions.sampling_means(pop, 100, 2000, 0)
    s = distributions.clt_summary(pop, means, 100)
    assert s["empirical_SE"] == pytest.approx(s["theoretical_SE"], rel=0.1)
    assert abs(s["skew_sample_means"]) < abs(s["skew_population"])


def test_confidence_interval_coverage(clean):
    cov = distributions.ci_coverage(clean["price"], 100, 600, 0.95, 1)
    assert 0.90 <= cov["coverage"] <= 0.99
    with pytest.raises(ValueError):
        distributions.confidence_interval([1.0])


# ---------- hypothesis tests ----------
def test_ttest_detects_difference_and_null():
    rng = np.random.default_rng(0)
    r = ht.two_sample_ttest(rng.normal(100, 10, 300), rng.normal(110, 10, 300))
    assert r["reject"] and r["diff"] < 0
    r0 = ht.two_sample_ttest(rng.normal(100, 10, 300), rng.normal(100, 10, 300))
    assert r0["p_value"] > 0.01
    with pytest.raises(ValueError):
        ht.two_sample_ttest([1], [1, 2])


def test_paired_anova_chi2(clean):
    r = ht.paired_ttest(clean["weekend_price"], clean["weekday_price"])
    assert r["reject"] and r["mean_diff"] > 0
    groups = {k: np.log(v["price"]) for k, v in clean.groupby("room_type")}
    a = ht.one_way_anova(groups)
    assert a["reject"] and a["tukey"] is not None and 0 < a["eta_squared"] < 1
    c = ht.chi_square_independence(clean, "room_type", "neighbourhood")
    assert c["p_value"] > 0.001 and c["dof"] == (clean["room_type"].nunique() - 1) * (clean["neighbourhood"].nunique() - 1)
    with pytest.raises(ValueError):
        ht.chi_square_independence(clean.assign(k="x"), "k", "room_type")


def test_one_sample(clean):
    r = ht.one_sample_ttest(clean["price"], clean["price"].mean())
    assert r["p_value"] == pytest.approx(1.0, abs=1e-6) and not r["reject"]


# ---------- correlation ----------
def test_correlation_matches_numpy(clean):
    x, y = clean["accommodates"], clean["price"]
    assert correlation.pearson(x, y)["r"] == pytest.approx(np.corrcoef(x, y)[0, 1])
    assert correlation.covariance(x, y) == pytest.approx(np.cov(x, y)[0, 1])
    assert correlation.spearman(x, y)["rho"] > 0.3
    tab = correlation.correlation_with_target(clean, "price", ["accommodates", "number_of_reviews"])
    assert set(tab["variable"]) == {"accommodates", "number_of_reviews"}
    cm = correlation.correlation_matrix(clean, ["price", "beds", "bedrooms"])
    assert np.allclose(np.diag(cm), 1)


# ---------- evaluation ----------
def test_metrics():
    m = evaluation.regression_metrics([1, 2, 3], [1, 2, 5])
    assert m["MAE"] == pytest.approx(2 / 3) and m["MSE"] == pytest.approx(4 / 3) and m["RMSE"] == pytest.approx(np.sqrt(4 / 3))
    assert evaluation.regression_metrics([1, 2, 3], [1, 2, 3])["R2"] == 1.0


# ---------- regression ----------
@pytest.fixture(scope="module")
def trained(raw, tmp_path_factory):
    return regression.train_and_save(raw, model_dir=tmp_path_factory.mktemp("m"), cv=3), tmp_path_factory


def test_six_models_and_reasonable_accuracy(trained):
    out, _ = trained
    res = out["results"]
    assert set(res["Model"]) == set(regression.MODEL_NAMES)
    assert res["R2"].max() > 0.5 and (res["RMSE"] > 0).all()
    assert out["best_name"] == res.sort_values("CV RMSE").iloc[0]["Model"]          # chosen by CV, not test
    assert res.set_index("Model").loc["Simple Linear Regression", "R2"] < res["R2"].max()
    assert "weekday_price" not in out["numeric"] and "weekend_price" not in out["numeric"]  # no target leakage


def test_split_is_disjoint_and_no_nan_in_pipeline_output(trained):
    out, _ = trained
    assert not set(out["X_train"].index) & set(out["X_test"].index)
    prep = out["models"]["Multiple Linear Regression"]["pipeline"].named_steps["prep"]
    assert not np.isnan(prep.transform(out["X_test"][out["numeric"] + out["categorical"]])).any()


@pytest.mark.parametrize("scaler", ["standard", "minmax", "none"])
def test_scalers_run(raw, scaler):
    out = regression.compare_models(raw, scaler=scaler, cv=3, models=["Ridge Regression"])
    assert len(out["results"]) == 1


def test_saved_model_roundtrip_and_prediction(trained):
    out, tpf = trained
    d = Path(str(out["bundle"]["trained_at"]) and tpf.getbasetemp()).glob("m*/model_bundle.joblib")
    path = sorted(d)[-1].parent
    b = regression.load_bundle(path)
    inp = {c: b["ranges"][c][2] for c in b["numeric"]} | {c: b["categories"][c][0] for c in b["categorical"]}
    p1 = regression.predict_price(b, inp)
    assert 10 < p1 < 2000
    # more guests / bedrooms should not make the best model cheaper
    bigger = {**inp, "accommodates": inp["accommodates"] + 3}
    assert regression.predict_price(b, bigger) >= p1 * 0.9
    assert (path / "preprocessor.joblib").exists() and (path / "model_comparison.csv").exists()


def test_input_validation(trained):
    b = trained[0]["bundle"]
    ok = {c: b["ranges"][c][2] for c in b["numeric"]} | {c: b["categories"][c][0] for c in b["categorical"]}
    assert regression.validate_inputs(b, ok) == []
    assert regression.validate_inputs(b, {**ok, "availability_365": 400})
    assert regression.validate_inputs(b, {**ok, "room_type": "Castle"})
    assert regression.validate_inputs(b, {**ok, "accommodates": "abc"})
    with pytest.raises(ValueError):
        regression.predict_price(b, {**ok, "accommodates": -2})


def test_reduced_dataset_uses_only_available_features(raw):
    small = raw[["neighbourhood", "room_type", "accommodates", "price"]]
    out = regression.compare_models(small, cv=3, models=["Ridge Regression", "Simple Linear Regression"])
    assert out["numeric"] == ["accommodates"] and set(out["categorical"]) == {"neighbourhood", "room_type"}


def test_unseen_category_does_not_crash(raw):
    out = regression.compare_models(raw, cv=3, models=["Ridge Regression"])
    pipe = out["models"]["Ridge Regression"]["pipeline"]
    X = out["X_test"].head(3).copy()
    X["room_type"] = "Spaceship"
    assert np.isfinite(pipe.predict(X[out["numeric"] + out["categorical"]])).all()


# ---------- scraping (real local HTTP server) ----------
@pytest.fixture(scope="module")
def server(raw):
    base = raw[raw["price"] > 0].drop_duplicates().reset_index(drop=True)
    hits = {"flaky": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path.startswith("/robots.txt"):
                self.send_response(404); self.end_headers(); return
            if self.path.startswith("/flaky"):
                hits["flaky"] += 1
                code = 503 if hits["flaky"] == 1 else 200
                self.send_response(code); self.end_headers(); self.wfile.write(scraping.build_demo_page(base, 1, 5).encode()); return
            if self.path.startswith("/list?page="):
                page = int(self.path.split("=")[1])
                if page > 3:
                    self.send_response(404); self.end_headers(); return
                body = scraping.build_demo_page(base, page, 10).encode()
                self.send_response(200); self.send_header("Content-Type", "text/html"); self.end_headers(); self.wfile.write(body); return
            self.send_response(500); self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_fetch_and_parse(server):
    r = scraping.fetch_page(server + "/list?page=1")
    assert r["ok"] and r["status_code"] == 200 and "listing-card" in r["text"]
    rows = scraping.parse_listings(r["text"], base_url=server)
    assert len(rows) == 10 and rows[0]["url"].startswith(server + "/rooms/")


def test_pagination_stops_on_404_and_cleans(server):
    df, log = scraping.scrape_pages(server + "/list?page={page}", 1, 10, delay=0, respect_robots=True)
    assert len(df) == 30 and any(l["status"] == 404 for l in log)
    assert (df["price"].dropna() > 0).all()                       # missing prices stay NaN (cleaned downstream)
    assert len(data_cleaning.basic_clean(df)) == df["price"].notna().sum()


def test_retry_on_503_and_failure_handling(server):
    assert scraping.fetch_page(server + "/flaky", retries=2, backoff=0.01)["ok"]
    bad = scraping.fetch_page(server + "/nope", retries=0)
    assert not bad["ok"] and bad["status_code"] == 500 and bad["error"]
    dead = scraping.fetch_page("http://127.0.0.1:9/", retries=0, timeout=1)
    assert not dead["ok"] and dead["error"] and dead["status_code"] is None


def test_missing_fields_become_nan():
    html = '<div class="listing-card" data-id="1"><h3 class="listing-title">X</h3><span class="price">$1,250 / night</span></div>'
    df = scraping.clean_scraped(pd.DataFrame(scraping.parse_listings(html)))
    assert df.loc[0, "price"] == 1250 and np.isnan(df.loc[0, "number_of_reviews"]) and df.loc[0, "neighbourhood"] is None
    assert scraping.parse_listings("<html></html>") == []


def test_id_like_columns_are_not_grouping_columns():
    from utils.data_io import categorical_columns
    df = pd.DataFrame({"id": [str(i) for i in range(60)], "room_type": ["a", "b"] * 30, "price": range(60)})
    assert categorical_columns(df) == ["room_type"]
