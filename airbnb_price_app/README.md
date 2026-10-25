# Airbnb Rental Price Prediction using Regression

One integrated Streamlit application that covers Statistical Inference Practicals 1–8: web scraping, cleaning & EDA,
probability distributions, the Central Limit Theorem, hypothesis testing, t-tests / ANOVA / chi-square / correlation,
six regression models with evaluation, and an interactive price predictor.

> **Read this first (honesty notes)**
> * The bundled dataset `data/airbnb_listings.csv` is **synthetic** — simulated by `scripts/generate_dataset.py`
>   with deliberately injected missing values, duplicates, invalid prices and outliers. It is *not* real Airbnb data
>   (Airbnb's site could not be accessed while building this, and scraping airbnb.com violates its terms anyway).
>   To use real data, download a city's `listings.csv` from <http://insideairbnb.com/get-the-data>, upload it in the sidebar,
>   then click **Train all models**. Column names and `"$1,234.00"` prices are normalised automatically.
> * Because the synthetic prices follow a log-linear formula, linear models win here. On real Airbnb data, tree ensembles
>   (Random Forest) frequently do better — the app picks the winner by cross-validation, it does not assume one.
> * Predictions are model estimates, **not Airbnb quotes**.

## 1. Setup in VS Code (Windows)

Requires **Python 3.10 or newer** (check with `py --version`).

```powershell
# 1. open the project folder in VS Code  (File > Open Folder > airbnb_price_app)
# 2. open a terminal (Ctrl+`) and create a virtual environment
py -m venv .venv
.venv\Scripts\activate
#    if PowerShell blocks activation:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 3. install dependencies
pip install -r requirements.txt
# 4. (optional) regenerate the dataset and retrain the models
python scripts/generate_dataset.py
python train_model.py
# 5. run the app
streamlit run app.py
```

Choose the `.venv` interpreter in VS Code (Ctrl+Shift+P → *Python: Select Interpreter*). `.vscode/launch.json` also has
a **Streamlit** debug configuration (F5). The app opens at <http://localhost:8501>.

If the saved model cannot be loaded (e.g. different scikit-learn version), the app shows a message — just run
`python train_model.py` or click **Train all models** on the Regression page.

> I developed and tested this on Linux with Python 3.12 (pandas 3.0, scikit-learn 1.8, Streamlit 1.66). I could not run it
> on Windows, so the commands above are standard but unverified there. The code avoids Python-3.12-only syntax and uses
> `pathlib` for paths.

## 2. Project structure

```
app.py                 main entry: navigation (10 pages), dataset selector, CSS
pages/                 one file per sidebar page (home, web_scraping, eda, distributions, clt, hypothesis,
                       correlation_tests, regression_models, predictor, about)
utils/
  data_io.py           loading, column standardisation, validation
  scraping.py          requests + BeautifulSoup, robots.txt, retries, pagination, offline demo
  data_cleaning.py     duplicates, missing values, outliers (IQR remove/cap)
  descriptive_stats.py mean/median/mode/variance/std/quartiles/IQR/skew
  distributions.py     normal/uniform/binomial/Poisson, sampling, CLT, SE, CI, coverage
  hypothesis_testing.py  t-tests, paired, ANOVA(+Tukey), chi-square, Mann-Whitney
  correlation.py       covariance, Pearson, Spearman, matrices
  regression.py        pipeline, 6 models, CV tuning, Joblib persistence, validation, prediction
  evaluation.py        MAE / MSE / RMSE / R², residuals
  ui.py                KPI cards, shared cached data access
data/                  airbnb_listings.csv (synthetic, 6,040 rows), sample_listings.csv (60 clean rows)
models/                model_bundle.joblib, best_model_pipeline.joblib, preprocessor.joblib, model_comparison.csv
assets/styles.css      shared styling
scripts/generate_dataset.py     synthetic data generator
train_model.py         command-line training
tests/                 pytest suite
docs/                  PROJECT_REPORT_OUTLINE.md, VIVA_QA.md
```

## 3. How the modules map to the app

| Page | What you can do |
|---|---|
| Home Dashboard | KPI cards, price histogram, price by room type, practical→page map, dataset preview, CSV download |
| Web Scraping | Offline demo (generated HTML parsed with BeautifulSoup) or live URL with robots.txt check, status codes, selectors, pagination, error log, CSV export, "use as active dataset" |
| Data Cleaning and EDA | `head/tail/info/shape/describe`, missing + duplicates + outliers (IQR), mean/median/mode/variance/std/quartiles/IQR, histogram, box, bar, pie, scatter, line, price by location & room type |
| Probability Distributions | Normal / Uniform / Binomial / Poisson with adjustable parameters and sample size vs theory; price fit vs Normal & Log-normal (KS test, Q–Q plot) |
| Central Limit Theorem | Sampling experiment on real prices, SE vs n, confidence intervals, 95 %-CI coverage simulation |
| Hypothesis Testing | H0/H1, α, p-value, t-test (Welch/Student, one/two-sided), one-sample, ANOVA, rejection-region plot, effect sizes, Mann-Whitney check |
| Correlation and Statistical Tests | Independent & paired t-test, ANOVA + Tukey, chi-square (+Cramér's V), covariance, Pearson, Spearman, heatmaps |
| Regression Model Comparison | Train/retrain, comparison table, actual vs predicted, residuals, feature importance, Min-Max vs Standard scaling demo, simple regression plot |
| Airbnb Price Predictor | Validated form (only features present in the training data), price estimate, what-if chart, scenario comparison, model performance |
| About the Project | Mapping, limitations, leakage safeguards |

## 4. Modelling method (and how leakage is avoided)

1. `basic_clean`: drop exact duplicates and rows with missing / non-positive price (row-level, learns nothing from data).
2. **Split first** (80/20). Only a whitelist of listing attributes is used as features — `weekday_price` / `weekend_price`
   are *excluded* because they are near-copies of the target.
3. Imputation (median / most-frequent), scaling (`StandardScaler`, Min-Max or none) and one-hot encoding live **inside the
   scikit-learn `Pipeline`**, so they are fitted on training rows only and re-fitted in each CV fold.
4. Models: Simple Linear (best single numeric predictor chosen on train), Multiple Linear, Ridge, Lasso, Decision Tree,
   Random Forest. Hyper-parameters tuned with `GridSearchCV` (5-fold) on the training set.
5. Target: `log(price)` by default (prices are right-skewed); predictions are back-transformed so all metrics are in dollars.
6. **Model selection uses cross-validated RMSE on the training data**; the test set is only used for the final report.
7. Extreme prices (> Q3 + 3·IQR, fences computed from the *training* targets) are excluded from fitting and from the headline
   test metrics; the table also shows `RMSE (all rows)` so nothing is hidden. Without this, ~1 % injected extreme prices
   dominate squared error (test RMSE ≈ 95 instead of ≈ 25).

### Results on the bundled synthetic dataset (seed 42)

| Model | MAE ($) | RMSE ($) | R² | CV RMSE ($) |
|---|---:|---:|---:|---:|
| Multiple Linear Regression | 15.80 | 24.66 | 0.69 | 22.18 |
| Ridge Regression | 15.80 | 24.66 | 0.69 | 22.18 |
| Lasso Regression | 15.80 | 24.66 | 0.69 | 22.19 |
| Random Forest Regression | 16.32 | 25.37 | 0.68 | 22.92 |
| Decision Tree Regression | 16.94 | 26.43 | 0.65 | 24.11 |
| Simple Linear Regression | 21.82 | 31.53 | 0.50 | 31.22 |

Multiple/Ridge/Lasso are statistically tied (differences ≪ one CV standard deviation of ≈ 0.4): with ~5,000 rows and ~19
features there is little over-fitting for regularisation to remove. The selected model is the one with the lowest CV RMSE.
These numbers describe the **synthetic** data only.

## 5. Testing

```powershell
python -m pytest -q                      # everything (≈ 30 s)
python -m pytest tests/test_modules.py   # unit tests of utils/
python -m pytest tests/test_app_pages.py # renders every Streamlit page headlessly
```

Status when delivered: **44 tests passed** (Linux, Python 3.12). What they cover:

| Module | Automated checks |
|---|---|
| 1 Scraping | real local HTTP server: GET + parse, pagination stops on 404, retry on 503, connection failure handled, missing fields → NaN |
| 2 Cleaning/EDA | duplicates/invalid prices removed, imputation, IQR remove/cap, descriptive stats identities, all chart types render |
| 3 Distributions | sample mean/variance match theory for all 4 distributions; every distribution renders |
| 4 CLT | empirical SE ≈ σ/√n, skew shrinks, 95 % CI coverage in [90 %, 99 %] |
| 5–6 Tests/correlation | t-test detects a planted difference and not a null, paired, ANOVA + Tukey, chi-square df, Pearson/covariance equal NumPy |
| 7 Regression | six models, best chosen by CV, no target-leakage columns, disjoint split, 3 scalers, unseen category, reduced schema |
| 8 Predictor | save/load round-trip, input validation, prediction, scenario comparison, corrupt-model handling |

Also verified manually in a sandbox: scraped data (reduced schema) and an Inside-Airbnb-style upload through every page,
retraining on a 2-feature dataset, and the predictor adapting its inputs. **Not verified:** a real browser session
(clicking by hand), Windows, and live scraping of any real website.

Manual checklist: open each sidebar page and change one control; on *Predictor* try availability = 400 (the field caps at
365) and compare two scenarios; upload a CSV with no `price` column (should be rejected with a clear message).

## 6. Limitations / future scope
Real multi-city data, text/amenity features (NLP), geospatial features, seasonality & calendar prices, gradient boosting and
quantile regression for prediction intervals, SHAP explanations, deployment to Streamlit Community Cloud.
