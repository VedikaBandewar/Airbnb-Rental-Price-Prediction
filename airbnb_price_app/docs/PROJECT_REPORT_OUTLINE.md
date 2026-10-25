# Project Report Outline — Airbnb Rental Price Prediction using Regression

**1. Title page** — project title, name, roll no., course (Statistical Inference), guide, date.
**2. Abstract** (150 words) — problem, data, methods, headline result.

**3. Introduction & Objectives**
- Why nightly price estimation matters (hosts, guests, platforms).
- Objectives: scrape/collect data; clean & explore; apply probability, CLT, hypothesis testing, correlation; build and compare six regression models; deploy a Streamlit predictor.

**4. Dataset description**
- Source (Inside Airbnb CSV *or* the bundled synthetic dataset — state clearly which was used), size, variables, target.
- Data-quality issues found (missing values, duplicates, price ≤ 0, outliers) and how each was treated.

**5. Methodology** (one subsection each, tie to practicals)
5.1 Data collection — requests, BeautifulSoup, robots.txt/ethics, pagination, CSV export.
5.2 Cleaning & EDA — descriptive statistics, IQR outliers, visualisations.
5.3 Probability distributions — Normal/Uniform/Binomial/Poisson; price fit (log-normal) with KS test.
5.4 Central Limit Theorem & estimation — sampling distribution, SE = σ/√n, t-based CI, coverage simulation.
5.5 Hypothesis testing — H0/H1, α, p-value, Welch t-test, effect size, assumptions.
5.6 Statistical tests & correlation — paired t, ANOVA + Tukey, chi-square + Cramér's V, covariance, Pearson vs Spearman; correlation ≠ causation.
5.7 Regression pipeline — split-first design, imputation, one-hot, Min-Max vs StandardScaler, log-target, GridSearchCV, leakage controls.

**6. Algorithms** — for each of Simple LR, Multiple LR, Ridge, Lasso, Decision Tree, Random Forest: idea, equation/objective, key hyper-parameter, strengths/weaknesses.
**7. Evaluation metrics** — MAE, MSE, RMSE, R² (definitions, interpretation), cross-validation.

**8. Results & discussion**
- Summary statistics and key plots (price histogram, box plots by room type/location).
- Test outcomes (t-test, ANOVA, chi-square, correlations) with p-values and effect sizes.
- Model comparison table (from `models/model_comparison.csv`), actual-vs-predicted, residual analysis, feature importance.
- Why the best model won; ties within CV noise; limits of synthetic vs real data.

**9. Application (frontend)** — screenshots of each page, input validation, model-estimate disclaimer, scenario comparison.
**10. Testing** — unit/integration tests, what was and was not verified.
**11. Conclusion**
**12. Limitations & future scope** — real multi-city data, amenities/NLP, geospatial, seasonality, gradient boosting, prediction intervals, SHAP, cloud deployment.
**13. References** — Inside Airbnb; scikit-learn, SciPy, Streamlit docs; course material.
**Appendix** — folder structure, key code listings, requirements.
