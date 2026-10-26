import streamlit as st
from utils import ui

ui.page_header("About the Project", "Airbnb Rental Price Prediction using Regression — Statistical Inference practicals 1–8.")
st.markdown("""
### Objective
Analyse Airbnb-style rental listings, apply the statistical-inference toolkit from Practicals 1–8, and predict the
estimated nightly price with regression models served through one Streamlit application.

### Honest limitations
- The **bundled dataset is synthetic** (simulated with a known price formula plus noise). It lets the project run offline,
  but conclusions about "what drives Airbnb prices" apply to the simulation, not to any real city. For real results, upload an
  [Inside Airbnb](http://insideairbnb.com/get-the-data) `listings.csv`, then retrain.
- Because the synthetic prices are log-linear, linear models do very well; on real data tree ensembles often win.
- Predictions are estimates, **not Airbnb quotes**. Scraping airbnb.com is against its terms of service.
- Correlation and feature importance describe association, not causation.

### Leakage safeguards
Train/test split first · imputation, scaling and encoding inside the Pipeline · hyper-parameters tuned by cross-validation on
training data only · model selected by CV, not by test score · target-like columns (`weekday_price`, `weekend_price`) excluded from features.

See `README.md` and `docs/` for setup, the project-report outline and viva questions.
""")
