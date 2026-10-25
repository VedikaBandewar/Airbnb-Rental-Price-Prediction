import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as sps
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import ui
from utils.data_io import validate_dataset
from utils.evaluation import residual_summary, residuals
from utils.regression import MODEL_NAMES, train_and_save

ui.page_header("Regression Model Comparison", "Six models, one leakage-safe pipeline, evaluated with MAE, MSE, RMSE and R².")
raw = ui.get_raw()
bundle = ui.get_bundle()

with st.expander("⚙️ Train / retrain on the active dataset", expanded=bundle is None):
    c1, c2, c3, c4 = st.columns(4)
    test_size = c1.slider("Test set share", 0.1, 0.4, 0.2, 0.05)
    scaler = c2.selectbox("Numeric scaling", ["standard", "minmax", "none"], format_func=lambda s: {"standard": "StandardScaler", "minmax": "Min-Max", "none": "None"}[s])
    cv = c3.slider("CV folds", 3, 10, 5)
    seed = c4.number_input("Random state", 0, 9999, 42)
    log_t = st.checkbox("Model log(price) (back-transformed to dollars for all metrics)", True)
    trim = st.checkbox("Exclude extreme prices (> Q3 + 3·IQR of the TRAIN set) from fitting and from the headline test metrics", True)
    st.caption("Training overwrites the files in `models/`. Hyper-parameters are tuned with GridSearchCV on the training set only.")
    if st.button("🚀 Train all models", type="primary"):
        ok, msgs = validate_dataset(raw)
        if not ok:
            st.error("Cannot train: " + " ".join(msgs))
        else:
            try:
                with st.spinner("Training 6 models with cross-validation..."):
                    train_and_save(raw, test_size=test_size, scaler=scaler, cv=cv, random_state=int(seed), log_target=log_t, trim_train_outliers=trim)
                ui.clear_model_cache()
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

bundle = ui.get_bundle()
if bundle is None:
    st.warning("No trained model found. Open the panel above and click **Train all models** (or run `python train_model.py`).")
    st.stop()

res, cfg = bundle["results"], bundle["config"]
ui.kpi_row([("Best model", bundle["best_name"].replace(" Regression", ""), "lowest cross-validated RMSE"),
            ("Test RMSE", f"${res.iloc[0]['RMSE']:.2f}", f"MAE ${res.iloc[0]['MAE']:.2f}"),
            ("Test R²", f"{res.iloc[0]['R2']:.3f}", "share of variance explained"),
            ("Training rows", f"{bundle['n_rows']:,}", f"{bundle['n_trimmed']} extreme prices trimmed from train"),
            ("Setup", f"{cfg['scaler']} · {'log' if cfg['log_target'] else 'raw'} price", f"CV={cfg['cv']}, test={cfg['test_size']:.0%}")])
st.caption(f"Model trained {bundle['trained_at']}. The best model is chosen by **cross-validated RMSE on the training data**, not by peeking at the test set.")
tabs = st.tabs(["Comparison", "Actual vs predicted", "Residual analysis", "Feature importance", "Preprocessing", "Simple regression"])

with tabs[0]:
    show = res[["Model", "MAE", "MSE", "RMSE", "R2", "RMSE (all rows)", "CV RMSE", "CV RMSE std", "Best params"]].copy()
    ui.show_df(show.style.format({c: "{:,.3f}" for c in ["MAE", "MSE", "RMSE", "R2", "RMSE (all rows)", "CV RMSE", "CV RMSE std"]})
               .highlight_min(subset=["MAE", "MSE", "RMSE", "CV RMSE"], color="#cfe9d4").highlight_max(subset=["R2"], color="#cfe9d4"), hide_index=True)
    st.caption("MAE/MSE/RMSE/R² are measured on the held-out test set. 'RMSE (all rows)' also includes the extreme-price listings "
               "that were excluded from the headline metrics. Green = best in column.")
    m = st.radio("Metric to plot", ["RMSE", "MAE", "R2", "CV RMSE"], horizontal=True)
    ui.show_fig(px.bar(res.sort_values(m, ascending=(m != "R2")), x=m, y="Model", orientation="h", color="Model").update_layout(showlegend=False))
    st.info("Differences between the top models are often within one CV standard deviation — treat close scores as ties and "
            "prefer the simpler, more interpretable model.")
    ui.download_csv(res.drop(columns=["Best params"]), "⬇️ Download comparison table", "model_comparison.csv", key="cmp")

tp = bundle["test_predictions"]
with tabs[1]:
    name = st.selectbox("Model", [m for m in MODEL_NAMES if m in tp.columns], index=[m for m in MODEL_NAMES if m in tp.columns].index(bundle["best_name"]))
    a, p = tp["actual"], tp[name]
    fig = px.scatter(x=a, y=p, opacity=0.45, labels={"x": "Actual price ($)", "y": "Predicted price ($)"})
    hi = float(max(a.max(), p.max()))
    fig.add_scatter(x=[0, hi], y=[0, hi], mode="lines", name="Perfect prediction", line=dict(dash="dash"))
    fig.update_layout(title=f"{name}: actual vs predicted (test set)")
    ui.show_fig(fig)

with tabs[2]:
    name = st.selectbox("Model", [m for m in MODEL_NAMES if m in tp.columns], index=[m for m in MODEL_NAMES if m in tp.columns].index(bundle["best_name"]), key="res_m")
    r = residuals(tp["actual"], tp[name])
    rs = residual_summary(r)
    ui.kpi_row([("Mean residual", f"{rs['mean_residual']:.2f}", "≈ 0 means unbiased"), ("Std of residuals", f"{rs['std_residual']:.2f}", ""),
                ("Within ±20 %", f"{rs['share_within_20pct']:.0%}", "of predictions"), ("Worst under-prediction", f"${rs['max_under_prediction']:.0f}", "")])
    a, b = st.columns(2)
    with a:
        ui.show_fig(px.scatter(r, x="predicted", y="residual", opacity=0.4, title="Residuals vs predicted").add_hline(y=0, line_dash="dash"))
    with b:
        ui.show_fig(px.histogram(r, x="residual", nbins=50, title="Residual distribution"))
    q = np.linspace(0.01, 0.99, 99)
    z = sps.norm.ppf(q); rq = r["residual"].quantile(q).to_numpy()
    qq = go.Figure(go.Scatter(x=z, y=rq, mode="markers", name="residuals"))
    qq.add_scatter(x=z, y=r["residual"].mean() + r["residual"].std() * z, mode="lines", name="Normal reference")
    qq.update_layout(title="Q–Q plot of residuals", xaxis_title="theoretical quantile", yaxis_title="residual")
    ui.show_fig(qq)
    st.caption("Look for: random scatter around 0 (no pattern), a funnel shape (error grows with price = heteroscedasticity), and heavy tails.")

with tabs[3]:
    imp = bundle["importance"].head(15)
    st.markdown(f"Best model: **{bundle['best_name']}** — measure: *{imp['kind'].iloc[0]}*")
    ui.show_fig(px.bar(imp.sort_values("value"), x="value", y="feature", orientation="h"))
    st.caption("For linear models these are absolute coefficients on scaled features (comparable only when features share a scale); "
               "for trees they are impurity-based importances. Neither proves causation.")

with tabs[4]:
    d, _ = ui.get_clean()
    col = st.selectbox("Numeric feature", [c for c in bundle["numeric"] if c in d.columns])
    x = d[[col]].astype(float)
    cmp = pd.DataFrame({"Original": x[col], "Min-Max [0,1]": MinMaxScaler().fit_transform(x).ravel(),
                        "Standardised (z-score)": StandardScaler().fit_transform(x).ravel()})
    ui.show_df(cmp.describe().T.round(3)[["mean", "std", "min", "max"]])
    ui.show_fig(px.histogram(cmp.melt(var_name="Scaling", value_name="value"), x="value", facet_col="Scaling", nbins=30).update_xaxes(matches=None))
    st.markdown("""
- **Min-Max:** x' = (x − min)/(max − min) → values in [0, 1]. **StandardScaler:** z = (x − μ)/σ → mean 0, std 1.
- **One-hot encoding:** each category becomes its own 0/1 column (e.g. `room_type_Private room`); unseen categories at prediction time are ignored safely.
- **Imputation:** median for numeric, most-frequent for categorical — fitted on training rows only.
- Tree models do not need scaling; Ridge/Lasso do, because their penalty depends on coefficient size.""")

with tabs[5]:
    f = bundle["simple_feature"]
    if f is None or f not in d.columns:
        st.info("No numeric feature available for simple regression.")
    else:
        row = res[res["Model"] == "Simple Linear Regression"].iloc[0]
        st.markdown(f"Simple linear regression uses **one** predictor — the numeric feature most correlated with price in the training data: **`{f}`**.")
        s = d.sample(min(len(d), 2000), random_state=0)
        k, b0 = np.polyfit(d[f], d["price"], 1)
        fig = px.scatter(s, x=f, y="price", opacity=0.35)
        xs = np.linspace(d[f].min(), d[f].max(), 50)
        fig.add_scatter(x=xs, y=b0 + k * xs, mode="lines", name="OLS line", line=dict(width=3))
        ui.show_fig(fig)
        st.write(f"Illustrative OLS fit on the cleaned data: price ≈ **{b0:.1f} + {k:.2f} × {f}**. "
                 f"Held-out performance of the trained model: RMSE ${row['RMSE']:.2f}, R² {row['R2']:.3f} "
                 f"(vs {res.iloc[0]['R2']:.3f} for the best multi-feature model).")
