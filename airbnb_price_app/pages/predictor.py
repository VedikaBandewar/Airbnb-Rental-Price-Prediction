import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from utils import ui
from utils.regression import predict_price, validate_inputs

ui.page_header("Airbnb Price Predictor", "Enter listing details to get a model-estimated nightly price.")
bundle = ui.get_bundle()
if bundle is None:
    st.warning("No trained model found. Go to **Regression Model Comparison** and train the models (or run `python train_model.py`).")
    st.stop()

LABELS = {"bedrooms": "Bedrooms", "beds": "Beds", "accommodates": "Guests accommodated", "bathrooms": "Bathrooms",
          "minimum_nights": "Minimum nights", "number_of_reviews": "Number of reviews", "availability_365": "Days available per year",
          "neighbourhood": "Neighbourhood", "room_type": "Room type"}
STEP = {"bathrooms": 0.5}
ui.note(f"Model in use: <b>{bundle['best_name']}</b> · trained on {bundle['n_rows']:,} listings · only inputs supported by the training dataset are shown.")
st.session_state.setdefault("scenarios", [])
tab_pred, tab_perf = st.tabs(["Predict", "Model performance"])

with tab_pred:
    with st.form("predict_form"):
        inputs, cols = {}, st.columns(3)
        i = 0
        for c in bundle["categorical"]:
            inputs[c] = cols[i % 3].selectbox(LABELS.get(c, c), bundle["categories"][c]); i += 1
        for c in bundle["numeric"]:
            lo, hi, med = bundle["ranges"][c]
            step = STEP.get(c, 1.0)
            hi_ui = 365.0 if c == "availability_365" else max(hi, med) * 1.5
            lo_ui = 1.0 if c in ("accommodates", "minimum_nights") else 0.0
            inputs[c] = cols[i % 3].number_input(LABELS.get(c, c), min_value=lo_ui, max_value=float(hi_ui), value=float(np.clip(round(med / step) * step, lo_ui, hi_ui)),
                                                 step=step, help=f"Training data range: {lo:g} – {hi:g}"); i += 1
        b1, b2 = st.columns(2)
        go_pred = b1.form_submit_button("💰 Predict price", type="primary")
        add_cmp = b2.form_submit_button("➕ Add to comparison")
    if go_pred or add_cmp:
        errs = validate_inputs(bundle, inputs)
        if errs:
            for e in errs:
                st.error(e)
        else:
            price = predict_price(bundle, inputs)
            st.session_state["last"] = (dict(inputs), price)
            if add_cmp:
                st.session_state["scenarios"].append({"Scenario": f"#{len(st.session_state['scenarios']) + 1}", **inputs, "Estimated price ($)": round(price, 2)})
    if "last" in st.session_state:
        vals, price = st.session_state["last"]
        mae = float(bundle["results"].iloc[0]["MAE"])
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f'<div class="price-box"><div class="caption">Estimated nightly price</div><div class="amount">${price:,.0f}</div>'
                        f'<div class="caption">typical error about ± ${mae:,.0f} (model MAE on unseen listings)</div></div>', unsafe_allow_html=True)
            vs = price / bundle["price_median"] - 1
            st.caption(f"{abs(vs):.0%} {'above' if vs >= 0 else 'below'} the dataset's median price (${bundle['price_median']:,.0f}).")
        with right:
            nums = [c for c in bundle["numeric"] if c not in ("availability_365",)] or bundle["numeric"]
            var = st.selectbox("What-if: vary one input", bundle["numeric"], format_func=lambda c: LABELS.get(c, c))
            lo, hi, _ = bundle["ranges"][var]
            grid = np.unique(np.linspace(max(lo, 1 if var in ("accommodates", "minimum_nights") else lo), hi, 12).round(1))
            sens = pd.DataFrame({var: grid, "price": [predict_price(bundle, {**vals, var: g}) for g in grid]})
            ui.show_fig(px.line(sens, x=var, y="price", markers=True, labels={"price": "Estimated price ($)", var: LABELS.get(var, var)}))
        st.warning("⚠️ This is a **statistical model estimate** based on the training dataset — **not an actual Airbnb quote**. "
                   "Real prices depend on season, amenities, photos, host pricing and many factors not in the data.")
    if st.session_state["scenarios"]:
        st.subheader("Scenario comparison")
        sc = pd.DataFrame(st.session_state["scenarios"])
        ui.show_df(sc, hide_index=True)
        ui.show_fig(px.bar(sc, x="Scenario", y="Estimated price ($)", text="Estimated price ($)"))
        c1, c2 = st.columns(2)
        with c1:
            ui.download_csv(sc, "⬇️ Download scenarios", "scenarios.csv", key="sc_dl")
        if c2.button("Clear scenarios"):
            st.session_state["scenarios"] = []
            st.rerun()

with tab_perf:
    res = bundle["results"]
    best = res.iloc[0]
    ui.kpi_row([("MAE", f"${best['MAE']:.2f}", "average absolute error"), ("RMSE", f"${best['RMSE']:.2f}", "penalises large errors"),
                ("MSE", f"{best['MSE']:,.0f}", "squared dollars"), ("R²", f"{best['R2']:.3f}", "variance explained")])
    st.write("")
    ui.show_df(res[["Model", "MAE", "RMSE", "R2", "CV RMSE"]].round(3), hide_index=True)
    imp = bundle["importance"].head(12)
    ui.show_fig(px.bar(imp.sort_values("value"), x="value", y="feature", orientation="h", title="Most influential features"))
