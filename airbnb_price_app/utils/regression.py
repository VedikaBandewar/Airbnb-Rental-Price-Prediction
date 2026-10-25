"""Regression training, model comparison, persistence and prediction.

Leakage control
---------------
* Only a whitelist of listing attributes is used as features (never weekday_price / weekend_price, which are
  near-copies of the target).
* The train/test split happens BEFORE any imputation, scaling or encoding; all of those steps live inside the
  sklearn Pipeline, so they are fitted on training rows only (and re-fitted inside every CV fold).
* The best model is chosen by cross-validated RMSE on the TRAINING set. The test set is only used to report
  the final, unbiased performance.
"""
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .data_cleaning import basic_clean, iqr_bounds
from .data_io import is_numeric
from .evaluation import regression_metrics

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
NUMERIC_CANDIDATES = ["bedrooms", "beds", "accommodates", "bathrooms", "minimum_nights",
                      "number_of_reviews", "availability_365"]
CATEGORICAL_CANDIDATES = ["neighbourhood", "room_type"]
TARGET = "price"
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "none": None}
MODEL_NAMES = ["Simple Linear Regression", "Multiple Linear Regression", "Ridge Regression",
               "Lasso Regression", "Decision Tree Regression", "Random Forest Regression"]


def available_features(df: pd.DataFrame):
    """Features supported by the dataset (only columns that exist and vary)."""
    num = [c for c in NUMERIC_CANDIDATES if c in df.columns and is_numeric(df[c]) and df[c].nunique(dropna=True) > 1]
    cat = [c for c in CATEGORICAL_CANDIDATES if c in df.columns and 1 < df[c].nunique(dropna=True) <= 100]
    return num, cat


def prepare_X(df: pd.DataFrame, numeric, categorical) -> pd.DataFrame:
    X = df[list(numeric) + list(categorical)].copy()
    for c in numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    for c in categorical:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return X


def build_preprocessor(numeric, categorical, scaler: str = "standard") -> ColumnTransformer:
    parts = []
    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if SCALERS[scaler] is not None:
            steps.append(("scaler", SCALERS[scaler]()))
        parts.append(("num", Pipeline(steps), list(numeric)))
    if categorical:
        parts.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                      list(categorical)))
    return ColumnTransformer(parts)


def _wrap(estimator, log_target: bool):
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p if log_target else None,
                                      inverse_func=np.expm1 if log_target else None)


def model_specs(random_state: int = 42):
    """(estimator, hyper-parameter grid) for each of the six models."""
    return {
        "Simple Linear Regression": (LinearRegression(), {}),
        "Multiple Linear Regression": (LinearRegression(), {}),
        "Ridge Regression": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso Regression": (Lasso(max_iter=20000), {"alpha": [0.0001, 0.001, 0.01, 0.1]}),
        "Decision Tree Regression": (DecisionTreeRegressor(random_state=random_state),
                                     {"max_depth": [4, 6, 10, 14], "min_samples_leaf": [5, 20]}),
        "Random Forest Regression": (RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
                                     {"max_depth": [10, None], "min_samples_leaf": [2, 5]}),
    }


def pick_simple_feature(X_train: pd.DataFrame, y_train: pd.Series, numeric) -> str:
    """Single numeric predictor for simple linear regression = highest |Pearson r| on TRAINING data."""
    corr = {c: abs(X_train[c].corr(y_train)) for c in numeric if X_train[c].notna().sum() > 2}
    return max(corr, key=lambda k: (0 if np.isnan(corr[k]) else corr[k]))


def compare_models(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, scaler: str = "standard",
                   log_target: bool = True, cv: int = 5, trim_train_outliers: bool = True, models=None) -> dict:
    data = basic_clean(df)
    numeric, categorical = available_features(data)
    if not numeric and not categorical:
        raise ValueError("The dataset has no usable feature columns for regression.")
    if len(data) < 50:
        raise ValueError("At least 50 valid rows are required to train regression models.")
    X, y = prepare_X(data, numeric, categorical), data[TARGET].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=random_state)
    n_trim = n_trim_test = 0
    X_te_all, y_te_all = X_te, y_te
    if trim_train_outliers:
        # Extreme-price fences (Q3 + 3*IQR) are computed from TRAIN targets only (no leakage) and then applied
        # to train rows (cleaner fitting) and test rows (evaluation on typical listings). Metrics on ALL test
        # rows are also reported ("RMSE (all rows)") so nothing is hidden.
        lo, hi = iqr_bounds(y_tr, 3.0)
        keep = y_tr.between(lo, hi)
        n_trim = int((~keep).sum())
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        keep_te = y_te.between(lo, hi)
        n_trim_test = int((~keep_te).sum())
        X_te, y_te = X_te[keep_te], y_te[keep_te]
    simple_feat = pick_simple_feature(X_tr, y_tr, numeric) if numeric else None

    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows, fitted, preds, params = [], {}, {"actual": y_te.to_numpy()}, {}
    for name, (est, grid) in model_specs(random_state).items():
        if models is not None and name not in models:
            continue
        if name == "Simple Linear Regression":
            if simple_feat is None:
                continue
            cols = [simple_feat]
            prep = build_preprocessor(cols, [], scaler)
        else:
            cols = numeric + categorical
            prep = build_preprocessor(numeric, categorical, scaler)
        pipe = Pipeline([("prep", prep), ("model", _wrap(est, log_target))])
        gs = GridSearchCV(pipe, {f"model__regressor__{k}": v for k, v in grid.items()}, cv=kf,
                          scoring="neg_root_mean_squared_error", refit=True)
        gs.fit(X_tr[cols], y_tr)
        yhat = gs.predict(X_te[cols])
        m = regression_metrics(y_te, yhat)
        rmse_all = regression_metrics(y_te_all, gs.predict(X_te_all[cols]))["RMSE"]
        cv_rmse = -gs.best_score_
        cv_std = gs.cv_results_["std_test_score"][gs.best_index_]
        rows.append({"Model": name, **m, "RMSE (all rows)": rmse_all, "CV RMSE": cv_rmse, "CV RMSE std": cv_std,
                     "Best params": ", ".join(f"{k.split('__')[-1]}={v}" for k, v in gs.best_params_.items()) or "—"})
        fitted[name] = {"pipeline": gs.best_estimator_, "columns": cols}
        preds[name] = yhat
        params[name] = gs.best_params_
    res = pd.DataFrame(rows).sort_values("CV RMSE").reset_index(drop=True)
    return {"results": res, "models": fitted, "best_name": res.loc[0, "Model"], "numeric": numeric,
            "categorical": categorical, "simple_feature": simple_feat, "X_train": X_tr, "X_test": X_te,
            "y_train": y_tr, "y_test": y_te, "predictions": pd.DataFrame(preds, index=y_te.index),
            "n_trimmed": n_trim, "n_trimmed_test": n_trim_test, "n_rows": len(data),
            "config": {"test_size": test_size, "random_state": random_state, "scaler": scaler,
                       "log_target": log_target, "cv": cv, "trim_train_outliers": trim_train_outliers}}


def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"].regressor_
    names = list(prep.get_feature_names_out())
    if hasattr(model, "feature_importances_"):
        vals, kind = model.feature_importances_, "importance"
    else:
        vals, kind = np.abs(model.coef_), "|coefficient|"
    df = pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in names], "value": vals})
    df["kind"] = kind
    return df.sort_values("value", ascending=False).reset_index(drop=True)


def train_and_save(df: pd.DataFrame, model_dir: Path | str = MODEL_DIR, **kwargs) -> dict:
    """Train all models, pick the best by CV RMSE, and persist everything with Joblib."""
    out = compare_models(df, **kwargs)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best = out["models"][out["best_name"]]
    data = basic_clean(df)
    num, cat = out["numeric"], out["categorical"]
    X_all = prepare_X(data, num, cat)
    bundle = {
        "model": best["pipeline"], "model_columns": best["columns"], "best_name": out["best_name"],
        "numeric": num, "categorical": cat, "simple_feature": out["simple_feature"],
        "categories": {c: sorted(map(str, data[c].dropna().unique())) for c in cat},
        "ranges": {c: (float(X_all[c].min()), float(X_all[c].max()), float(X_all[c].median())) for c in num},
        "results": out["results"], "importance": feature_importance(best["pipeline"]),
        "test_predictions": out["predictions"].reset_index(drop=True), "config": out["config"],
        "n_rows": out["n_rows"], "n_trimmed": out["n_trimmed"], "n_trimmed_test": out["n_trimmed_test"], "trained_at": datetime.now().isoformat(timespec="seconds"),
        "price_median": float(data[TARGET].median()),
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")
    joblib.dump(best["pipeline"], model_dir / "best_model_pipeline.joblib")
    joblib.dump(best["pipeline"].named_steps["prep"], model_dir / "preprocessor.joblib")
    out["results"].to_csv(model_dir / "model_comparison.csv", index=False)
    out["bundle"] = bundle
    return out


def load_bundle(model_dir: Path | str = MODEL_DIR) -> dict | None:
    p = Path(model_dir) / "model_bundle.joblib"
    return joblib.load(p) if p.exists() else None


def validate_inputs(bundle: dict, inputs: dict) -> list:
    """Return a list of human-readable problems (empty list = valid)."""
    errs = []
    for c in bundle["numeric"]:
        if c not in inputs or inputs[c] is None:
            errs.append(f"'{c}' is required.")
            continue
        try:
            v = float(inputs[c])
        except (TypeError, ValueError):
            errs.append(f"'{c}' must be a number.")
            continue
        if not np.isfinite(v) or v < 0:
            errs.append(f"'{c}' must be a non-negative number.")
        elif c == "availability_365" and v > 365:
            errs.append("'availability_365' cannot exceed 365 days.")
        elif c in ("accommodates", "minimum_nights") and v < 1:
            errs.append(f"'{c}' must be at least 1.")
    for c in bundle["categorical"]:
        if inputs.get(c) not in bundle["categories"][c]:
            errs.append(f"'{c}' must be one of: {', '.join(bundle['categories'][c])}.")
    return errs


def predict_price(bundle: dict, inputs: dict) -> float:
    """Apply the SAME fitted preprocessing + model that was used in training."""
    errs = validate_inputs(bundle, inputs)
    if errs:
        raise ValueError(" ".join(errs))
    row = pd.DataFrame([inputs])
    X = prepare_X(row, bundle["numeric"], bundle["categorical"])[bundle["model_columns"]]
    return float(bundle["model"].predict(X)[0])
