"""Regression evaluation metrics and residual helpers."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred)}


def residuals(y_true, y_pred) -> pd.DataFrame:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return pd.DataFrame({"actual": y_true, "predicted": y_pred, "residual": y_true - y_pred})


def residual_summary(res: pd.DataFrame) -> dict:
    r = res["residual"]
    return {"mean_residual": r.mean(), "std_residual": r.std(ddof=1), "max_over_prediction": r.min(),
            "max_under_prediction": r.max(), "share_within_20pct": float((r.abs() / res["actual"] <= 0.2).mean())}
