"""Train all regression models on the bundled dataset and save them to models/.
Usage:  python train_model.py [--csv path/to/listings.csv] [--scaler standard|minmax|none] [--no-log]
"""
import argparse
from utils.data_io import DATA_PATH, load_csv, validate_dataset
from utils.regression import train_and_save

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DATA_PATH))
    ap.add_argument("--scaler", default="standard", choices=["standard", "minmax", "none"])
    ap.add_argument("--no-log", action="store_true", help="train on raw price instead of log(price)")
    a = ap.parse_args()
    df = load_csv(a.csv)
    ok, msgs = validate_dataset(df)
    if not ok:
        raise SystemExit("Dataset problem: " + " ".join(msgs))
    out = train_and_save(df, scaler=a.scaler, log_target=not a.no_log)
    print(out["results"][["Model", "MAE", "RMSE", "R2", "RMSE (all rows)", "CV RMSE"]].round(3).to_string(index=False))
    print(f"\nBest model (lowest CV RMSE): {out['best_name']}  -> saved to models/")
