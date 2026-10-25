"""Generate a SYNTHETIC Airbnb-style dataset (Inside Airbnb schema) for offline use.

The data is simulated, NOT real Airbnb data. It deliberately contains realistic
data-quality problems (missing values, duplicates, invalid prices, price outliers)
so the cleaning module has something to do.
Run:  python scripts/generate_dataset.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def generate(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    neigh = {"Downtown": 0.35, "Harbour View": 0.30, "Old Town": 0.18, "Midtown": 0.12,
             "University District": 0.0, "Airport Area": -0.15, "Suburb North": -0.20,
             "Suburb South": -0.28}
    neigh_p = np.array([.14, .10, .13, .15, .12, .06, .15, .15])
    rooms = {"Entire home/apt": 0.0, "Private room": -0.55, "Shared room": -0.95, "Hotel room": 0.10}
    room_p = [.58, .34, .03, .05]

    nb = rng.choice(list(neigh), n, p=neigh_p / neigh_p.sum())
    rt = rng.choice(list(rooms), n, p=room_p)
    acc = np.empty(n, dtype=int)
    for i, r in enumerate(rt):
        if r == "Entire home/apt":
            acc[i] = rng.choice(np.arange(1, 9), p=[.04, .27, .14, .22, .08, .13, .04, .08])
        elif r == "Private room":
            acc[i] = rng.choice([1, 2, 3], p=[.45, .45, .10])
        elif r == "Shared room":
            acc[i] = rng.choice([1, 2], p=[.7, .3])
        else:
            acc[i] = rng.choice([2, 3, 4], p=[.6, .25, .15])
    bedrooms = np.where(rt == "Entire home/apt", np.clip(np.ceil(acc / 2.2) - rng.integers(0, 2, n), 0, 5), 1).astype(int)
    beds = np.maximum(1, np.ceil(acc / 2) + rng.integers(0, 2, n)).astype(int)
    bathrooms = np.clip(np.round((bedrooms * 0.6 + 0.7) * 2) / 2 + rng.choice([0, 0, 0, .5, -.5], n), 0.5, 4)
    min_nights = rng.choice([1, 2, 3, 5, 7, 14, 30, 60], n, p=[.30, .25, .15, .08, .08, .04, .08, .02])
    reviews = rng.negative_binomial(1.2, 0.02, n)
    avail = np.where(rng.random(n) < 0.12, 0, rng.integers(1, 366, n))

    log_price = (3.70 + 0.10 * acc + 0.14 * bedrooms + 0.08 * bathrooms
                 + np.array([rooms[r] for r in rt]) + np.array([neigh[x] for x in nb])
                 - 0.04 * np.log1p(min_nights) - 0.0004 * avail - 0.0006 * np.minimum(reviews, 150)
                 + rng.normal(0, 0.28, n))
    price = np.maximum(10, np.round(np.exp(log_price))).astype(float)
    weekday = np.round(price * (1 + rng.normal(0, 0.04, n)), 2)          # paired observations
    weekend = np.round(weekday * (1.08 + rng.normal(0, 0.05, n)), 2)

    df = pd.DataFrame({
        "id": np.arange(100001, 100001 + n),
        "name": [f"{r} in {x}" for r, x in zip(rt, nb)],
        "neighbourhood": nb, "room_type": rt, "bedrooms": bedrooms.astype(float), "beds": beds.astype(float),
        "accommodates": acc, "bathrooms": bathrooms, "minimum_nights": min_nights,
        "number_of_reviews": reviews.astype(float), "availability_365": avail,
        "weekday_price": weekday, "weekend_price": weekend, "price": price})

    # --- inject realistic data-quality problems ---
    for col, frac in [("bedrooms", .03), ("beds", .03), ("bathrooms", .04), ("number_of_reviews", .02)]:
        df.loc[rng.random(n) < frac, col] = np.nan
    out = rng.random(n) < 0.01
    df.loc[out, "price"] = (df.loc[out, "price"] * rng.uniform(4, 8, out.sum())).round()
    df.loc[rng.choice(n, 5, replace=False), "price"] = 0                  # invalid prices
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)  # exact duplicates
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    (ROOT / "data").mkdir(exist_ok=True)
    full = generate()
    full.to_csv(ROOT / "data" / "airbnb_listings.csv", index=False)
    full.drop_duplicates().query("price > 0").dropna().head(60).to_csv(ROOT / "data" / "sample_listings.csv", index=False)
    print("Wrote", len(full), "rows to data/airbnb_listings.csv and a 60-row data/sample_listings.csv")
