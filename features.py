import pandas as pd
import numpy as np

def price_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_price"]=(df["bid_p1"] + df["ask_p1"])/2
    df["spread"] = df["ask_p1"]-df["bid_p1"]
    df["spread_bps"]=(df["spread"] / df["mid_price"]) * 10_000
    return df


def imbalance_features(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:

    df["imbalance_l1"]=(
        (df["bid_v1"]-df["ask_v1"])/
        (df["bid_v1"] + df["ask_v1"])
    )

    weights = np.array([1/i for i in range(1,levels+1)])
    weights = weights = weights/weights.sum()

    bid_weighted = sum(
        weights[i-1] * df[f"bid_v{i}"] for i in range(1,levels+1)

    )

    ask_weighted= sum(
        weights[i-1] * df[f"ask_v{i}"] for i in range(1,levels+1)
    )

    df["imbalance_weighted"]=(
        (bid_weighted - ask_weighted) /
        (bid_weighted + ask_weighted)
    )

    return df

def depth_features(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    df["total_bid_depth"]= sum(
        df[f"bid_v{i}"] for i in range(1,levels+1)
    )

    df["total_ask_depth"]=sum(
        df[f"ask_v{i}"] for i in range(1,levels+1)
    )

    df["depth_ratio"]=(
        df["total_bid_depth"] /
        (df["total_bid_depth"] + df["total_ask_depth"]) 
    )

    df["bid_depth_l1_ratio"] = df["bid_v1"] / df["total_bid_depth"]
    df["ask_depth_l1_ratio"] = df["ask_v1"] / df["total_ask_depth"]

    return df

def price_level_features(df: pd.DataFrame, levels: int = 10)-> pd.DataFrame:
    for i in range(1,levels+1):
        df[f"bid_dist_{i}"] = (df["mid_price"] - df[f"bid_p{i}"]) / df["mid_price"]
        df[f"ask_dist_{i}"] = (df[f"ask_p{i}"] - df["mid_price"]) / df["mid_price"]

    return df
    

def rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_return"] = df["mid_price"].pct_change()

    for window in [5,10,20]:
        df[f"return_mean{window}"] = (
            df["mid_return"].rolling(window).mean()
        )

        df[f"return_std_{window}"]=(
            df["mid_return"].rolling(window).std()
        )

        df[f"imbalance_mean_{window}"] = (
            df["imbalance_l1"].rolling(window).mean()
        )

    return df


def make_labels(df: pd.DataFrame, horizon: int = 10, threshold: float= 0.0002) -> pd.DataFrame:
    future_mid = df["mid_price"].rolling(horizon).mean().shift(-horizon)
    past_mid = df["mid_price"].rolling(horizon).mean()

    smooth_return = (future_mid - past_mid) / past_mid

    conditions = [
        smooth_return > threshold,
        smooth_return < -threshold

    ]
    choices = [1,2]

    df["label"]=np.select(conditions, choices, default = 0)
    df["smooth_return"] = smooth_return

    return df

def build_features(df: pd.DataFrame, horizon: int = 10, threshold: float = 0.0002) -> pd.DataFrame:
    df = price_features(df)
    df = imbalance_features(df)
    df = depth_features(df)
    df = price_level_features(df)
    df = rolling_features(df)
    df = make_labels(df, horizon, threshold=threshold)

    df = df.dropna().reset_index(drop=True)

    return df


if __name__ == "__main__":

    df = pd.read_csv("data/lob_raw.csv", parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Raw rows: {len(df)}")

    df = build_features(df, horizon=100, threshold=0.00005971)

    print(f"Rows after feature engineering + label creation: {len(df)}")

    label_counts = df["label"].value_counts().sort_index()
    total = len(df)
    print("\nLabel distribution:")
    print(f"  0 (FLAT) : {label_counts.get(0,0):>6} ({label_counts.get(0,0)/total*100:.1f}%)")
    print(f"  1 (UP)   : {label_counts.get(1,0):>6} ({label_counts.get(1,0)/total*100:.1f}%)")
    print(f"  2 (DOWN) : {label_counts.get(2,0):>6} ({label_counts.get(2,0)/total*100:.1f}%)")

    print(f"\nFeature columns: {len(df.columns) - 1}")
    print(df.head(3).to_string())

    df.to_csv("data/lob_features.csv", index=False)
    print("\n✓ Saved → data/lob_features.csv")

    print("\nSmooth return percentiles:")
    sr = df["smooth_return"].abs()
    for p in [50, 75, 90, 95, 99]:
        print(f"  {p}th percentile: {sr.quantile(p/100):.8f}")





    








    



