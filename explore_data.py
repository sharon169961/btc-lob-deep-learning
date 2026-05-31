import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

DATA_FILE = "data/lob_raw.csv"

def load_and_validate(path: str)-> pd.DataFrame:
    print(f"Loading {path}...")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"\n{'─'*50}")
    print(f"  Rows         : {len(df):,}")
    print(f"  Columns      : {len(df.columns)}")
    print(f"  Time range   : {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    duration = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds()
    print(f"  Duration     : {duration:.0f}s ({duration/60:.1f} min)")
    print(f"  Avg interval : {duration / len(df):.3f}s")
    print(f"  Null count   : {df.isnull().sum().sum()}")
    print(f"{'─'*50}")
    return df

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df["mid_price"] = (df["bid_p1"] + df["ask_p1"]) /2
    df["spread"] = df["ask_p1"]-df["bid_p1"]
    df["spread_bps"] = (df["spread"]/ df["mid_price"]) * 10_000

    df["imbalance_l1"]=(
        (df["bid_v1"] - df["ask_v1"]) /
        (df["bid_v1"] + df["ask_v1"])
    )

    bid_vol_cols = [f"bid_v{i}" for i in range(1,11)]
    ask_vol_cols = [f"ask_v{i}" for i in range(1,11)]
    df["total_bid_vol"] = df[bid_vol_cols].sum(axis=1)
    df["total_ask_vol"]=df[ask_vol_cols].sum(axis=1)

    df["imbalance_all"] = (
        (df["total_bid_vol"] - df["total_ask_vol"]) /
        (df["total_bid_vol"] + df["total_ask_vol"])
    )

    df["mid_return"] = df["mid_price"].pct_change()
    return df

def print_snapshot(df: pd.DataFrame, idx: int = 0):
    row = df.iloc[idx]
    print(f"\nSnapshot at t={row['timestamp']}  (row {idx})")
    print(f"  {'LEVEL':<8} {'ASK PRICE':>12} {'ASK VOL':>10}")
    for i in range(10, 0, -1):
        print(f"  L{i:<7} {row[f'ask_p{i}']:>12,.2f} {row[f'ask_v{i}']:>10.4f}")
    print(f"  {'─'*34}  ← SPREAD = {row['spread']:.2f}")
    for i in range(1, 11):
        print(f"  L{i:<7} {row[f'bid_p{i}']:>12,.2f} {row[f'bid_v{i}']:>10.4f}")
    print(f"\n  Mid-price : ${row['mid_price']:,.4f}")
    print(f"  Spread    : ${row['spread']:.4f} ({row['spread_bps']:.3f} bps)")
    print(f"  Imbalance : {row['imbalance_l1']:+.4f}")

def plot_overview(df: pd.DataFrame, save_path: str = "data/exploration.png"):
    fig = plt.figure(figsize=(14,10))
    fig.suptitle("LOB Data - Exploration Overview", fontsize=15, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2,2,hspace=0.38,wspace=0.32)
    t=df["timestamp"]

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, df["mid_price"], linewidth=0.8, color="#2563EB")
    ax1.set_title("Mid-Price Over Time")
    ax1.set_ylabel("Price (USDT)")
    ax1.tick_params(axis="x", rotation=30, labelsize=7)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["spread_bps"], linewidth=0.6, color="#DC2626", alpha=0.8)
    ax2.set_title("Bid-Ask Spread (basis points)")
    ax2.set_ylabel("Spread (bps)")
    ax2.tick_params(axis="x", rotation=30, labelsize=7)
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(t, df["imbalance_all"], linewidth=0.6, color="#059669", alpha=0.85)
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax3.fill_between(t, 0, df["imbalance_all"],
                     where=df["imbalance_all"] > 0, alpha=0.15, color="#059669")
    ax3.fill_between(t, 0, df["imbalance_all"],
                     where=df["imbalance_all"] < 0, alpha=0.15, color="#DC2626")
    ax3.set_title("Order Imbalance (all 10 levels)")
    ax3.set_ylabel("Imbalance")
    ax3.tick_params(axis="x", rotation=30, labelsize=7)
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    returns = df["mid_return"].dropna()
    ax4.hist(returns, bins=80, color="#7C3AED", alpha=0.75, edgecolor="none")
    ax4.axvline(0, color="black", linewidth=1)
    ax4.set_title("Distribution of Tick-to-Tick Returns")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Count")
    ax4.grid(alpha=0.3)

    kurt = returns.kurtosis()
    ax4.text(0.97, 0.95, f"Kurtosis: {kurt:.1f}", transform=ax4.transAxes,
             ha="right", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Plot saved → {save_path}")
    plt.show()


def print_stats(df: pd.DataFrame):
    cols = ["mid_price", "spread_bps", "imbalance_l1", "imbalance_all", "mid_return"]
    print("\nKey Statistics:")
    print(df[cols].describe().round(6).to_string())

    print("\nThings to notice:")
    print(f"  • Avg spread  : {df['spread_bps'].mean():.4f} bps")
    print(f"  • Mean imbal  : {df['imbalance_all'].mean():+.4f}")
    print(f"  • Return kurt : {df['mid_return'].dropna().kurtosis():.2f}")
    pct_up   = (df["mid_return"] > 0).mean() * 100
    pct_down = (df["mid_return"] < 0).mean() * 100
    pct_flat = (df["mid_return"] == 0).mean() * 100
    print(f"  • Up ticks    : {pct_up:.1f}%")
    print(f"  • Down ticks  : {pct_down:.1f}%")
    print(f"  • Flat ticks  : {pct_flat:.1f}%")

if __name__ == "__main__":
    df = load_and_validate(DATA_FILE)
    df=add_derived(df)
    print_snapshot(df,idx=0)
    print_stats(df)
    plot_overview(df)
    enriched_path= "data/lob_enriched.csv"
    df.to_csv(enriched_path,index=False)
    print(f"\n✓ Enriched data saved → {enriched_path}")
    





