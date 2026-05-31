import requests
import pandas as pd
import time
import os
from datetime import datetime

SYMBOL       = "BTCUSDT"
DEPTH_LEVELS = 10
INTERVAL     = 0.5
DURATION     = 28800
OUTPUT_FILE  = "data/lob_raw.csv"

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth"


def make_columns(levels: int) -> list[str]:
    cols =["timestamp"]
    for i in range(1,levels+1):
        cols += [f"bid_p{i}", f"bid_v{i}"]
    for i in range(1,levels+1):
        cols += [f"ask_p{i}", f"ask_v{i}"]

    return cols

def fetch_snapshot(symbol: str, levels: int) -> dict | None:
    try:
        response = requests.get(
            BINANCE_DEPTH_URL,
            params={"symbol": symbol, "limit": levels},
            timeout = 5
        )

        response.raise_for_status()
        data = response.json()


    except Exception as e:
        print(f" [!] Fetch failed: {e}")
        return None
    

    row = {"timestamp": datetime.utcnow().isoformat()}

    for i, (price,volume) in enumerate(data["bids"][:levels], start=1):
        row[f"bid_p{i}"] = float(price)
        row[f"bid_v{i}"]=float(volume)

    for i, (price, volume) in enumerate(data["asks"][:levels], start=1):
        row[f"ask_p{i}"] = float(price)
        row[f"ask_v{i}"] = float(volume)

    return row


def collect(symbol,levels, interval, duration, output):
    os.makedirs(os.path.dirname(output), exist_ok = True)
    columns = make_columns(levels)
    snapshots = []
    total_ticks = int(duration/interval)
    start = time.time()

    for tick in range(total_ticks):
        row = fetch_snapshot(symbol, levels)

        if row:
            snapshots.append(row)
            if tick % 20 == 0:
              mid = (row["bid_p1"] + row["ask_p1"]) / 2
              spread = row["ask_p1"] - row["bid_p1"]
              elapsed = time.time() - start
              print(f"  tick {tick:>5} | mid=${mid:,.2f} | spread=${spread:.2f} | elapsed={elapsed:.0f}s")

        
        elapsed_tick = time.time() - (start + tick * interval)
        sleep_time = max(0, interval - elapsed_tick)
        time.sleep(sleep_time)

    df = pd.DataFrame(snapshots, columns=columns)
    df.to_csv(output, index = False)
    print(f"\n✓ Saved {len(df)} snapshots → {output}")
    return df

        
    

if __name__ == "__main__":
    df = collect(
        symbol = SYMBOL,
        levels = DEPTH_LEVELS,
        interval = INTERVAL,
        duration = DURATION,
        output = OUTPUT_FILE,
    )
    print(df.head(3).to_string())


        

