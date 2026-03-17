"""
Fetch 1H XAUUSDT candles from AsterDex and update new_candles_2026.csv
Uses the klines endpoint to get historical data.
"""
import requests
import pandas as pd
from datetime import datetime, timezone

ASTERDEX_BASE = "https://fapi.asterdex.com"
SYMBOL = "XAUUSDT"
INTERVAL = "1h"
CSV_PATH = "new_candles_2026.csv"

def fetch_klines(start_ms, end_ms=None, limit=1500):
    """Fetch klines from AsterDex"""
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": int(start_ms),
        "limit": limit,
    }
    if end_ms:
        params["endTime"] = int(end_ms)
    
    url = f"{ASTERDEX_BASE}/fapi/v1/klines"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    
    rows = []
    for k in data:
        ts = datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc)
        rows.append({
            "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
        })
    return pd.DataFrame(rows)

def main():
    # Load existing CSV
    try:
        existing = pd.read_csv(CSV_PATH)
        existing['datetime'] = pd.to_datetime(existing['datetime'])
        print(f"Existing CSV: {len(existing)} rows")
        print(f"  First: {existing['datetime'].iloc[0]}")
        print(f"  Last:  {existing['datetime'].iloc[-1]}")
        last_ts = existing['datetime'].iloc[-1]
    except FileNotFoundError:
        existing = pd.DataFrame()
        last_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        print("No existing CSV found, starting from 2026-01-01")

    # Fetch all 2026 data from AsterDex
    start_2026 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    start_ms = int(start_2026.timestamp() * 1000)
    
    all_klines = []
    print(f"\nFetching klines from AsterDex ({SYMBOL} {INTERVAL})...")
    
    while True:
        df_chunk = fetch_klines(start_ms)
        if df_chunk.empty:
            break
        all_klines.append(df_chunk)
        print(f"  Fetched {len(df_chunk)} candles: {df_chunk['datetime'].iloc[0]} -> {df_chunk['datetime'].iloc[-1]}")
        
        # Move start to after last candle
        last_dt = pd.to_datetime(df_chunk['datetime'].iloc[-1])
        new_start_ms = int(last_dt.timestamp() * 1000) + 3600000  # +1h
        
        if new_start_ms <= start_ms or len(df_chunk) < 100:
            break
        start_ms = new_start_ms
    
    if not all_klines:
        print("No data fetched!")
        return
    
    fresh = pd.concat(all_klines, ignore_index=True)
    fresh['datetime'] = pd.to_datetime(fresh['datetime'])
    fresh = fresh.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    
    print(f"\nTotal fresh data: {len(fresh)} candles")
    print(f"  First: {fresh['datetime'].iloc[0]}")
    print(f"  Last:  {fresh['datetime'].iloc[-1]}")
    
    # Compare a few rows with existing data to check consistency
    if not existing.empty:
        merged = pd.merge(existing, fresh, on='datetime', suffixes=('_old', '_new'))
        if len(merged) > 0:
            merged['close_diff'] = abs(merged['close_old'] - merged['close_new'])
            avg_diff = merged['close_diff'].mean()
            max_diff = merged['close_diff'].max()
            print(f"\nPrice comparison (existing vs fresh):")
            print(f"  Overlapping candles: {len(merged)}")
            print(f"  Avg close diff: ${avg_diff:.4f}")
            print(f"  Max close diff: ${max_diff:.4f}")
            if max_diff > 5.0:
                print(f"  WARNING: Max diff > $5 - data sources may differ!")
            else:
                print(f"  OK: Data is consistent (max diff < $5)")
    
    # Save updated CSV
    fresh['datetime'] = fresh['datetime'].dt.strftime("%Y-%m-%d %H:%M:%S")
    fresh.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {len(fresh)} candles to {CSV_PATH}")
    print(f"  Last candle: {fresh['datetime'].iloc[-1]}")

if __name__ == "__main__":
    main()
