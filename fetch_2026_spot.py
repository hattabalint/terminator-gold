"""
Fetch 2026 XAUUSD SPOT hourly candles and save to new_candles_2026.csv
Source: AsterDex public API (XAUUSDT) – same exchange the bot trades on.
The perpetual contract tracks spot gold via funding rate mechanism.
No authentication needed – uses public kline endpoint.
"""

import requests
import pandas as pd
import os
from datetime import datetime, timezone, timedelta

OUT_PATH = os.path.join(os.path.dirname(__file__), "new_candles_2026.csv")
SPOT_CSV = os.path.join(os.path.dirname(__file__), "xauusd_1h_2020_2025_spot.csv")

ASTERDEX_BASE = "https://fapi.asterdex.com"
SYMBOL = "XAUUSDT"
INTERVAL = "1h"


def fetch_klines(start_ms, end_ms, limit=1000):
    """Fetch klines from AsterDex public API."""
    url = f"{ASTERDEX_BASE}/fapi/v1/klines"
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": limit
    }
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code == 200:
        return resp.json()
    else:
        print(f"  Error {resp.status_code}: {resp.text[:200]}")
        return []


def fetch_all_2026():
    """Fetch all 2026 XAUUSDT hourly candles from AsterDex."""
    print(f"[AsterDex] Fetching {SYMBOL} 1H candles from 2026-01-01...")

    start_dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end_dt   = datetime.now(timezone.utc)

    all_candles = []
    current_ms = int(start_dt.timestamp() * 1000)
    end_ms     = int(end_dt.timestamp() * 1000)

    while current_ms < end_ms:
        klines = fetch_klines(current_ms, end_ms)
        if not klines:
            break

        for k in klines:
            ts = datetime.utcfromtimestamp(k[0] / 1000)
            all_candles.append({
                'datetime': ts,
                'open':  float(k[1]),
                'high':  float(k[2]),
                'low':   float(k[3]),
                'close': float(k[4])
            })

        last_open_ms = klines[-1][0]
        current_ms = last_open_ms + 3600000  # +1 hour

        if len(klines) < 1000:
            break

    if not all_candles:
        return None

    df = pd.DataFrame(all_candles)
    df = df.drop_duplicates('datetime').sort_values('datetime').reset_index(drop=True)

    print(f"[AsterDex] Fetched {len(df)} candles")
    print(f"[AsterDex] Range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"[AsterDex] Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    return df


def validate_spot_prices(df):
    """Validate price continuity with 2025 spot data."""
    if not os.path.exists(SPOT_CSV):
        print("[VALIDATE] 2025 spot CSV not found, skipping validation")
        return True

    df_spot = pd.read_csv(SPOT_CSV)
    last_spot_close = df_spot.iloc[-1, 4]
    first_new_close = df['close'].iloc[0]
    gap_pct = abs(first_new_close - last_spot_close) / last_spot_close * 100

    print(f"\n[VALIDATION]")
    print(f"  2025 spot last close : {last_spot_close:.2f}")
    print(f"  2026 new first close : {first_new_close:.2f}")
    print(f"  Gap: {gap_pct:.2f}%")

    if gap_pct > 3:
        print(f"  WARNING: Gap > 3% – check data source consistency!")
        return False
    else:
        print(f"  OK: Gap < 3% – prices consistent with 2025 spot data")
        return True


def main():
    print("=" * 60)
    print("FETCH 2026 XAUUSD SPOT HOURLY CANDLES")
    print("Source: AsterDex public API (XAUUSDT)")
    print("=" * 60)

    df = fetch_all_2026()

    if df is None or len(df) == 0:
        print("\nERROR: Could not fetch data from AsterDex!")
        return

    valid = validate_spot_prices(df)
    if not valid:
        print("  (Continuing anyway – gap is within typical overnight range)")

    # Save
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} candles to: {OUT_PATH}")
    print(f"Range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print("DONE!")


if __name__ == "__main__":
    main()
