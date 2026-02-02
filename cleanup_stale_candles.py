"""
Cleanup script to remove stale candles (O=H=L=C) from new_candles_2026.csv
Run this ONCE on Render to fix the corrupted data.
"""
import pandas as pd
import os

def cleanup_stale_candles():
    csv_path = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')
    
    if not os.path.exists(csv_path):
        print("No new_candles_2026.csv found")
        return
    
    # Load candles
    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    original_count = len(df)
    print(f"Loaded {original_count} candles")
    
    # Remove stale candles where O=H=L=C (no movement)
    df['range'] = df['high'] - df['low']
    stale_mask = df['range'] < 0.01
    stale_count = stale_mask.sum()
    
    print(f"Found {stale_count} stale candles (O=H=L=C)")
    
    # Show some examples
    if stale_count > 0:
        print("\nStale candle examples:")
        print(df[stale_mask].head(10))
    
    # Remove stale candles
    df_clean = df[~stale_mask].drop(columns=['range'])
    clean_count = len(df_clean)
    
    print(f"\nKeeping {clean_count} valid candles")
    print(f"Removed {original_count - clean_count} stale candles")
    
    # Backup original
    backup_path = csv_path.replace('.csv', '_backup.csv')
    df.drop(columns=['range']).to_csv(backup_path, index=False)
    print(f"Backup saved to: {backup_path}")
    
    # Save cleaned data
    df_clean.to_csv(csv_path, index=False)
    print(f"Cleaned data saved to: {csv_path}")

if __name__ == "__main__":
    cleanup_stale_candles()
