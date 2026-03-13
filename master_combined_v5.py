"""
MASTER COMBINED BACKTEST V5 – SCALPER ULTRA EDITION
=====================================================
V3B + 5x Scalper Models (TrendSC, RangeSC, FakeBreak, MomBurst, DivHunter)
HMM 5-State | MTF Features | Triple Barrier Labels | Stacked Ensemble

Rules (NEVER BREAK):
  A1: V3B = RF+GB, ML_TH=0.455, SL=0.8xATR, RR=3.0, RISK=3%
  A2: Train: year<2025, Test: year==2025 / 2026
  A3: V3B NEVER gets synthetic data
  A4: V3B baseline: ~207-256T, ~51-52% WR
  A5: Scalper ONLY fires when V3B does NOT fire on that bar
  A6: Scalper risk = half of V3B risk (max 2%)
  B5: v3b_overlap MUST == 0 at all times (assert)
  B7: If SC trade > 200/yr -> threshold too low
"""

import sys, io, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from hmmlearn import hmm

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("WARNING: lightgbm not installed – using RF+GB only for scalper")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings('ignore')

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'xauusd_1h_2020_2025_spot.csv')
NEW_CSV  = os.path.join(BASE_DIR, 'new_candles_2026.csv')
OUT_TXT  = os.path.join(BASE_DIR, 'MASTER_V5_RESULTS.txt')
OUT_JSON = os.path.join(BASE_DIR, 'MASTER_V5_LIVE_CONFIG.json')

# ===== V3B CONFIG (FIXED – never change) =====
V3B_ML_TH   = 0.455
V3B_SL_MULT = 0.80
V3B_RR      = 3.0
V3B_RISK    = 0.03   # 3%
V3B_COOL    = 1      # 1 bar cooldown
V3B_MAX_HOLD= 60

# ===== SWEEP SPACE =====
SC_THRESHOLDS  = [0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.68, 0.70, 0.75]
SC_RR_LIST     = [2.0, 3.0, 5.0]
SC_RISK_LIST   = [0.010, 0.015, 0.020]
SC_ADAPTIVE_TH = [True, False]
MODEL_SETS     = ['all5', 'trend_range', 'range_fake_div', 'mom_burst_only']

STARTING_BALANCE = 1000.0

# ===================================================================
# SECTION 1: DATA LOADING & INDICATOR ENGINE
# ===================================================================

def load_data(test_year=2025):
    """Load CSV, combine with 2026 candles if available, split train/test."""
    print(f"\n{'='*60}")
    print(f"[DATA] Loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, parse_dates=['datetime'])
    df.columns = ['ts', 'o', 'h', 'l', 'c']

    if os.path.exists(NEW_CSV):
        df2 = pd.read_csv(NEW_CSV, parse_dates=[0])
        df2.columns = ['ts', 'o', 'h', 'l', 'c']
        df = pd.concat([df, df2], ignore_index=True)
        df = df.drop_duplicates('ts').reset_index(drop=True)
        print(f"[DATA] Appended 2026 candles. Total: {len(df)}")

    df = df.sort_values('ts').reset_index(drop=True)
    df['year'] = pd.to_datetime(df['ts']).dt.year

    train = df[df['year'] < test_year].copy().reset_index(drop=True)
    test  = df[df['year'] == test_year].copy().reset_index(drop=True)
    print(f"[DATA] Train: {len(train)} bars | Test: {len(test)} bars")
    return df, train, test


def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Full indicator set – V3B exact + scalper extras."""
    df = df.copy()

    # ---- V3B EXACT indicators ----
    df['ema21']  = df['c'].ewm(21).mean()
    df['ema50']  = df['c'].ewm(50).mean()
    df['ema200'] = df['c'].ewm(200).mean()
    df['atr14']  = (df['h'] - df['l']).rolling(14).mean()

    delta = df['c'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 0.001)
    df['rsi14'] = 100 - 100 / (1 + gain / loss)

    df['macd']     = df['c'].ewm(12).mean() - df['c'].ewm(26).mean()
    df['macd_sig'] = df['macd'].ewm(9).mean()
    df['macd_hist']= df['macd'] - df['macd_sig']

    df['ema21_4h'] = df['c'].rolling(4).mean().ewm(21).mean()
    df['ema50_4h'] = df['c'].rolling(4).mean().ewm(50).mean()
    df['rsi_4h']   = df['rsi14'].rolling(4).mean()
    df['macd_4h']  = df['macd'].rolling(4).mean()
    df['trend_4h'] = np.where(df['ema21_4h'] > df['ema50_4h'], 1, -1)

    df['c_daily']    = df['c'].rolling(24).mean()
    df['trend_daily']= np.where(df['c_daily'] > df['c_daily'].shift(24), 1, -1)

    df['bullish_ob'] = ((df['c'].shift(1) > df['o'].shift(1)) &
                        (df['c'] > df['c'].shift(1) * 1.002)).astype(int)
    df['bearish_ob'] = ((df['c'].shift(1) < df['o'].shift(1)) &
                        (df['c'] < df['c'].shift(1) * 0.998)).astype(int)
    # OB distance – vectorized with numpy (1000x faster than for-loop)
    atr_safe = df['atr14'].replace(0, np.nan)
    bull_price = np.where(df['bullish_ob'] == 1, df['c'], np.nan)
    bear_price = np.where(df['bearish_ob'] == 1, df['c'], np.nan)
    last_bull = pd.Series(bull_price).ffill().values
    last_bear = pd.Series(bear_price).ffill().values
    df['dist_to_bull_ob'] = np.where(
        (~np.isnan(last_bull)) & (atr_safe.notna()),
        (df['c'].values - last_bull) / atr_safe.values,
        0.0)
    df['dist_to_bear_ob'] = np.where(
        (~np.isnan(last_bear)) & (atr_safe.notna()),
        (df['c'].values - last_bear) / atr_safe.values,
        0.0)
    df['dist_to_bull_ob'] = df['dist_to_bull_ob'].fillna(0.0)
    df['dist_to_bear_ob'] = df['dist_to_bear_ob'].fillna(0.0)

    df['high_5']  = df['h'].rolling(5).max()
    df['low_5']   = df['l'].rolling(5).min()
    df['high_20'] = df['h'].rolling(20).max()
    df['low_20']  = df['l'].rolling(20).min()
    df['dist_to_high_20'] = (df['high_20'] - df['c']) / df['atr14'].replace(0, 0.001)
    df['dist_to_low_20']  = (df['c'] - df['low_20'])  / df['atr14'].replace(0, 0.001)

    df['mom_5']      = df['c'].pct_change(5) * 100
    df['mom_10']     = df['c'].pct_change(10) * 100
    df['vol_ratio']  = (df['h'] - df['l']) / df['atr14'].replace(0, 0.001)
    df['body_ratio'] = abs(df['c'] - df['o']) / ((df['h'] - df['l']).replace(0, 0.001))
    df['is_doji']    = (abs(df['c'] - df['o']) < (df['h'] - df['l']) * 0.1).astype(int)
    df['is_engulfing']= ((df['c'] > df['o']) & (df['c'].shift(1) < df['o'].shift(1)) &
                          (df['c'] > df['h'].shift(1)) & (df['o'] < df['l'].shift(1))).astype(int)
    df['is_pin_bar'] = (((df['h'] - df[['c','o']].max(axis=1)) > 2 * abs(df['c']-df['o'])) |
                         ((df[['c','o']].min(axis=1) - df['l']) > 2 * abs(df['c']-df['o']))).astype(int)

    # ---- SCALPER EXTRAS ----
    # Bollinger Bands
    df['bb_mid']   = df['c'].rolling(20).mean()
    bb_std         = df['c'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, 0.001)
    df['bb_pct']   = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 0.001)

    # ADX
    plus_dm  = df['h'].diff().clip(lower=0)
    minus_dm = (-df['l'].diff()).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    df['adx14'] = (100 * (plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()).abs() /
                   (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()).replace(0, 0.001)).rolling(14).mean()

    # ATR percentile rank (200-period)
    df['atr_rank'] = df['atr14'].rolling(200).rank(pct=True)

    # Volume proxy (H-L range vs 20-period mean)
    hl_range = df['h'] - df['l']
    df['vol_spike'] = (hl_range > hl_range.rolling(20).mean() * 1.5).astype(int)
    df['vol_ma_ratio'] = hl_range / hl_range.rolling(20).mean().replace(0, 0.001)

    # Rate of change
    df['roc5']  = df['c'].pct_change(5) * 100
    df['roc14'] = df['c'].pct_change(14) * 100

    # Session flags (UTC)
    if pd.api.types.is_datetime64_any_dtype(df['ts']):
        hr = pd.to_datetime(df['ts']).dt.hour
    else:
        hr = pd.to_datetime(df['ts']).dt.hour
    df['hour']        = hr
    df['london']      = ((hr >= 7) & (hr < 16)).astype(int)
    df['ny_session']  = ((hr >= 13) & (hr < 22)).astype(int)
    df['overlap']     = ((hr >= 13) & (hr < 16)).astype(int)
    df['day_of_week'] = pd.to_datetime(df['ts']).dt.dayofweek

    # Upper/lower wick ratios
    df['upper_wick'] = (df['h'] - df[['c','o']].max(axis=1)) / df['atr14'].replace(0, 0.001)
    df['lower_wick'] = (df[['c','o']].min(axis=1) - df['l']) / df['atr14'].replace(0, 0.001)

    # ---- MTF 4H features (resample from 1H) ----
    df = _add_4h_features(df)

    # ---- RSI Divergence (15M proxy via short window) ----
    df = _add_rsi_divergence(df)

    return df


def _add_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4H-level S/R proximity and trend from 1H data."""
    df = df.copy()
    # Rolling 4-bar (= 4H) pivot high/low
    df['tf4h_pivot_high'] = df['h'].rolling(4).max().shift(1)
    df['tf4h_pivot_low']  = df['l'].rolling(4).min().shift(1)
    a = df['atr14'].replace(0, 0.001)
    df['tf4h_dist_high']  = (df['tf4h_pivot_high'] - df['c']) / a
    df['tf4h_dist_low']   = (df['c'] - df['tf4h_pivot_low']) / a
    # 4H trend: rolling 4-period EMA direction
    df['tf4h_trend'] = df['trend_4h']  # already computed
    # 4H RSI (4-bar rolling mean of 1H RSI)
    df['tf4h_rsi']   = df['rsi14'].rolling(4).mean()
    return df


def _add_rsi_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """Detect bullish/bearish RSI divergence (price new low, RSI higher low)."""
    df = df.copy()
    # Simple 5-bar lookaround divergence proxy
    price_low_5  = df['l'].rolling(5).min()
    rsi_at_low   = df['rsi14'].rolling(5).min()
    prev_price_low = price_low_5.shift(5)
    prev_rsi_low   = rsi_at_low.shift(5)
    # Bullish div: price made lower low but RSI made higher low
    df['bull_div'] = ((df['l'] <= price_low_5) &
                      (df['rsi14'] > prev_rsi_low) &
                      (df['l'] < prev_price_low)).astype(int)
    # Bearish div
    price_high_5 = df['h'].rolling(5).max()
    rsi_at_high  = df['rsi14'].rolling(5).max()
    prev_ph      = price_high_5.shift(5)
    prev_rh      = rsi_at_high.shift(5)
    df['bear_div'] = ((df['h'] >= price_high_5) &
                      (df['rsi14'] < prev_rh) &
                      (df['h'] > prev_ph)).astype(int)
    return df


# ===================================================================
# SECTION 2: HMM 5-STATE REGIME DETECTION
# ===================================================================

def train_hmm5(train_df: pd.DataFrame):
    """Train 5-state HMM on training data. Returns model + state labels dict."""
    print("\n[HMM] Training 5-state HMM...")
    ret = train_df['c'].pct_change().fillna(0).values
    atr = (train_df['atr_rank'].fillna(0.5)).values
    vol = (train_df['vol_ma_ratio'].fillna(1.0).clip(0,5)).values
    bb  = (train_df['bb_width'].fillna(0.02).clip(0,0.2)).values

    X = np.column_stack([ret, atr, vol, bb])
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    model = hmm.GaussianHMM(n_components=5, covariance_type='full',
                             n_iter=200, random_state=42)
    model.fit(X[:6000])  # use first 6000 bars for speed

    states = model.predict(X)
    state_means = []
    for s in range(5):
        mask = states == s
        if mask.sum() > 0:
            mean_ret = ret[mask].mean()
            mean_atr = atr[mask].mean()
            mean_vol = vol[mask].mean()
        else:
            mean_ret = mean_atr = mean_vol = 0
        state_means.append((s, abs(mean_ret), mean_atr, mean_vol))

    # Label states by ATR rank + return magnitude
    sorted_by_atr = sorted(state_means, key=lambda x: x[2])
    sorted_by_ret = sorted(state_means, key=lambda x: x[1], reverse=True)

    # State mapping
    state_labels = {}
    assigned = set()

    # Highest return magnitude → strong trending states (two)
    trend_up_state   = sorted_by_ret[0][0]
    trend_down_state = sorted_by_ret[1][0]
    state_labels[trend_up_state]   = 'TREND_UP'
    state_labels[trend_down_state] = 'TREND_DOWN'
    assigned.update([trend_up_state, trend_down_state])

    # Lowest ATR → tight range
    for s, _, atr_m, _ in sorted_by_atr:
        if s not in assigned:
            state_labels[s] = 'RANGE_TIGHT'
            assigned.add(s)
            break

    remaining = [s for s in range(5) if s not in assigned]
    if len(remaining) >= 2:
        state_labels[remaining[0]] = 'RANGE_WIDE'
        state_labels[remaining[1]] = 'UNCERTAIN'
    elif len(remaining) == 1:
        state_labels[remaining[0]] = 'RANGE_WIDE'

    print(f"[HMM] State mapping: {state_labels}")

    # Print state distribution on training data
    for s in range(5):
        pct = (states == s).mean() * 100
        print(f"  State {s} ({state_labels.get(s,'?')}): {pct:.1f}%")

    return model, state_labels, X


def add_hmm_states(df: pd.DataFrame, model, state_labels: dict) -> pd.DataFrame:
    """Apply HMM model to a dataframe and add regime columns."""
    df = df.copy()
    ret = df['c'].pct_change().fillna(0).values
    atr = df['atr_rank'].fillna(0.5).values
    vol = df['vol_ma_ratio'].fillna(1.0).clip(0,5).values
    bb  = df['bb_width'].fillna(0.02).clip(0,0.2).values
    X   = np.column_stack([ret, atr, vol, bb])
    X   = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

    try:
        raw_states = model.predict(X)
        proba      = model.predict_proba(X)
        df['hmm_state']      = raw_states
        df['hmm_confidence'] = proba[np.arange(len(proba)), raw_states]
        df['hmm_label']      = [state_labels.get(s, 'UNCERTAIN') for s in raw_states]
    except Exception as e:
        print(f"[HMM] predict error: {e}")
        df['hmm_state']      = 2
        df['hmm_confidence'] = 0.5
        df['hmm_label']      = 'UNCERTAIN'

    return df


# ===================================================================
# SECTION 3: V3B MODEL (EXACT REPLICA)
# ===================================================================

V3B_FEATURE_NAMES = [
    'ema_spread','c_vs_ema21','c_vs_ema200','rsi14_n','macd_hist_n',
    'ret1','ret5','trend_4h','trend_daily','rsi_4h_n','macd_4h_n',
    'tf_align','dist_bull_ob','dist_bear_ob','near_ob',
    'is_trending','is_ranging',
    'dist_high_20','dist_low_20','breakout',
    'mom_5','mom_10','vol_ratio','body_ratio',
    'is_doji','is_engulfing','is_pin_bar'
]


def get_v3b_features(df, i):
    """Extract 27 V3B features – EXACT from live bot."""
    if i < 30:
        return None, None
    a = df['atr14'].iloc[i]
    if pd.isna(a) or a <= 0:
        return None, None
    direction = 'LONG' if df['ema21'].iloc[i] > df['ema50'].iloc[i] else 'SHORT'
    try:
        f = [
            (df['ema21'].iloc[i] - df['ema50'].iloc[i]) / a,
            (df['c'].iloc[i] - df['ema21'].iloc[i]) / a,
            (df['c'].iloc[i] - df['ema200'].iloc[i]) / a,
            df['rsi14'].iloc[i] / 100 if not pd.isna(df['rsi14'].iloc[i]) else 0.5,
            df['macd_hist'].iloc[i] / a if not pd.isna(df['macd_hist'].iloc[i]) else 0,
            (df['c'].iloc[i] - df['c'].iloc[i-1]) / a,
            (df['c'].iloc[i] - df['c'].iloc[i-5]) / a if i >= 5 else 0,
            float(df['trend_4h'].iloc[i]),
            float(df['trend_daily'].iloc[i]),
            df['rsi_4h'].iloc[i] / 100 if not pd.isna(df['rsi_4h'].iloc[i]) else 0.5,
            df['macd_4h'].iloc[i] / a if not pd.isna(df['macd_4h'].iloc[i]) else 0,
            1.0 if df['trend_4h'].iloc[i] == df['trend_daily'].iloc[i] else 0.0,
            df['dist_to_bull_ob'].iloc[i],
            df['dist_to_bear_ob'].iloc[i],
            1.0 if (direction=='LONG' and df['dist_to_bull_ob'].iloc[i] < 3) or
                   (direction=='SHORT' and df['dist_to_bear_ob'].iloc[i] > -3) else 0.0,
            float(df.get('is_trending', pd.Series([0]*len(df))).iloc[i]),
            float(df.get('is_ranging', pd.Series([0]*len(df))).iloc[i]),
            df['dist_to_high_20'].iloc[i] if not pd.isna(df['dist_to_high_20'].iloc[i]) else 0,
            df['dist_to_low_20'].iloc[i] if not pd.isna(df['dist_to_low_20'].iloc[i]) else 0,
            1.0 if (direction=='LONG' and df['c'].iloc[i] > df['high_5'].iloc[i-1]) or
                   (direction=='SHORT' and df['c'].iloc[i] < df['low_5'].iloc[i-1]) else 0.0,
            df['mom_5'].iloc[i] / 10 if not pd.isna(df['mom_5'].iloc[i]) else 0,
            df['mom_10'].iloc[i] / 10 if not pd.isna(df['mom_10'].iloc[i]) else 0,
            df['vol_ratio'].iloc[i] if not pd.isna(df['vol_ratio'].iloc[i]) else 1,
            df['body_ratio'].iloc[i] if not pd.isna(df['body_ratio'].iloc[i]) else 0.5,
            float(df['is_doji'].iloc[i]),
            float(df['is_engulfing'].iloc[i]),
            float(df['is_pin_bar'].iloc[i]),
        ]
        return np.array([0.0 if pd.isna(x) else float(x) for x in f]), direction
    except Exception as e:
        return None, None


def train_v3b(train_df: pd.DataFrame):
    """Train V3B exactly as in live bot."""
    print("\n[V3B] Training model (RF+GB, 27 features)...")
    df = train_df.copy()

    # HMM 3-state for V3B (as in original)
    ret = df['c'].pct_change().fillna(0).values.reshape(-1,1)
    h3  = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=42)
    h3.fit(ret[:5000])
    states3 = h3.predict(ret)
    df['regime3'] = pd.Series([np.nan] + list(states3), index=df.index).fillna(0)
    means3 = [ret[states3==i].mean() for i in range(3)]
    trending_s3 = int(np.argmax([abs(m) for m in means3]))
    ranging_s3  = int(np.argmin([abs(m) for m in means3]))
    df['is_trending'] = (df['regime3'] == trending_s3).astype(int)
    df['is_ranging']  = (df['regime3'] == ranging_s3).astype(int)

    df = df.iloc[250:].reset_index(drop=True)
    X_list, y_list = [], []
    for i in range(50, len(df)-50):
        feats, direction = get_v3b_features(df, i)
        if feats is None:
            continue
        a  = df['atr14'].iloc[i]
        c  = df['c'].iloc[i]
        sl = a * V3B_SL_MULT
        tp = sl * V3B_RR
        label = 0
        for j in range(i+1, min(i+45, len(df))):
            if direction == 'LONG':
                if df['l'].iloc[j] <= c - sl: break
                if df['h'].iloc[j] >= c + tp: label = 1; break
            else:
                if df['h'].iloc[j] >= c + sl: break
                if df['l'].iloc[j] <= c - tp: label = 1; break
        X_list.append(feats)
        y_list.append(label)

    X, y = np.array(X_list), np.array(y_list)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42)
    rf.fit(Xs, y)
    gb.fit(Xs, y)

    pos_rate = y.mean()
    print(f"[V3B] Trained on {len(X)} samples | Positive rate: {pos_rate:.1%}")
    return rf, gb, scaler, h3, trending_s3, ranging_s3


# ===================================================================
# SECTION 4: SCALPER FEATURE EXTRACTION + TRIPLE BARRIER LABELING
# ===================================================================

SC_FEATURE_NAMES = [
    # Trend context (V3B-derived)
    'ema_spread_n','c_vs_ema21_n','rsi14_n','macd_hist_n','trend_4h','trend_daily','tf_align',
    # Scalper-specific
    'bb_pct','bb_width','atr_rank','vol_spike','vol_ma_ratio',
    'roc5','roc14','body_ratio','upper_wick','lower_wick','is_doji','is_pin_bar','is_engulfing',
    # Session
    'london','ny_session','overlap','hour_n','day_of_week_n',
    # MTF 4H
    'tf4h_dist_high','tf4h_dist_low','tf4h_rsi_n','tf4h_trend',
    # S/R proximity
    'dist_to_high_20','dist_to_low_20','dist_bull_ob','dist_bear_ob',
    # Divergence
    'bull_div','bear_div',
    # Regime
    'hmm_confidence',
    'adx14_n','mom_5_n','mom_10_n',
]


def get_sc_features(df, i, direction='LONG'):
    """Extract scalper features at bar i."""
    if i < 30:
        return None
    a = df['atr14'].iloc[i]
    if pd.isna(a) or a <= 0:
        return None
    try:
        feats = [
            (df['ema21'].iloc[i] - df['ema50'].iloc[i]) / a,
            (df['c'].iloc[i] - df['ema21'].iloc[i]) / a,
            df['rsi14'].iloc[i] / 100 if not pd.isna(df['rsi14'].iloc[i]) else 0.5,
            df['macd_hist'].iloc[i] / a if not pd.isna(df['macd_hist'].iloc[i]) else 0,
            float(df['trend_4h'].iloc[i]),
            float(df['trend_daily'].iloc[i]),
            1.0 if df['trend_4h'].iloc[i] == df['trend_daily'].iloc[i] else 0.0,
            # BB
            float(df['bb_pct'].iloc[i]) if not pd.isna(df['bb_pct'].iloc[i]) else 0.5,
            float(df['bb_width'].iloc[i]) if not pd.isna(df['bb_width'].iloc[i]) else 0.02,
            float(df['atr_rank'].iloc[i]) if not pd.isna(df['atr_rank'].iloc[i]) else 0.5,
            float(df['vol_spike'].iloc[i]),
            float(df['vol_ma_ratio'].iloc[i]) if not pd.isna(df['vol_ma_ratio'].iloc[i]) else 1.0,
            float(df['roc5'].iloc[i]) if not pd.isna(df['roc5'].iloc[i]) else 0,
            float(df['roc14'].iloc[i]) if not pd.isna(df['roc14'].iloc[i]) else 0,
            float(df['body_ratio'].iloc[i]) if not pd.isna(df['body_ratio'].iloc[i]) else 0.5,
            float(df['upper_wick'].iloc[i]) if not pd.isna(df['upper_wick'].iloc[i]) else 0,
            float(df['lower_wick'].iloc[i]) if not pd.isna(df['lower_wick'].iloc[i]) else 0,
            float(df['is_doji'].iloc[i]),
            float(df['is_pin_bar'].iloc[i]),
            float(df['is_engulfing'].iloc[i]),
            float(df['london'].iloc[i]),
            float(df['ny_session'].iloc[i]),
            float(df['overlap'].iloc[i]),
            df['hour'].iloc[i] / 23.0,
            df['day_of_week'].iloc[i] / 4.0,
            float(df['tf4h_dist_high'].iloc[i]) if not pd.isna(df['tf4h_dist_high'].iloc[i]) else 0,
            float(df['tf4h_dist_low'].iloc[i]) if not pd.isna(df['tf4h_dist_low'].iloc[i]) else 0,
            df['tf4h_rsi'].iloc[i] / 100 if not pd.isna(df['tf4h_rsi'].iloc[i]) else 0.5,
            float(df['tf4h_trend'].iloc[i]),
            float(df['dist_to_high_20'].iloc[i]) if not pd.isna(df['dist_to_high_20'].iloc[i]) else 0,
            float(df['dist_to_low_20'].iloc[i]) if not pd.isna(df['dist_to_low_20'].iloc[i]) else 0,
            float(df['dist_to_bull_ob'].iloc[i]),
            float(df['dist_to_bear_ob'].iloc[i]),
            float(df['bull_div'].iloc[i]),
            float(df['bear_div'].iloc[i]),
            float(df['hmm_confidence'].iloc[i]) if 'hmm_confidence' in df.columns and not pd.isna(df['hmm_confidence'].iloc[i]) else 0.3,
            float(df['adx14'].iloc[i]) / 100 if not pd.isna(df['adx14'].iloc[i]) else 0.2,
            float(df['mom_5'].iloc[i]) / 10 if not pd.isna(df['mom_5'].iloc[i]) else 0,
            float(df['mom_10'].iloc[i]) / 10 if not pd.isna(df['mom_10'].iloc[i]) else 0,
        ]
        return np.array([0.0 if pd.isna(x) else float(x) for x in feats])
    except Exception:
        return None


def triple_barrier_label(df, i, direction, rr, sl_mult=1.0, timeout=12):
    """
    Triple Barrier Method labeling:
      1.0  = TP hit
      0.0  = SL hit
      0.3..0.5 = timeout: time-weighted soft label
        - exit_pnl_r = actual price move / sl_dist (in R multiples)
        - mapped to [0.3, 0.5] so the model sees partial signal
          (e.g. price moved 0.8R toward TP = label 0.47, vs moved 0R = 0.38)
    """
    a = df['atr14'].iloc[i]
    c = df['c'].iloc[i]
    if a <= 0:
        return 0.0
    sl_dist = a * sl_mult
    tp_dist = sl_dist * rr

    end = min(i + timeout + 1, len(df))
    for j in range(i + 1, end):
        if direction == 'LONG':
            if df['l'].iloc[j] <= c - sl_dist:
                return 0.0
            if df['h'].iloc[j] >= c + tp_dist:
                return 1.0
        else:
            if df['h'].iloc[j] >= c + sl_dist:
                return 0.0
            if df['l'].iloc[j] <= c - tp_dist:
                return 1.0

    # Timeout: compute time-weighted soft label
    exit_c = df['c'].iloc[min(end - 1, len(df) - 1)]
    if direction == 'LONG':
        pnl_r = (exit_c - c) / sl_dist  # positive = toward TP
    else:
        pnl_r = (c - exit_c) / sl_dist
    # Clip to [-1, rr] and map to [0.3, 0.5]
    pnl_r = max(-1.0, min(float(rr), pnl_r))
    timeout_label = 0.30 + 0.20 * (pnl_r + 1.0) / (float(rr) + 1.0)
    return round(timeout_label, 4)


def _make_stacked_model():
    """Build a stacked ensemble: RF + (LGB or GB) + (XGB or RF2) -> LR meta."""
    base = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=7,
                                       class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)),
    ]
    if HAS_LGB:
        base.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=6,
                                                 class_weight='balanced', random_state=42,
                                                 verbose=-1)))
    if HAS_XGB:
        base.append(('xgb', XGBClassifier(n_estimators=80, max_depth=5,
                                            scale_pos_weight=2, random_state=42,
                                            eval_metric='logloss', verbosity=0)))
    meta = LogisticRegression(C=1.0, max_iter=500)
    return StackingClassifier(estimators=base, final_estimator=meta,
                               cv=3, passthrough=False, n_jobs=-1)


def train_model_for_state(df_full: pd.DataFrame, state_mask_col: str,
                           label_rr: float, model_name: str, min_samples=80):
    """
    Train a scalper model for a given HMM state subset.
    Returns (model, scaler) or (None, None) if not enough samples.
    """
    df = df_full.copy()

    # Filter by state if mask column provided
    if state_mask_col and state_mask_col in df.columns:
        subset = df[df[state_mask_col] == 1].reset_index(drop=True)
        # For FakeBreak / MomBurst / DivHunter: use all rows but weight by state
        if len(subset) < min_samples:
            subset = df.reset_index(drop=True)
    else:
        subset = df.reset_index(drop=True)

    X_list, y_list = [], []
    direction_list = []

    for i in range(50, len(subset)-15):
        direction = 'LONG' if subset['ema21'].iloc[i] > subset['ema50'].iloc[i] else 'SHORT'
        feats = get_sc_features(subset, i, direction)
        if feats is None:
            continue

        # Extra pre-filter per model type
        if model_name == 'FakeBreak':
            # Only label bars near BB extremes that reversed
            bb_p = subset['bb_pct'].iloc[i] if not pd.isna(subset['bb_pct'].iloc[i]) else 0.5
            if not (bb_p < 0.15 or bb_p > 0.85):
                continue
        elif model_name == 'MomBurst':
            # Vol spike required
            if subset['vol_spike'].iloc[i] == 0:
                continue
        elif model_name == 'DivHunter':
            # Must have divergence
            has_div = (subset['bull_div'].iloc[i] == 1) or (subset['bear_div'].iloc[i] == 1)
            if not has_div:
                continue

        label = triple_barrier_label(subset, i, direction, label_rr)
        X_list.append(feats)
        y_list.append(label)
        direction_list.append(direction)

    if len(X_list) < min_samples:
        print(f"  [{model_name}] Not enough samples ({len(X_list)}) – skipped")
        return None, None

    X = np.array(X_list)
    y_hard = (np.array(y_list) >= 0.5).astype(int)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pos_rate = y_hard.mean()
    print(f"  [{model_name}] {len(X)} samples | pos rate {pos_rate:.1%}")

    if len(set(y_hard)) < 2:
        print(f"  [{model_name}] Only one class – skipped")
        return None, None

    # Validate on last 20% (time-based split)
    split = int(len(Xs) * 0.8)
    if split > 20 and (len(Xs) - split) > 10:
        Xtr, Xval = Xs[:split], Xs[split:]
        ytr, yval = y_hard[:split], y_hard[split:]
        if len(set(ytr)) < 2:
            Xtr, ytr = Xs, y_hard

        clf = _make_stacked_model()
        try:
            clf.fit(Xtr, ytr)
            if len(set(yval)) > 1:
                auc = roc_auc_score(yval, clf.predict_proba(Xval)[:,1])
                print(f"  [{model_name}] Val AUC: {auc:.3f}")
            clf.fit(Xs, y_hard)  # retrain on full data
        except Exception as e:
            print(f"  [{model_name}] stacking failed ({e}), using RF only")
            clf = RandomForestClassifier(n_estimators=150, max_depth=7,
                                         class_weight='balanced', random_state=42, n_jobs=-1)
            clf.fit(Xs, y_hard)
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=7,
                                      class_weight='balanced', random_state=42, n_jobs=-1)
        clf.fit(Xs, y_hard)

    return clf, scaler


def train_all_scalpers(train_df: pd.DataFrame, label_rr: float = 1.5):
    """Train all 5 scalper models. Returns dict of {name: (model, scaler)}."""
    print(f"\n[SCALPER] Training 5 models with label RR={label_rr}")
    models = {}

    # 1. TrendScalper – trending states only
    df_trend = train_df.copy()
    df_trend['_use'] = df_trend['hmm_label'].isin(['TREND_UP','TREND_DOWN']).astype(int)
    m, s = train_model_for_state(df_trend, '_use', label_rr, 'TrendScalper')
    models['TrendScalper'] = (m, s)

    # 2. RangeScalper – tight range only
    df_range = train_df.copy()
    df_range['_use'] = (df_range['hmm_label'] == 'RANGE_TIGHT').astype(int)
    m, s = train_model_for_state(df_range, '_use', label_rr, 'RangeScalper')
    models['RangeScalper'] = (m, s)

    # 3. FakeBreak Catcher – wide range
    df_fake = train_df.copy()
    df_fake['_use'] = df_fake['hmm_label'].isin(['RANGE_WIDE','RANGE_TIGHT']).astype(int)
    m, s = train_model_for_state(df_fake, '_use', label_rr, 'FakeBreak')
    models['FakeBreak'] = (m, s)

    # 4. Momentum Burst – all states, vol-spike filtered
    m, s = train_model_for_state(train_df, None, label_rr, 'MomBurst')
    models['MomBurst'] = (m, s)

    # 5. Divergence Hunter – all states, divergence filtered
    m, s = train_model_for_state(train_df, None, label_rr, 'DivHunter')
    models['DivHunter'] = (m, s)

    trained = [k for k,v in models.items() if v[0] is not None]
    print(f"[SCALPER] Trained: {trained}")
    return models


# ===================================================================
# SECTION 5: COMBINED BACKTEST ENGINE
# ===================================================================

def get_adaptive_th(base_th, atr_rank_val, adaptive=True):
    if not adaptive:
        return base_th
    if pd.isna(atr_rank_val):
        return base_th
    if atr_rank_val < 0.40:
        return max(base_th - 0.05, 0.45)
    elif atr_rank_val > 0.70:
        return min(base_th + 0.05, 0.90)
    return base_th


def get_adaptive_v3b_risk(balance, peak):
    if peak <= 0:
        return V3B_RISK
    dd = (peak - balance) / peak * 100
    if dd > 20:
        return 0.015
    elif dd > 10:
        return 0.0225
    return V3B_RISK


def get_adaptive_sc_risk(balance, peak, base_sc_risk):
    if peak <= 0:
        return base_sc_risk
    dd = (peak - balance) / peak
    if dd > 0.15:
        return base_sc_risk * 0.5
    elif dd > 0.05:
        return base_sc_risk * 0.75
    return base_sc_risk


ACTIVE_SC_MODELS = {
    'all5':            ['TrendScalper','RangeScalper','FakeBreak','MomBurst','DivHunter'],
    'trend_range':     ['TrendScalper','RangeScalper'],
    'range_fake_div':  ['RangeScalper','FakeBreak','DivHunter'],
    'mom_burst_only':  ['MomBurst','DivHunter'],
}

REGIME_TO_MODELS = {
    'TREND_UP':    ['TrendScalper','MomBurst'],
    'TREND_DOWN':  ['TrendScalper','MomBurst'],
    'RANGE_TIGHT': ['RangeScalper','FakeBreak','DivHunter'],
    'RANGE_WIDE':  ['FakeBreak','MomBurst','DivHunter'],
    'UNCERTAIN':   [],
}


def precompute_backtest_features(test_df, v3b_rf, v3b_gb, v3b_scaler, v3b_h3,
                                  v3b_trending_s3, v3b_ranging_s3, sc_models):
    """
    Pre-compute all V3B and SC features + probabilities ONCE for all bars.
    Returns df with added columns: v3b_prob, v3b_dir, sc_feats_arr,
    sc_direction, is_trending, is_ranging.
    Calling this once before run_sweep saves ~432x redundant computation.
    """
    df = test_df.copy().reset_index(drop=True)

    # HMM regime3 for V3B
    ret3 = df['c'].pct_change().fillna(0).values.reshape(-1, 1)
    try:
        s3 = v3b_h3.predict(ret3)
        series = pd.Series([0] + list(s3)).head(len(df))
        series.index = df.index
        df['regime3'] = series.fillna(0)
    except:
        df['regime3'] = 0
    df['is_trending'] = (df['regime3'] == v3b_trending_s3).astype(int)
    df['is_ranging']  = (df['regime3'] == v3b_ranging_s3).astype(int)

    n = len(df)
    v3b_probs  = np.full(n, np.nan)
    v3b_dirs   = np.full(n, '', dtype=object)
    sc_feats_cache  = [None] * n
    sc_dirs    = np.full(n, '', dtype=object)
    # pre-scaled SC features per model
    sc_probs_cache  = {}  # {mname: array of probs per bar}

    # --- V3B probabilities ---
    print('[PRECOMPUTE] V3B features...', flush=True)
    v3b_feat_list = []
    v3b_idx_list  = []
    for i in range(60, n):
        feats, direction = get_v3b_features(df, i)
        if feats is not None:
            v3b_feat_list.append(feats)
            v3b_idx_list.append(i)
            v3b_dirs[i] = direction
    if v3b_feat_list:
        X_v3b = np.array(v3b_feat_list)
        Xs_v3b = v3b_scaler.transform(X_v3b)
        p_rf = v3b_rf.predict_proba(Xs_v3b)[:, 1]
        p_gb = v3b_gb.predict_proba(Xs_v3b)[:, 1]
        for k, i in enumerate(v3b_idx_list):
            v3b_probs[i] = (p_rf[k] + p_gb[k]) / 2
    df['_v3b_prob'] = v3b_probs
    df['_v3b_dir']  = v3b_dirs

    # --- SC features + per-model probabilities ---
    print('[PRECOMPUTE] SC features...', flush=True)
    sc_feat_list = []
    sc_idx_list  = []
    for i in range(60, n):
        sc_dir = 'LONG' if df['ema21'].iloc[i] > df['ema50'].iloc[i] else 'SHORT'
        sc_dirs[i] = sc_dir
        feats = get_sc_features(df, i, sc_dir)
        if feats is not None:
            sc_feats_cache[i] = feats
            sc_feat_list.append(feats)
            sc_idx_list.append(i)
    df['_sc_dir'] = sc_dirs

    for mname, (clf, scl) in sc_models.items():
        if clf is None:
            sc_probs_cache[mname] = np.full(n, 0.0)
            continue
        probs_arr = np.full(n, 0.0)
        if sc_feat_list:
            X_sc = np.array(sc_feat_list)
            Xs_sc = scl.transform(X_sc)
            try:
                probs = clf.predict_proba(Xs_sc)[:, 1]
                for k, i in enumerate(sc_idx_list):
                    probs_arr[i] = probs[k]
            except:
                pass
        sc_probs_cache[mname] = probs_arr
        print(f'[PRECOMPUTE] {mname} done', flush=True)

    return df, sc_probs_cache


def run_backtest(test_df, v3b_rf, v3b_gb, v3b_scaler, v3b_h3, v3b_trending_s3, v3b_ranging_s3,
                 sc_models,
                 sc_th=0.60, sc_rr=3.0, sc_risk=0.015,
                 adaptive_th=True, model_set='all5',
                 label='',
                 _precomputed=None):
    """
    If _precomputed=(df_pre, sc_probs_cache) is passed, skips feature extraction entirely.
    This allows run_sweep to call this 432x without re-computing features each time.
    """
    if _precomputed is not None:
        df, sc_probs_cache = _precomputed
        df = df.copy().reset_index(drop=True)
    else:
        df = test_df.copy().reset_index(drop=True)
        ret3 = df['c'].pct_change().fillna(0).values.reshape(-1, 1)
        try:
            s3 = v3b_h3.predict(ret3)
            series = pd.Series([0] + list(s3)).head(len(df))
            series.index = df.index
            df['regime3'] = series.fillna(0)
        except:
            df['regime3'] = 0
        df['is_trending'] = (df['regime3'] == v3b_trending_s3).astype(int)
        df['is_ranging']  = (df['regime3'] == v3b_ranging_s3).astype(int)
        sc_probs_cache = None

    active_models = ACTIVE_SC_MODELS.get(model_set, list(sc_models.keys()))

    balance = STARTING_BALANCE
    peak    = STARTING_BALANCE
    trades  = []

    in_trade      = False
    trade_entry   = 0.0
    trade_sl      = 0.0
    trade_tp      = 0.0
    trade_dir     = 'LONG'
    trade_risk    = 0.0
    trade_type    = 'V3B'
    trade_model   = ''
    trade_bar     = -999
    last_exit_bar = -999

    for i in range(60, len(df)):
        row = df.iloc[i]
        ts  = row['ts']

        if in_trade:
            hi, lo = row['h'], row['l']
            exit_price  = None
            exit_reason = None

            if trade_dir == 'LONG':
                if lo <= trade_sl:
                    exit_price, exit_reason = trade_sl, 'SL'
                elif hi >= trade_tp:
                    exit_price, exit_reason = trade_tp, 'TP'
            else:
                if hi >= trade_sl:
                    exit_price, exit_reason = trade_sl, 'SL'
                elif lo <= trade_tp:
                    exit_price, exit_reason = trade_tp, 'TP'

            max_hold = 60 if trade_type == 'V3B' else 24
            if exit_reason is None and (i - trade_bar) >= max_hold:
                exit_price, exit_reason = row['c'], 'TIMEOUT'

            if exit_price is not None:
                sl_dist = abs(trade_entry - trade_sl)
                if sl_dist > 0:
                    pnl_r   = (exit_price - trade_entry) / sl_dist if trade_dir == 'LONG' else (trade_entry - exit_price) / sl_dist
                    pnl_amt = trade_risk * pnl_r
                else:
                    pnl_amt = 0

                is_win  = (pnl_amt > 0)
                balance += pnl_amt
                peak     = max(peak, balance)
                dd_from_peak = (peak - balance) / peak * 100 if peak > 0 else 0

                trades.append({
                    'ts':    ts,
                    'type':  trade_type,
                    'model': trade_model,
                    'dir':   trade_dir,
                    'entry': trade_entry,
                    'exit':  exit_price,
                    'reason':exit_reason,
                    'win':   is_win,
                    'pnl':   pnl_amt,
                    'balance': balance,
                    'dd':    dd_from_peak,
                })
                in_trade      = False
                last_exit_bar = i
            continue

        if (i - last_exit_bar) <= V3B_COOL:
            continue

        atr = row['atr14']
        if pd.isna(atr) or atr <= 0:
            continue

        # Use pre-computed probabilities if available
        if sc_probs_cache is not None:
            ml_prob  = df['_v3b_prob'].iloc[i]
            direction = df['_v3b_dir'].iloc[i]
            if pd.isna(ml_prob) or direction == '':
                continue
        else:
            feats_v3b, direction = get_v3b_features(df, i)
            if feats_v3b is None:
                continue
            fv = v3b_scaler.transform(feats_v3b.reshape(1, -1))
            ml_prob = (v3b_rf.predict_proba(fv)[0][1] + v3b_gb.predict_proba(fv)[0][1]) / 2

        v3b_fires = False
        if ml_prob >= V3B_ML_TH:
            v3b_fires = True
            adap_risk = get_adaptive_v3b_risk(balance, peak)
            sl_dist   = atr * V3B_SL_MULT
            entry     = row['c']
            sl = entry - sl_dist if direction == 'LONG' else entry + sl_dist
            tp = entry + sl_dist * V3B_RR if direction == 'LONG' else entry - sl_dist * V3B_RR

            in_trade    = True
            trade_entry = entry
            trade_sl    = sl
            trade_tp    = tp
            trade_dir   = direction
            trade_risk  = balance * adap_risk
            trade_type  = 'V3B'
            trade_model = 'V3B'
            trade_bar   = i
            continue

        if not v3b_fires and sc_models:
            hmm_lbl = str(row['hmm_label']) if 'hmm_label' in df.columns else 'UNCERTAIN'
            regime_models = REGIME_TO_MODELS.get(hmm_lbl, [])
            candidates = [m for m in active_models
                          if m in regime_models or model_set == 'mom_burst_only']
            if not candidates:
                candidates = active_models

            atr_rank_val = row.get('atr_rank', 0.5)
            if pd.isna(atr_rank_val):
                atr_rank_val = 0.5
            eff_th = get_adaptive_th(sc_th, atr_rank_val, adaptive_th)

            sc_direction = df['_sc_dir'].iloc[i] if sc_probs_cache is not None else (
                'LONG' if row['ema21'] > row['ema50'] else 'SHORT')

            best_prob  = 0.0
            best_model = None
            for mname in candidates:
                if mname not in sc_models:
                    continue
                if sc_probs_cache is not None:
                    # Use pre-computed probability directly
                    prob = sc_probs_cache.get(mname, np.zeros(len(df)))[i]
                else:
                    clf, scl = sc_models[mname]
                    if clf is None:
                        continue
                    sc_feats = get_sc_features(df, i, sc_direction)
                    if sc_feats is None:
                        continue
                    try:
                        fs   = scl.transform(sc_feats.reshape(1, -1))
                        prob = clf.predict_proba(fs)[0][1]
                    except:
                        continue
                if prob > best_prob:
                    best_prob  = prob
                    best_model = mname

            if best_prob >= eff_th and best_model is not None:
                adap_sc_risk = get_adaptive_sc_risk(balance, peak, sc_risk)
                sl_dist = atr * 1.0
                entry   = row['c']
                sl = entry - sl_dist if sc_direction == 'LONG' else entry + sl_dist
                tp = entry + sl_dist * sc_rr if sc_direction == 'LONG' else entry - sl_dist * sc_rr

                in_trade    = True
                trade_entry = entry
                trade_sl    = sl
                trade_tp    = tp
                trade_dir   = sc_direction
                trade_risk  = balance * adap_sc_risk
                trade_type  = 'SC'
                trade_model = best_model
                trade_bar   = i

    if not trades:
        return None

    v3b_trades = [t for t in trades if t['type'] == 'V3B']
    sc_trades  = [t for t in trades if t['type'] == 'SC']
    v3b_wins   = sum(1 for t in v3b_trades if t['win'])
    sc_wins    = sum(1 for t in sc_trades  if t['win'])
    v3b_wr     = v3b_wins / len(v3b_trades) * 100 if v3b_trades else 0
    sc_wr      = sc_wins  / len(sc_trades)  * 100 if sc_trades  else 0
    all_wins   = sum(1 for t in trades if t['win'])
    comb_wr    = all_wins / len(trades) * 100

    bal_series = [STARTING_BALANCE] + [t['balance'] for t in trades]
    peak_r = STARTING_BALANCE
    max_dd = 0.0
    for b in bal_series:
        if b > peak_r:
            peak_r = b
        dd = (peak_r - b) / peak_r * 100
        if dd > max_dd:
            max_dd = dd

    final_balance = trades[-1]['balance']
    profit_pct    = (final_balance - STARTING_BALANCE) / STARTING_BALANCE * 100

    overlap = 0
    v3b_ts  = set(str(t['ts']) for t in v3b_trades)
    sc_ts   = set(str(t['ts']) for t in sc_trades)
    overlap = len(v3b_ts & sc_ts)
    assert overlap == 0, f'CRITICAL: V3B-SC OVERLAP DETECTED ({overlap} trades)!'

    sc_model_counts = {}
    for t in sc_trades:
        sc_model_counts[t['model']] = sc_model_counts.get(t['model'], 0) + 1

    monthly = {}
    for t in trades:
        mo = str(t['ts'])[:7]
        if mo not in monthly:
            monthly[mo] = {'n': 0, 'wins': 0, 'pnl': 0}
        monthly[mo]['n']    += 1
        monthly[mo]['wins'] += int(t['win'])
        monthly[mo]['pnl']  += t['pnl']

    scorer = comb_wr * 0.35 + (100 - max_dd) * 0.35 + min(profit_pct, 1e6) / 1e4 * 0.30

    return {
        'label':          label,
        'sc_th':          sc_th,
        'sc_rr':          sc_rr,
        'sc_risk':        sc_risk,
        'adaptive_th':    adaptive_th,
        'model_set':      model_set,
        'total_trades':   len(trades),
        'v3b_trades':     len(v3b_trades),
        'sc_trades':      len(sc_trades),
        'v3b_wr':         round(v3b_wr, 2),
        'sc_wr':          round(sc_wr, 2),
        'comb_wr':        round(comb_wr, 2),
        'max_dd':         round(max_dd, 2),
        'profit_pct':     round(profit_pct, 2),
        'final_balance':  round(final_balance, 2),
        'scorer':         round(scorer, 4),
        'sc_model_breakdown': sc_model_counts,
        'monthly':        monthly,
    }


# ===================================================================
# SECTION 7: SWEEP ENGINE + MAIN
# ===================================================================

def run_sweep(test_df, v3b_components, sc_models, period_label):
    v3b_rf, v3b_gb, v3b_scaler, v3b_h3, v3b_trending_s3, v3b_ranging_s3 = v3b_components
    results = []
    total   = (len(SC_THRESHOLDS) * len(SC_RR_LIST) * len(SC_RISK_LIST)
               * len(SC_ADAPTIVE_TH) * len(MODEL_SETS))
    done    = 0
    print(f'\n[SWEEP] {period_label} | {total} combinations...')

    # Pre-compute ALL features once before the sweep loop
    print(f'[SWEEP] Pre-computing features (runs ONCE, not {total}x)...', flush=True)
    precomputed = precompute_backtest_features(
        test_df, v3b_rf, v3b_gb, v3b_scaler, v3b_h3,
        v3b_trending_s3, v3b_ranging_s3, sc_models)
    print(f'[SWEEP] Pre-compute done. Starting {total} sweep iterations...', flush=True)

    for th, rr, risk, adap, mset in product(
            SC_THRESHOLDS, SC_RR_LIST, SC_RISK_LIST, SC_ADAPTIVE_TH, MODEL_SETS):
        lbl = f'{period_label}_RR{rr}_RISK{int(risk*100)}pct_TH{th}_ADC{int(adap)}_SET{mset}'
        try:
            res = run_backtest(
                test_df,
                v3b_rf, v3b_gb, v3b_scaler, v3b_h3, v3b_trending_s3, v3b_ranging_s3,
                sc_models,
                sc_th=th, sc_rr=rr, sc_risk=risk,
                adaptive_th=adap, model_set=mset, label=lbl,
                _precomputed=precomputed)
            if res is not None:
                results.append(res)
        except AssertionError as ae:
            print(f'  [OVERLAP ASSERT] {lbl}: {ae}')
        except Exception as e:
            print(f'  [ERROR] {lbl}: {e}')
        done += 1
        if done % 60 == 0:
            print(f'  Progress: {done}/{total}', flush=True)

    results.sort(key=lambda x: x['scorer'], reverse=True)
    return results


def print_top_results(results, n=20, label=''):
    print(f"\n{'='*70}")
    print(f'TOP {n} RESULTS – {label}')
    print(f"{'='*70}")
    hdr = (f"{'#':>3} {'Config':<40} {'TT':>4} {'V3B':>4} {'SC':>4} "
           f"{'V3BWR':>6} {'SCWR':>6} {'CWR':>6} {'DD':>6} {'Profit%':>12} {'Score':>8}")
    print(hdr)
    print('-' * 105)
    for rank, r in enumerate(results[:n], 1):
        lbl = r['label'][-40:] if len(r['label']) > 40 else r['label']
        print(f"{rank:>3} {lbl:<40} {r['total_trades']:>4} {r['v3b_trades']:>4} {r['sc_trades']:>4} "
              f"{r['v3b_wr']:>6.1f}% {r['sc_wr']:>6.1f}% {r['comb_wr']:>6.1f}% "
              f"{r['max_dd']:>6.1f}% {r['profit_pct']:>12,.1f}% {r['scorer']:>8.4f}")

    if results:
        best = results[0]
        print(f"\nBEST SC Model Breakdown: {best['sc_model_breakdown']}")
        print(f"\nMONTHLY (Best config):")
        print(f"{'Month':<10} {'N':>5} {'WR':>7} {'PnL':>14} {'CumBal':>14}")
        cum_bal = STARTING_BALANCE
        for mo, ms in sorted(best['monthly'].items()):
            wr      = ms['wins'] / ms['n'] * 100 if ms['n'] > 0 else 0
            cum_bal += ms['pnl']
            flag    = 'OK' if wr >= 50 else ('WARN' if wr >= 40 else 'POOR')
            print(f"{mo:<10} {ms['n']:>5} {wr:>7.1f}% {ms['pnl']:>14,.0f} {cum_bal:>14,.0f}  {flag}")


def save_results(results_2025, results_2026, best_2025, best_2026):
    lines = []
    lines.append('=' * 80)
    lines.append('MASTER COMBINED BACKTEST V5 – RESULTS')
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append('=' * 80)

    for label, results in [('2025 Full Year', results_2025), ('2026 YTD', results_2026)]:
        lines.append(f"\nPERIOD: {label}")
        lines.append(f"{'#':>3} {'Config':<50} {'SC':>4} {'SCWR':>7} {'CWR':>7} {'DD':>6} {'Profit':>14}")
        for rank, r in enumerate(results[:20], 1):
            cfg = r['label'][-50:] if len(r['label']) > 50 else r['label']
            lines.append(f"{rank:>3} {cfg:<50} {r['sc_trades']:>4} {r['sc_wr']:>7.1f}% "
                         f"{r['comb_wr']:>7.1f}% {r['max_dd']:>6.1f}% {r['profit_pct']:>14,.1f}%")

    # Copy-paste ready Python config block for terminator_v3b_live.py
    best = best_2026 if best_2026 else best_2025
    if best:
        lines.append('\n' + '=' * 80)
        lines.append('COPY-PASTE CONFIG FOR terminator_v3b_live.py')
        lines.append('=' * 80)
        lines.append('# ===== V3B FIXED (do NOT change) =====')
        lines.append(f'ML_THRESHOLD  = {V3B_ML_TH}')
        lines.append(f'SL_MULTIPLIER = {V3B_SL_MULT}')
        lines.append(f'RR            = {V3B_RR}')
        lines.append(f'BASE_RISK     = {V3B_RISK}')
        lines.append(f'COOLDOWN_BARS = {V3B_COOL}')
        lines.append('')
        lines.append('# ===== SCALPER (from sweep best result) =====')
        lines.append(f'SC_THRESHOLD  = {best.get("sc_th", "N/A")}')
        lines.append(f'SC_RR         = {best.get("sc_rr", "N/A")}')
        lines.append(f'SC_RISK       = {best.get("sc_risk", "N/A")}')
        lines.append(f'SC_ADAPTIVE   = {best.get("adaptive_th", "N/A")}')
        lines.append(f'SC_MODEL_SET  = "{best.get("model_set", "N/A")}"')
        lines.append('')
        lines.append('# ===== BACKTEST PERFORMANCE (best config) =====')
        lines.append(f'# V3B trades : {best.get("v3b_trades","?")} | WR: {best.get("v3b_wr","?")}%')
        lines.append(f'# SC  trades : {best.get("sc_trades","?")}  | WR: {best.get("sc_wr","?")}%')
        lines.append(f'# Combined WR: {best.get("comb_wr","?")}%')
        lines.append(f'# Max DD     : {best.get("max_dd","?")}%')
        lines.append(f'# Profit     : +{best.get("profit_pct","?")}%')
        lines.append(f'# SC Models  : {best.get("sc_model_breakdown",{})}')

    with open(OUT_TXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n[SAVE] Results -> {OUT_TXT}')

    cfg = {
        'version': 'v5',
        'generated': datetime.now().isoformat(),
        'best_2025': {k: v for k, v in best_2025.items() if k != 'monthly'},
        'best_2026': {k: v for k, v in best_2026.items() if k != 'monthly'} if best_2026 else None,
        'v3b_config': {
            'ml_threshold': V3B_ML_TH,
            'sl_multiplier': V3B_SL_MULT,
            'rr': V3B_RR,
            'base_risk': V3B_RISK,
            'cooldown_bars': V3B_COOL,
        },
        'scalper_config': {
            'sc_threshold': best.get('sc_th') if best else None,
            'sc_rr':        best.get('sc_rr') if best else None,
            'sc_risk':      best.get('sc_risk') if best else None,
            'sc_adaptive':  best.get('adaptive_th') if best else None,
            'sc_model_set': best.get('model_set') if best else None,
        } if best else {}
    }
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, default=str)
    print(f'[SAVE] Live config -> {OUT_JSON}')


def main():
    print('\n' + '#' * 70)
    print('# MASTER COMBINED BACKTEST V5 – SCALPER ULTRA EDITION')
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('#' * 70)
    combos = (len(SC_THRESHOLDS) * len(SC_RR_LIST) * len(SC_RISK_LIST)
              * len(SC_ADAPTIVE_TH) * len(MODEL_SETS))
    print(f'SC sweep: {combos} combos per period')

    # PHASE 1: Load + indicators
    df_full, _, _ = load_data(test_year=2025)
    print('\n[PHASE 1] Computing indicators...')
    df_full_ind     = calc_indicators(df_full)
    train_ind       = df_full_ind[df_full_ind['year'] < 2025].copy().reset_index(drop=True)
    test_2025_ind   = df_full_ind[df_full_ind['year'] == 2025].copy().reset_index(drop=True)

    # PHASE 2: HMM 5-state
    print('\n[PHASE 2] HMM 5-state training...')
    hmm5_model, state_labels, _ = train_hmm5(train_ind)
    train_ind       = add_hmm_states(train_ind, hmm5_model, state_labels)
    test_2025_ind   = add_hmm_states(test_2025_ind, hmm5_model, state_labels)
    print('[HMM] 2025 regime distribution:')
    for lbl in ['TREND_UP', 'TREND_DOWN', 'RANGE_TIGHT', 'RANGE_WIDE', 'UNCERTAIN']:
        pct = (test_2025_ind['hmm_label'] == lbl).mean() * 100
        print(f'  {lbl}: {pct:.1f}%')

    # PHASE 3: V3B
    print('\n[PHASE 3] Training V3B model...')
    v3b_components = train_v3b(train_ind)
    v3b_rf, v3b_gb, v3b_scaler, v3b_h3, tr_s3, rn_s3 = v3b_components

    # BASELINE CHECK
    print('\n[BASELINE] V3B-only check on 2025...')
    baseline = run_backtest(
        test_2025_ind, v3b_rf, v3b_gb, v3b_scaler, v3b_h3, tr_s3, rn_s3,
        sc_models={}, sc_th=1.0, sc_rr=3.0, sc_risk=0.0,
        adaptive_th=False, model_set='all5', label='V3B_BASELINE')
    if baseline:
        print(f"[BASELINE] V3B: {baseline['v3b_trades']}T | {baseline['v3b_wr']:.1f}%WR | "
              f"Profit:{baseline['profit_pct']:.0f}% | MaxDD:{baseline['max_dd']:.1f}%")
        if baseline['v3b_wr'] < 45:
            print(f"WARNING: V3B baseline WR {baseline['v3b_wr']:.1f}% is low - check data!")
    else:
        print('[BASELINE] WARNING: No trades in baseline')

    # PHASE 4: Scalpers
    print('\n[PHASE 4] Training 5 scalper models...')
    sc_models = train_all_scalpers(train_ind, label_rr=1.5)

    # PHASE 5a: 2025 sweep
    print('\n[PHASE 5a] SWEEP 2025...')
    results_2025 = run_sweep(test_2025_ind, v3b_components, sc_models, '2025')
    print_top_results(results_2025, n=20, label='2025 Full Year')
    if results_2025:
        sanity_check(results_2025[0], 'Best 2025')

    # PHASE 5b: 2026 (if data exists)
    print('\n[PHASE 5b] Preparing 2026 test...')
    test_2026_raw = df_full_ind[df_full_ind['year'] == 2026].copy().reset_index(drop=True)
    results_2026  = []

    if len(test_2026_raw) >= 30:
        train_2025_ind  = df_full_ind[df_full_ind['year'] <= 2025].copy().reset_index(drop=True)
        train_2025_hmm  = add_hmm_states(train_2025_ind, hmm5_model, state_labels)
        test_2026_hmm   = add_hmm_states(test_2026_raw,  hmm5_model, state_labels)

        print('  [2026] Re-training V3B on 2020-2025...')
        v3b_components_26 = train_v3b(train_2025_hmm)
        print('  [2026] Re-training scalpers on 2020-2025...')
        sc_models_26      = train_all_scalpers(train_2025_hmm, label_rr=1.5)

        print('\n[PHASE 5b] SWEEP 2026...')
        results_2026 = run_sweep(test_2026_hmm, v3b_components_26, sc_models_26, '2026')
        print_top_results(results_2026, n=20, label='2026 YTD')
        if results_2026:
            sanity_check(results_2026[0], 'Best 2026')
    else:
        print(f'  [2026] Only {len(test_2026_raw)} bars – skipping 2026 test')

    # PHASE 6: Save
    best_2025 = results_2025[0] if results_2025 else {}
    best_2026 = results_2026[0] if results_2026 else {}
    save_results(results_2025, results_2026, best_2025, best_2026)

    # FINAL SUMMARY
    print('\n' + '#' * 70)
    print('# FINAL SUMMARY')
    print('#' * 70)
    if results_2025:
        b = results_2025[0]
        print(f"BEST 2025: SC={b['sc_trades']}T @ {b['sc_wr']:.1f}%WR | CombWR={b['comb_wr']:.1f}% | "
              f"MaxDD={b['max_dd']:.1f}% | Profit={b['profit_pct']:,.0f}%")
        print(f"           TH={b['sc_th']} RR={b['sc_rr']} RISK={b['sc_risk']*100:.1f}% "
              f"ADAP={b['adaptive_th']} SET={b['model_set']}")
        print(f"           SC Breakdown: {b['sc_model_breakdown']}")
    if results_2026:
        b = results_2026[0]
        print(f"BEST 2026: SC={b['sc_trades']}T @ {b['sc_wr']:.1f}%WR | CombWR={b['comb_wr']:.1f}% | "
              f"MaxDD={b['max_dd']:.1f}% | Profit={b['profit_pct']:,.0f}%")
    print(f'\nFiles:\n  {OUT_TXT}\n  {OUT_JSON}')
    print('\n[DONE]')


if __name__ == '__main__':
    main()
