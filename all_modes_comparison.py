# -*- coding: utf-8 -*-
"""
ALL MODES COMPARISON - V3B vs SC vs Combined
=============================================
Tests 4 modes:
  MODE A: V3B only (no SC)
  MODE B: V3B+SC current (SC blocks V3B - original code)
  MODE C: V3B priority (V3B can open even if SC is open - SC gets closed)
  MODE D: Parallel (V3B and SC can both be open simultaneously)

For both 2025 and 2026.
"""
import warnings
warnings.filterwarnings('ignore')
import sys, os, time
import numpy as np
import pandas as pd

from master_combined_v5 import (
    load_data, calc_indicators, train_hmm5, add_hmm_states,
    train_v3b, train_all_scalpers,
    precompute_backtest_features, run_backtest,
    ACTIVE_SC_MODELS, REGIME_TO_MODELS,
    V3B_ML_TH, V3B_SL_MULT, V3B_RR, V3B_RISK, V3B_COOL, STARTING_BALANCE
)

OUT = os.path.join(os.path.dirname(__file__), 'all_modes_comparison.txt')
f = open(OUT, 'w', encoding='utf-8')
def log(s=''):
    f.write(s + '\n'); f.flush(); print(s)

log("=" * 80)
log("ALL MODES COMPARISON - V3B vs SC vs Combined")
log("=" * 80)
log()

# SC config (1m-verified optimal)
SC_TH = 0.42; SC_RR = 6.0; SC_RISK = 0.03; SC_SL_MULT = 0.8
SC_ADAPTIVE = True; SC_MODEL_SET = 'mom_burst_only'

def get_adaptive_risk_v3b(balance, peak):
    dd = (peak - balance) / peak if peak > 0 else 0
    if dd > 0.20: return V3B_RISK * 0.5
    elif dd > 0.10: return V3B_RISK * 0.75
    return V3B_RISK

def get_adaptive_risk_sc(balance, peak):
    dd = (peak - balance) / peak if peak > 0 else 0
    if dd > 0.15: return SC_RISK * 0.5
    elif dd > 0.05: return SC_RISK * 0.75
    return SC_RISK

def get_eff_th(atr_rank):
    if not SC_ADAPTIVE or pd.isna(atr_rank): return SC_TH
    if atr_rank < 0.40: return max(SC_TH - 0.05, 0.35)
    elif atr_rank > 0.70: return min(SC_TH + 0.05, 0.90)
    return SC_TH

# ============================================================
# CUSTOM BACKTEST ENGINE supporting all 4 modes
# ============================================================
def run_mode(df, sc_probs, mode='A'):
    """
    mode='A': V3B only
    mode='B': V3B+SC original (only 1 trade at a time, SC blocks V3B)
    mode='C': V3B priority (V3B can open even if SC open; SC closed on V3B entry)
    mode='D': Parallel (both can be open at same time)
    """
    n = len(df)
    active_models = ACTIVE_SC_MODELS.get(SC_MODEL_SET, [])

    balance = STARTING_BALANCE
    peak = STARTING_BALANCE
    max_dd = 0.0

    # V3B state
    v3b_in = False; v3b_entry = 0; v3b_sl = 0; v3b_tp = 0
    v3b_dir = ''; v3b_sl_dist = 0; v3b_risk = 0; v3b_bar = -999
    v3b_last_exit = -999

    # SC state
    sc_in = False; sc_entry = 0; sc_sl = 0; sc_tp = 0
    sc_dir = ''; sc_sl_dist = 0; sc_risk = 0; sc_bar = -999; sc_be_done = False
    sc_last_exit = -999

    v3b_stats = {'trades':0,'wins':0,'profit':0.0}
    sc_stats = {'trades':0,'wins':0,'profit':0.0,'tp':0,'sl':0,'be':0,'timeout':0}

    for i in range(60, n):
        row = df.iloc[i]
        hi, lo = row['h'], row['l']

        if balance > peak: peak = balance
        dd = (peak - balance) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd

        # ---- Process V3B trade exit ----
        if v3b_in:
            v3b_exit = None; v3b_reason = None
            if v3b_dir == 'LONG':
                if hi >= v3b_tp: v3b_exit, v3b_reason = v3b_tp, 'TP'
                elif lo <= v3b_sl: v3b_exit, v3b_reason = v3b_sl, 'SL'
            else:
                if lo <= v3b_tp: v3b_exit, v3b_reason = v3b_tp, 'TP'
                elif hi >= v3b_sl: v3b_exit, v3b_reason = v3b_sl, 'SL'
            if v3b_exit is None and (i - v3b_bar) >= 60:
                v3b_exit, v3b_reason = row['c'], 'TIMEOUT'
            if v3b_exit is not None:
                pnl_r = (v3b_exit - v3b_entry)/v3b_sl_dist if v3b_dir=='LONG' else (v3b_entry - v3b_exit)/v3b_sl_dist
                pnl = v3b_risk * pnl_r
                balance += pnl
                v3b_stats['trades'] += 1; v3b_stats['profit'] += pnl
                if pnl > 0: v3b_stats['wins'] += 1
                v3b_in = False; v3b_last_exit = i

        # ---- Process SC trade exit ----
        if sc_in:
            sc_exit = None; sc_reason = None
            be_just_set = False
            if not sc_be_done and sc_sl_dist > 0:
                if sc_dir == 'LONG' and hi >= sc_entry + 2.0*sc_sl_dist:
                    sc_sl = sc_entry + sc_sl_dist; sc_be_done = True; be_just_set = True
                elif sc_dir == 'SHORT' and lo <= sc_entry - 2.0*sc_sl_dist:
                    sc_sl = sc_entry - sc_sl_dist; sc_be_done = True; be_just_set = True
            if sc_dir == 'LONG':
                if hi >= sc_tp: sc_exit, sc_reason = sc_tp, 'TP'
                elif lo <= sc_sl and not be_just_set: sc_exit, sc_reason = sc_sl, ('BE' if sc_be_done else 'SL')
            else:
                if lo <= sc_tp: sc_exit, sc_reason = sc_tp, 'TP'
                elif hi >= sc_sl and not be_just_set: sc_exit, sc_reason = sc_sl, ('BE' if sc_be_done else 'SL')
            max_hold = min(int(SC_RR*12), 48)
            if sc_exit is None and (i - sc_bar) >= max_hold:
                sc_exit, sc_reason = row['c'], 'TIMEOUT'
            if sc_exit is not None:
                pnl_r = (sc_exit - sc_entry)/sc_sl_dist if sc_dir=='LONG' else (sc_entry - sc_exit)/sc_sl_dist
                pnl = sc_risk * pnl_r
                balance += pnl
                sc_stats['trades'] += 1; sc_stats['profit'] += pnl
                if pnl > 0: sc_stats['wins'] += 1
                r = sc_reason.lower()
                if r in sc_stats: sc_stats[r] += 1
                sc_in = False; sc_last_exit = i

        # ---- V3B entry ----
        atr = row['atr14']
        if pd.isna(atr) or atr <= 0: continue

        v3b_prob = df['_v3b_prob'].iloc[i]
        v3b_d = df['_v3b_dir'].iloc[i]
        v3b_fires = (not pd.isna(v3b_prob)) and (v3b_prob >= V3B_ML_TH) and (v3b_d != '')

        can_v3b = False
        if mode == 'A':
            can_v3b = not v3b_in and (i - v3b_last_exit) > V3B_COOL
        elif mode == 'B':
            can_v3b = not v3b_in and not sc_in and (i - max(v3b_last_exit, sc_last_exit)) > V3B_COOL
        elif mode == 'C':
            can_v3b = not v3b_in and (i - v3b_last_exit) > V3B_COOL
        elif mode == 'D':
            can_v3b = not v3b_in and (i - v3b_last_exit) > V3B_COOL

        if v3b_fires and can_v3b:
            # Mode C: close SC if open
            if mode == 'C' and sc_in:
                close_price = row['c']
                pnl_r = (close_price - sc_entry)/sc_sl_dist if sc_dir=='LONG' else (sc_entry - close_price)/sc_sl_dist
                pnl = sc_risk * pnl_r
                balance += pnl
                sc_stats['trades'] += 1; sc_stats['profit'] += pnl
                if pnl > 0: sc_stats['wins'] += 1
                sc_in = False; sc_last_exit = i

            v3b_in = True; v3b_dir = v3b_d
            v3b_entry = row['c']; v3b_sl_dist = atr * V3B_SL_MULT
            v3b_sl = v3b_entry - v3b_sl_dist if v3b_d=='LONG' else v3b_entry + v3b_sl_dist
            v3b_tp = v3b_entry + v3b_sl_dist*V3B_RR if v3b_d=='LONG' else v3b_entry - v3b_sl_dist*V3B_RR
            v3b_risk = get_adaptive_risk_v3b(balance, peak) * balance
            v3b_bar = i
            continue

        # ---- SC entry ----
        if mode == 'A': continue  # No SC in mode A

        can_sc = False
        if mode == 'B':
            can_sc = not v3b_in and not sc_in and (i - max(v3b_last_exit, sc_last_exit)) > V3B_COOL
        elif mode == 'C':
            can_sc = not sc_in and not v3b_in and (i - sc_last_exit) > V3B_COOL
        elif mode == 'D':
            can_sc = not sc_in and (i - sc_last_exit) > V3B_COOL

        if not can_sc: continue
        if v3b_fires: continue  # V3B always has priority

        sc_d = df['_sc_dir'].iloc[i] if df['_sc_dir'].iloc[i] != '' else 'LONG'
        hmm_lbl = str(df['hmm_label'].iloc[i]) if 'hmm_label' in df.columns else 'UNCERTAIN'
        regime_models = REGIME_TO_MODELS.get(hmm_lbl, [])
        candidates = [m for m in active_models if m in regime_models or SC_MODEL_SET=='mom_burst_only']
        if not candidates: candidates = active_models

        atr_rank = df['atr_rank'].iloc[i] if 'atr_rank' in df.columns else 0.5
        eff_th = get_eff_th(atr_rank)

        best_prob = 0.0; best_model = None
        for mname in candidates:
            if mname in sc_probs:
                p = sc_probs[mname][i]
                if p > best_prob: best_prob = p; best_model = mname

        if best_prob >= eff_th and best_model is not None:
            sc_in = True; sc_dir = sc_d; sc_be_done = False
            sc_entry = row['c']; sc_sl_dist = atr * SC_SL_MULT
            sc_sl = sc_entry - sc_sl_dist if sc_d=='LONG' else sc_entry + sc_sl_dist
            sc_tp = sc_entry + sc_sl_dist*SC_RR if sc_d=='LONG' else sc_entry - sc_sl_dist*SC_RR
            sc_risk = get_adaptive_risk_sc(balance, peak) * balance
            sc_bar = i

    total_trades = v3b_stats['trades'] + sc_stats['trades']
    total_wins = v3b_stats['wins'] + sc_stats['wins']
    total_return = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100
    v3b_wr = (v3b_stats['wins']/v3b_stats['trades']*100) if v3b_stats['trades']>0 else 0
    sc_wr = (sc_stats['wins']/sc_stats['trades']*100) if sc_stats['trades']>0 else 0
    overall_wr = (total_wins/total_trades*100) if total_trades>0 else 0

    return {
        'mode': mode, 'balance': balance, 'return': total_return, 'max_dd': max_dd*100,
        'total_trades': total_trades, 'overall_wr': overall_wr,
        'v3b_trades': v3b_stats['trades'], 'v3b_wr': v3b_wr, 'v3b_profit': v3b_stats['profit'],
        'sc_trades': sc_stats['trades'], 'sc_wr': sc_wr, 'sc_profit': sc_stats['profit'],
        'sc_tp': sc_stats.get('tp',0), 'sc_sl': sc_stats.get('sl',0),
        'sc_be': sc_stats.get('be',0), 'sc_timeout': sc_stats.get('timeout',0),
    }


def run_year(test_year):
    log(f"\n{'='*80}")
    log(f"  TESTING YEAR {test_year}")
    log(f"{'='*80}")

    _, train_df, test_df = load_data(test_year=test_year)
    train_df = calc_indicators(train_df)
    test_df = calc_indicators(test_df)

    hmm5, labels, _ = train_hmm5(train_df)
    train_df = add_hmm_states(train_df, hmm5, labels)
    test_df = add_hmm_states(test_df, hmm5, labels)

    v3b_rf, v3b_gb, v3b_scaler, v3b_h3, tr_s3, rn_s3 = train_v3b(train_df)
    sc_models = train_all_scalpers(train_df, label_rr=2.0)
    df_pre, sc_probs = precompute_backtest_features(
        test_df, v3b_rf, v3b_gb, v3b_scaler, v3b_h3, tr_s3, rn_s3, sc_models)

    results = {}
    for mode in ['A', 'B', 'C', 'D']:
        log(f"\n  Running Mode {mode}...")
        r = run_mode(df_pre, sc_probs, mode)
        results[mode] = r
        log(f"    V3B: {r['v3b_trades']}T {r['v3b_wr']:.1f}%WR ${r['v3b_profit']:,.0f}")
        log(f"    SC:  {r['sc_trades']}T {r['sc_wr']:.1f}%WR ${r['sc_profit']:,.0f}")
        log(f"    Total: +{r['return']:,.0f}% | DD: {r['max_dd']:.1f}% | ${r['balance']:,.0f}")

    return results


# ============================================================
# RUN BOTH YEARS
# ============================================================
results_2025 = run_year(2025)
results_2026 = run_year(2026)

# ============================================================
# FINAL COMPARISON TABLE
# ============================================================
log("\n" + "=" * 80)
log("FINAL COMPARISON TABLE")
log("=" * 80)

mode_names = {
    'A': 'V3B ONLY',
    'B': 'V3B+SC Original (SC blocks V3B)',
    'C': 'V3B Priority (V3B can interrupt SC)',
    'D': 'Parallel (both open simultaneously)',
}

for year, results in [('2025', results_2025), ('2026', results_2026)]:
    log(f"\n--- {year} ---")
    log(f"{'Mode':<45} {'V3B_T':>6} {'V3B_WR':>7} {'V3B$':>12} {'SC_T':>5} {'SC_WR':>6} {'SC$':>12} {'Total%':>10} {'DD':>6} {'Balance':>14}")
    log("-" * 130)
    for mode in ['A', 'B', 'C', 'D']:
        r = results[mode]
        log(f"{mode_names[mode]:<45} {r['v3b_trades']:>6} {r['v3b_wr']:>6.1f}% {r['v3b_profit']:>12,.0f} "
            f"{r['sc_trades']:>5} {r['sc_wr']:>5.1f}% {r['sc_profit']:>12,.0f} "
            f"{r['return']:>9,.0f}% {r['max_dd']:>5.1f}% {r['balance']:>14,.0f}")

log("\n" + "=" * 80)
log("RECOMMENDATIONS")
log("=" * 80)

# Find best mode for each year
for year, results in [('2025', results_2025), ('2026', results_2026)]:
    best_mode = max(results, key=lambda m: results[m]['return'])
    best = results[best_mode]
    log(f"\n{year} BEST: Mode {best_mode} ({mode_names[best_mode]})")
    log(f"  Return: +{best['return']:,.0f}% | Balance: ${best['balance']:,.0f} | DD: {best['max_dd']:.1f}%")
    log(f"  V3B: {best['v3b_trades']}T {best['v3b_wr']:.1f}%WR | SC: {best['sc_trades']}T {best['sc_wr']:.1f}%WR")

# Overall recommendation
best_2025 = max(results_2025, key=lambda m: results_2025[m]['return'])
best_2026 = max(results_2026, key=lambda m: results_2026[m]['return'])

log(f"\nOVERALL: Best for 2025 = Mode {best_2025}, Best for 2026 = Mode {best_2026}")
if best_2025 == best_2026:
    log(f"CONSISTENT: Mode {best_2025} ({mode_names[best_2025]}) is best in BOTH years!")
else:
    log(f"MIXED: Different modes work best in different years.")
    log(f"  Safest choice: Mode {best_2025} (proven on full year 2025)")

f.close()
print(f"\nResults saved to: {OUT}")
