# -*- coding: utf-8 -*-
"""
BEST SCALPER CONFIGURATION - SAVED
==================================
1-MINUTE VERIFIED optimal settings (108k 1m candles, 5400 combos tested)
ALL exits verified on real 1-minute AsterDex data - NO guesswork
"""

# =====================================================================
# BEST OVERALL (1716% return, 67 SC trades, 53.7% SC WR)
# =====================================================================
BEST_SCALPER_CONFIG = {
    'SC_THRESHOLD': 0.420,
    'SC_RR': 6.0,
    'SC_RISK': 0.030,  # 3.0%
    'SC_SL_MULT': 0.8,
    'SC_ADAPTIVE_TH': True,
    'SC_MODEL_SET': 'mom_burst_only',
}

# =====================================================================
# BEST WIN RATE (61.9% WR, 63 SC trades, 1:3 RR)
# =====================================================================
BEST_WR_CONFIG = {
    'SC_THRESHOLD': 0.400,
    'SC_RR': 3.0,
    'SC_RISK': 0.030,
    'SC_SL_MULT': 1.5,
    'SC_ADAPTIVE_TH': True,
    'SC_MODEL_SET': 'mom_burst_only',
}

# =====================================================================
# CONSERVATIVE (59.1% WR, 22 SC trades, fewer but cleaner)
# =====================================================================
CONSERVATIVE_CONFIG = {
    'SC_THRESHOLD': 0.500,
    'SC_RR': 4.0,
    'SC_RISK': 0.030,
    'SC_SL_MULT': 0.5,
    'SC_ADAPTIVE_TH': True,
    'SC_MODEL_SET': 'trend_range',
}

# V3B DEFAULT (unchanged)
DEFAULT_V3B_CONFIG = {
    'V3B_ML_TH': 0.455,
    'V3B_RR': 3.0,
    'V3B_RISK': 0.030,
    'V3B_SL_MULT': 1.0,
    'V3B_COOL': 2,
}

print("Best Scalper configuration saved:")
for k, v in BEST_SCALPER_CONFIG.items():
    print(f"  {k}: {v}")
