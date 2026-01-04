"""
🥇 TERMINATOR GENESIS TRADING ENGINE - GOLD EDITION 🥇
Advanced HFT Algorithm with Hidden Markov Models, Multi-Timeframe Analysis, and AI Pattern Recognition
Optimized for XAU/USD (Gold) Trading
Python 3.11 | Fully Autonomous | Self-Learning | 24/7 Cloud Compatible

WARNING: High-risk trading bot. Use at your own risk. Past performance does not guarantee future results.
"""

import asyncio
import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime, timedelta
import json
import os
from scipy.stats import entropy
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import logging
from telegram import Bot
from telegram.error import TelegramError
import aiosqlite
import time


# ==================== CONFIGURATION (GOLD OPTIMIZED) ====================
class Config:
    # EXCHANGE SETTINGS - FOREX/CFD Compatible
    EXCHANGE = "oanda"  # Forex broker (ccxt compatible) - later Vantage
    SYMBOL = "XAU/USD"  # Gold vs US Dollar
    API_KEY = os.environ.get('OANDA_API_KEY')  # Or VANTAGE_API_KEY later
    API_SECRET = os.environ.get('OANDA_API_SECRET')
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID_GOLD')  # Separate group for Gold
    TESTNET = True
    PAPER_TRADING = True  # Enable paper trading mode (no real trades)

    # RISK MANAGEMENT (Adjusted for Gold's lower volatility)
    ACCOUNT_RISK_PERCENT = 1.0  # Base risk per trade
    MAX_POSITION_SIZE = 0.1  # Max 10% of account
    TERMINATOR_MULTIPLIER = 2.5  # Aggressive mode multiplier

    # TIMEFRAMES - EXTENDED FOR SMC MULTI-TIMEFRAME ANALYSIS
    # All main timeframes for proper market structure analysis
    TIMEFRAMES = {
        "scalp": "15m",      # Scalp entries, FVG detection
        "intraday": "30m",   # Intraday structure
        "medium": "1h",      # Medium-term structure
        "swing": "4h",       # Swing structure, main CHoCH/BOS
        "daily": "1d"        # Daily bias, major structure
    }
    
    # SMC (SMART MONEY CONCEPTS) CONFIGURATION
    ZIGZAG_DEPTH = 5              # Min candles between swing points
    ZIGZAG_DEVIATION = 0.001      # Min 0.1% move for swing ($2.60 at $2600)
    CHOCH_LOOKBACK = 100          # Candles to look back for structure
    FVG_MIN_GAP_PCT = 0.0005      # Min 0.05% gap for FVG ($1.30 at $2600)
    
    # STRICT SL LIMITS (USD) - PRECISE LEVELS!
    SL_LIMITS = {
        "SCALP": 6,      # Max $6 SL for scalp trades
        "STANDARD": 10,  # Max $10 SL for normal trades
        "SWING": 25      # Max $25 SL for swing trades
    }

    # THRESHOLDS (GOLD-SPECIFIC - lower volatility asset)
    MIN_SCORE = 60  # Raised for quality over quantity
    TERMINATOR_SCORE = 80  # Ultra-high confidence threshold
    HMM_BEAR_THRESHOLD = 0.80  # 80% probability to cancel longs
    HMM_BULL_THRESHOLD = 0.80  # 80% probability to cancel shorts
    FLASH_CRASH_THRESHOLD = 0.005  # 0.5% drop in 1 second (Gold is less volatile)
    ENTROPY_THRESHOLD = 0.25  # Lower chaos threshold for Gold
    ADX_THRESHOLD = 20  # Lower trend strength minimum (Gold trends smoother)
    MIN_ACTIVE_ENGINES = 2  # Raised back to 2 for quality signals

    # FLEXIBLE RR TIER CONFIGURATION
    # SCALP: 60-69 score -> 1:3-1:5 RR (quick trades, max $6 SL)
    # STANDARD: 70-79 score -> 1:5-1:8 RR (regular trades, max $10 SL)  
    # SWING: 80+ score -> 1:8-1:10 RR (high confidence, max $25 SL)
    RR_TIERS = {
        "SCALP": {"min_score": 60, "max_score": 69, "sl_mult": 1.0, "tp1_mult": 3.0, "tp2_mult": 5.0, "tp3_mult": None, "max_rr": 5},
        "STANDARD": {"min_score": 70, "max_score": 79, "sl_mult": 1.5, "tp1_mult": 4.5, "tp2_mult": 7.5, "tp3_mult": 12.0, "max_rr": 8},
        "SWING": {"min_score": 80, "max_score": 100, "sl_mult": 2.0, "tp1_mult": 6.0, "tp2_mult": 10.0, "tp3_mult": 16.0, "max_rr": 10}
    }

    # PATTERN RECOGNITION
    PATTERN_HISTORY_CANDLES = 30
    PATTERN_DATABASE_SIZE = 5000
    PATTERN_SIMILARITY_THRESHOLD = 0.85

    # DATABASE
    DB_PATH = "terminator_gold_data.db"
    BLACKLIST_PATH = "blacklist_gold.json"

    # GOLD-SPECIFIC: Trading Sessions (Gold has specific active hours)
    # London session (8:00-16:00 UTC) - Most active for Gold
    # NY session (13:00-21:00 UTC) - Also very active
    # Asian session - Less volatile but still tradeable


# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.FileHandler('terminator_gold.log'),
              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ==================== KEEP ALIVE SERVER (REPLIT COMPATIBILITY) ====================
from flask import Flask
from threading import Thread

app = Flask('')


@app.route('/')
def home():
    return "🥇 TERMINATOR GENESIS GOLD EDITION IS ONLINE 🥇"


def run_flask():
    app.run(host='0.0.0.0', port=8000)  # Koyeb default port


def keep_alive():
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()
    logger.info("🌐 Flask Keep-Alive Server Started on Port 8000 (GOLD)")


# ==================== HIDDEN MARKOV MODEL (ORACLE BRAIN) ====================
class HiddenMarkovModel:
    """Simplified HMM for regime detection (Bull/Bear/Sideways)"""

    def __init__(self):
        # States: 0=Bull, 1=Bear, 2=Sideways
        self.states = ['BULL', 'BEAR', 'SIDEWAYS']
        self.n_states = 3

        # Transition matrix (GOLD-OPTIMIZED: Gold tends to trend more smoothly)
        self.transition_matrix = np.array([
            [0.75, 0.10, 0.15],  # From Bull - Gold trends longer
            [0.10, 0.75, 0.15],  # From Bear - Gold trends longer
            [0.25, 0.25, 0.50]   # From Sideways - Gold consolidates more
        ])

        self.current_state = 0  # Start in Bull

    def calculate_state_probabilities(
            self, price_changes: List[float]) -> Dict[str, float]:
        """Calculate probability of each regime based on recent price action"""
        if len(price_changes) < 10:
            return {state: 1 / 3 for state in self.states}

        # Simplified observation model (GOLD-ADJUSTED thresholds)
        recent_volatility = np.std(price_changes[-20:])
        recent_trend = np.mean(price_changes[-10:])

        # Calculate likelihoods (Lower volatility thresholds for Gold)
        bull_likelihood = 1.0 if recent_trend > 0 and recent_volatility < 0.01 else 0.3
        bear_likelihood = 1.0 if recent_trend < 0 and recent_volatility < 0.01 else 0.3
        sideways_likelihood = 1.0 if abs(recent_trend) < 0.0005 else 0.4

        # Current state probabilities
        current_probs = np.zeros(self.n_states)
        current_probs[self.current_state] = 1.0

        # Forward algorithm (simplified)
        next_probs = current_probs @ self.transition_matrix

        # Combine with observations
        likelihoods = np.array(
            [bull_likelihood, bear_likelihood, sideways_likelihood])
        posterior = next_probs * likelihoods
        posterior = posterior / posterior.sum()

        return {
            self.states[i]: float(posterior[i])
            for i in range(self.n_states)
        }


# ==================== SMC (SMART MONEY CONCEPTS) ENGINE ====================
class SMCEngine:
    """Smart Money Concepts: ZigZag, CHoCH, BOS, FVG detection for professional trading"""
    
    def __init__(self, config: Config):
        self.config = config
        self.structure_cache = {}  # Cache for multi-timeframe structure
    
    def find_zigzag_pivots(self, df: pd.DataFrame) -> List[Dict]:
        """
        ZigZag Algorithm - Find swing highs and lows
        Returns list of pivots: [{'type': 'HIGH'/'LOW', 'price': float, 'index': int, 'timestamp': datetime}]
        """
        if len(df) < self.config.ZIGZAG_DEPTH * 2:
            return []
        
        pivots = []
        depth = self.config.ZIGZAG_DEPTH
        deviation = self.config.ZIGZAG_DEVIATION
        
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        
        # Find swing highs
        for i in range(depth, len(df) - depth):
            # Check if this is a swing high (highest high in window)
            window_highs = highs[i - depth:i + depth + 1]
            if highs[i] == max(window_highs):
                # Check deviation from surrounding
                left_low = min(lows[i - depth:i])
                right_low = min(lows[i:i + depth + 1])
                
                if (highs[i] - left_low) / left_low >= deviation:
                    pivots.append({
                        'type': 'HIGH',
                        'price': float(highs[i]),
                        'index': i,
                        'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i
                    })
        
        # Find swing lows
        for i in range(depth, len(df) - depth):
            # Check if this is a swing low (lowest low in window)
            window_lows = lows[i - depth:i + depth + 1]
            if lows[i] == min(window_lows):
                # Check deviation from surrounding
                left_high = max(highs[i - depth:i])
                right_high = max(highs[i:i + depth + 1])
                
                if (left_high - lows[i]) / lows[i] >= deviation:
                    pivots.append({
                        'type': 'LOW',
                        'price': float(lows[i]),
                        'index': i,
                        'timestamp': df.iloc[i]['timestamp'] if 'timestamp' in df.columns else i
                    })
        
        # Sort by index
        pivots.sort(key=lambda x: x['index'])
        
        # Remove duplicate pivots (same type in sequence)
        filtered_pivots = []
        last_type = None
        for pivot in pivots:
            if pivot['type'] != last_type:
                filtered_pivots.append(pivot)
                last_type = pivot['type']
            else:
                # Keep the more extreme one
                if filtered_pivots:
                    if pivot['type'] == 'HIGH' and pivot['price'] > filtered_pivots[-1]['price']:
                        filtered_pivots[-1] = pivot
                    elif pivot['type'] == 'LOW' and pivot['price'] < filtered_pivots[-1]['price']:
                        filtered_pivots[-1] = pivot
        
        return filtered_pivots
    
    def detect_market_structure(self, df: pd.DataFrame) -> Dict:
        """
        Detect CHoCH (Change of Character) and BOS (Break of Structure)
        
        Returns:
        {
            'trend': 'BULLISH' / 'BEARISH' / 'RANGING',
            'choch': {'detected': bool, 'type': 'BULLISH'/'BEARISH', 'price': float},
            'bos': {'detected': bool, 'type': 'BULLISH'/'BEARISH', 'price': float},
            'last_swing_high': float,
            'last_swing_low': float,
            'structure_break_level': float  # Key level for direction filter
        }
        """
        pivots = self.find_zigzag_pivots(df)
        
        result = {
            'trend': 'RANGING',
            'choch': {'detected': False, 'type': None, 'price': None},
            'bos': {'detected': False, 'type': None, 'price': None},
            'last_swing_high': None,
            'last_swing_low': None,
            'structure_break_level': None,
            'pivots': pivots
        }
        
        if len(pivots) < 4:
            return result
        
        # Get last few swing points
        swing_highs = [p for p in pivots if p['type'] == 'HIGH'][-3:]
        swing_lows = [p for p in pivots if p['type'] == 'LOW'][-3:]
        
        if not swing_highs or not swing_lows:
            return result
        
        result['last_swing_high'] = swing_highs[-1]['price']
        result['last_swing_low'] = swing_lows[-1]['price']
        
        current_price = float(df['close'].iloc[-1])
        
        # Determine trend from higher highs/lower lows
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]['price'] > swing_highs[-2]['price']  # Higher High
            hl = swing_lows[-1]['price'] > swing_lows[-2]['price']    # Higher Low
            lh = swing_highs[-1]['price'] < swing_highs[-2]['price']  # Lower High
            ll = swing_lows[-1]['price'] < swing_lows[-2]['price']    # Lower Low
            
            if hh and hl:
                result['trend'] = 'BULLISH'
            elif lh and ll:
                result['trend'] = 'BEARISH'
        
        # Detect CHoCH (Change of Character)
        # BULLISH CHoCH: Price breaks above last swing high after downtrend
        # BEARISH CHoCH: Price breaks below last swing low after uptrend
        
        if result['trend'] == 'BEARISH':
            # Look for bullish CHoCH (break of last swing high)
            if current_price > result['last_swing_high']:
                result['choch'] = {
                    'detected': True,
                    'type': 'BULLISH',
                    'price': result['last_swing_high']
                }
                result['structure_break_level'] = result['last_swing_low']
                
        elif result['trend'] == 'BULLISH':
            # Look for bearish CHoCH (break of last swing low)
            if current_price < result['last_swing_low']:
                result['choch'] = {
                    'detected': True,
                    'type': 'BEARISH',
                    'price': result['last_swing_low']
                }
                result['structure_break_level'] = result['last_swing_high']
        
        # Detect BOS (Break of Structure) - trend continuation
        if result['trend'] == 'BULLISH' and not result['choch']['detected']:
            if current_price > result['last_swing_high']:
                result['bos'] = {
                    'detected': True,
                    'type': 'BULLISH',
                    'price': result['last_swing_high']
                }
                result['structure_break_level'] = result['last_swing_low']
                
        elif result['trend'] == 'BEARISH' and not result['choch']['detected']:
            if current_price < result['last_swing_low']:
                result['bos'] = {
                    'detected': True,
                    'type': 'BEARISH',
                    'price': result['last_swing_low']
                }
                result['structure_break_level'] = result['last_swing_high']
        
        return result
    
    def find_fvg_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Find Fair Value Gaps (FVG) - Imbalance zones
        
        BULLISH FVG: Gap between candle 1 high and candle 3 low (3 candle pattern)
        BEARISH FVG: Gap between candle 1 low and candle 3 high
        
        Returns list of FVG zones for potential entries/targets
        """
        fvg_zones = []
        min_gap = self.config.FVG_MIN_GAP_PCT
        
        if len(df) < 5:
            return fvg_zones
        
        for i in range(2, len(df)):
            candle1_high = df['high'].iloc[i - 2]
            candle1_low = df['low'].iloc[i - 2]
            candle3_high = df['high'].iloc[i]
            candle3_low = df['low'].iloc[i]
            current_price = df['close'].iloc[-1]
            
            # BULLISH FVG: Candle 3 low > Candle 1 high (gap up)
            if candle3_low > candle1_high:
                gap_size = (candle3_low - candle1_high) / candle1_high
                if gap_size >= min_gap:
                    # Check if FVG is still valid (not filled)
                    zone_top = candle3_low
                    zone_bottom = candle1_high
                    
                    # FVG is valid if price hasn't filled it completely
                    if current_price > zone_bottom:
                        fvg_zones.append({
                            'type': 'BULLISH',
                            'top': float(zone_top),
                            'bottom': float(zone_bottom),
                            'midpoint': float((zone_top + zone_bottom) / 2),
                            'index': i,
                            'filled': current_price < zone_top  # Partially filled
                        })
            
            # BEARISH FVG: Candle 3 high < Candle 1 low (gap down)
            if candle3_high < candle1_low:
                gap_size = (candle1_low - candle3_high) / candle3_high
                if gap_size >= min_gap:
                    zone_top = candle1_low
                    zone_bottom = candle3_high
                    
                    if current_price < zone_top:
                        fvg_zones.append({
                            'type': 'BEARISH',
                            'top': float(zone_top),
                            'bottom': float(zone_bottom),
                            'midpoint': float((zone_top + zone_bottom) / 2),
                            'index': i,
                            'filled': current_price > zone_bottom
                        })
        
        # Return only recent valid FVGs (last 10)
        return fvg_zones[-10:]
    
    def get_smc_bias(self, multi_tf_structures: Dict) -> Dict:
        """
        Analyze multi-timeframe SMC structure to determine trading bias
        
        Returns:
        {
            'bias': 'LONG' / 'SHORT' / 'NEUTRAL',
            'confidence': 0-100,
            'long_allowed': bool,
            'short_allowed': bool,
            'reason': str
        }
        """
        # Get structure from each timeframe
        daily_struct = multi_tf_structures.get('daily', {})
        swing_struct = multi_tf_structures.get('swing', {})  # 4H
        medium_struct = multi_tf_structures.get('medium', {})  # 1H
        scalp_struct = multi_tf_structures.get('scalp', {})  # 15m
        
        result = {
            'bias': 'NEUTRAL',
            'confidence': 50,
            'long_allowed': True,
            'short_allowed': True,
            'reason': 'Analyzing structure...',
            'structures': multi_tf_structures
        }
        
        # RULE 1: Daily CHoCH overrides everything
        if daily_struct.get('choch', {}).get('detected'):
            choch_type = daily_struct['choch']['type']
            if choch_type == 'BEARISH':
                result['long_allowed'] = False
                result['bias'] = 'SHORT'
                result['reason'] = 'Daily CHoCH BEARISH - LONGS FORBIDDEN!'
                result['confidence'] = 85
            elif choch_type == 'BULLISH':
                result['short_allowed'] = False
                result['bias'] = 'LONG'
                result['reason'] = 'Daily CHoCH BULLISH - SHORTS FORBIDDEN!'
                result['confidence'] = 85
            return result
        
        # RULE 2: 4H CHoCH is major direction filter
        if swing_struct.get('choch', {}).get('detected'):
            choch_type = swing_struct['choch']['type']
            if choch_type == 'BEARISH':
                result['long_allowed'] = False
                result['bias'] = 'SHORT'
                result['reason'] = '4H CHoCH BEARISH - LONGS FORBIDDEN!'
                result['confidence'] = 80
            elif choch_type == 'BULLISH':
                result['short_allowed'] = False
                result['bias'] = 'LONG'
                result['reason'] = '4H CHoCH BULLISH - SHORTS FORBIDDEN!'
                result['confidence'] = 80
            return result
        
        # RULE 3: 4H BOS confirms trend continuation
        if swing_struct.get('bos', {}).get('detected'):
            bos_type = swing_struct['bos']['type']
            if bos_type == 'BULLISH':
                result['bias'] = 'LONG'
                result['reason'] = '4H BOS BULLISH - Trend continuation'
                result['confidence'] = 70
            elif bos_type == 'BEARISH':
                result['bias'] = 'SHORT'
                result['reason'] = '4H BOS BEARISH - Trend continuation'
                result['confidence'] = 70
        
        # RULE 4: 1H structure adds confidence
        if medium_struct.get('trend') == 'BULLISH' and result['bias'] == 'LONG':
            result['confidence'] += 10
            result['reason'] += ' | 1H confirms BULLISH'
        elif medium_struct.get('trend') == 'BEARISH' and result['bias'] == 'SHORT':
            result['confidence'] += 10
            result['reason'] += ' | 1H confirms BEARISH'
        
        # RULE 5: If no clear bias from higher TF, use 1H trend
        if result['bias'] == 'NEUTRAL':
            if medium_struct.get('trend') == 'BULLISH':
                result['bias'] = 'LONG'
                result['reason'] = '1H trend BULLISH'
                result['confidence'] = 60
            elif medium_struct.get('trend') == 'BEARISH':
                result['bias'] = 'SHORT'
                result['reason'] = '1H trend BEARISH'
                result['confidence'] = 60
        
        return result
    
    def calculate_structure_sl(self, direction: str, structure: Dict, current_price: float, tier: str) -> float:
        """
        Calculate precise SL based on swing structure levels
        SL should be placed just beyond the last swing low (for LONG) or swing high (for SHORT)
        """
        max_sl = self.config.SL_LIMITS.get(tier, 10)
        buffer = 1.0  # $1 buffer beyond structure
        
        if direction == 'LONG':
            # SL below last swing low
            swing_low = structure.get('last_swing_low')
            if swing_low:
                structure_sl = swing_low - buffer
                sl_distance = current_price - structure_sl
                
                # If structure SL is too far, cap it
                if sl_distance > max_sl:
                    structure_sl = current_price - max_sl
                    
                return float(structure_sl)
            else:
                return float(current_price - max_sl)
        
        else:  # SHORT
            # SL above last swing high
            swing_high = structure.get('last_swing_high')
            if swing_high:
                structure_sl = swing_high + buffer
                sl_distance = structure_sl - current_price
                
                # If structure SL is too far, cap it
                if sl_distance > max_sl:
                    structure_sl = current_price + max_sl
                    
                return float(structure_sl)
            else:
                return float(current_price + max_sl)


# ==================== TECHNICAL INDICATORS ENGINE ====================
class IndicatorEngine:
    """Calculate all technical indicators across multiple timeframes"""

    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all indicators to dataframe"""
        try:
            # Price-based
            df['returns'] = df['close'].pct_change()

            # Trend
            df.ta.sma(length=20, append=True)
            df.ta.ema(length=50, append=True)
            df.ta.ema(length=200, append=True)

            # Momentum
            df.ta.rsi(length=14, append=True)
            df.ta.macd(append=True)
            df.ta.adx(length=14, append=True)

            # Volatility
            df.ta.bbands(length=20, append=True)
            df.ta.atr(length=14, append=True)
            df.ta.kc(length=20, append=True)

            # Volume (Note: Forex volume is less reliable than crypto)
            df.ta.obv(append=True)
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # VWAP
            df['vwap'] = (df['volume'] *
                          (df['high'] + df['low'] + df['close']) /
                          3).cumsum() / df['volume'].cumsum()

            # TTM Squeeze
            df = IndicatorEngine.ttm_squeeze(df)

            return df
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
            return df

    @staticmethod
    def ttm_squeeze(df: pd.DataFrame) -> pd.DataFrame:
        """TTM Squeeze indicator"""
        try:
            # Bollinger Bands
            bb_length = 20
            bb_mult = 2
            bb_basis = df['close'].rolling(bb_length).mean()
            bb_dev = df['close'].rolling(bb_length).std()
            bb_upper = bb_basis + bb_mult * bb_dev
            bb_lower = bb_basis - bb_mult * bb_dev

            # Keltner Channels
            kc_length = 20
            kc_mult = 1.5
            kc_basis = df['close'].rolling(kc_length).mean()
            atr = df.ta.atr(length=kc_length)
            kc_upper = kc_basis + kc_mult * atr
            kc_lower = kc_basis - kc_mult * atr

            # Squeeze: BB inside KC
            df['squeeze_on'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)

            return df
        except:
            df['squeeze_on'] = False
            return df


# ==================== PATTERN RECOGNITION (AI) ====================
class PatternRecognition:
    """AI-powered pattern matching with self-learning"""

    def __init__(self, config: Config):
        self.config = config
        self.pattern_database = []
        self.blacklist = self.load_blacklist()

    def load_blacklist(self) -> List[str]:
        """Load failed patterns from file"""
        try:
            if os.path.exists(self.config.BLACKLIST_PATH):
                with open(self.config.BLACKLIST_PATH, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []

    def save_to_blacklist(self, pattern_hash: str):
        """Save failed pattern to blacklist"""
        if pattern_hash not in self.blacklist:
            self.blacklist.append(pattern_hash)
            try:
                with open(self.config.BLACKLIST_PATH, 'w') as f:
                    json.dump(self.blacklist, f)
            except:
                pass

    def normalize_pattern(self, prices: np.ndarray) -> np.ndarray:
        """Normalize price pattern to [-1, 1]"""
        if len(prices) < 2:
            return prices
        min_val, max_val = prices.min(), prices.max()
        if max_val - min_val == 0:
            return np.zeros_like(prices)
        return 2 * (prices - min_val) / (max_val - min_val) - 1

    def get_pattern_hash(self, pattern: np.ndarray) -> str:
        """Create unique hash for pattern"""
        return str(hash(pattern.tobytes()))[:16]

    def find_similar_patterns(
            self, current_pattern: np.ndarray,
            historical_data: pd.DataFrame) -> Tuple[float, int]:
        """Find similar patterns in history and return confidence score"""
        if len(current_pattern) < 10 or len(historical_data) < 100:
            return 0.0, 0

        current_normalized = self.normalize_pattern(current_pattern)
        pattern_hash = self.get_pattern_hash(current_normalized)

        # Check blacklist
        if pattern_hash in self.blacklist:
            logger.warning("⚠️ Pattern found in BLACKLIST - Skipping")
            return 0.0, 0

        similarities = []
        outcomes = []

        # Slide through historical data
        window_size = len(current_pattern)
        for i in range(len(historical_data) - window_size - 10):
            historical_window = historical_data['close'].iloc[
                i:i + window_size].values
            historical_normalized = self.normalize_pattern(historical_window)

            # Calculate correlation
            correlation = np.corrcoef(current_normalized,
                                      historical_normalized)[0, 1]

            if correlation > self.config.PATTERN_SIMILARITY_THRESHOLD:
                # Check what happened after this pattern
                future_return = (
                    historical_data['close'].iloc[i + window_size + 5] /
                    historical_data['close'].iloc[i + window_size] - 1)

                similarities.append(correlation)
                outcomes.append(1 if future_return > 0 else 0)

        if len(similarities) == 0:
            return 0.0, 0

        # Calculate confidence: weighted average of outcomes
        weights = np.array(similarities)
        confidence = np.average(outcomes, weights=weights) * 100

        return confidence, len(similarities)


# ==================== MARKET DATA MANAGER ====================
class MarketDataManager:
    """Manages real-time and historical market data"""

    def __init__(self, exchange: ccxt.Exchange, config: Config):
        self.exchange = exchange
        self.config = config
        self.last_prices = []
        self.last_timestamp = 0

    async def fetch_ohlcv(self,
                          timeframe: str,
                          limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV data"""
        try:
            ohlcv = await self.exchange.fetch_ohlcv(self.config.SYMBOL,
                                                    timeframe,
                                                    limit=limit)
            df = pd.DataFrame(ohlcv,
                              columns=[
                                  'timestamp', 'open', 'high', 'low', 'close',
                                  'volume'
                              ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Error fetching OHLCV {timeframe}: {e}")
            return pd.DataFrame()

    async def get_order_book_imbalance(self) -> float:
        """Calculate order book imbalance (OBI)"""
        try:
            orderbook = await self.exchange.fetch_order_book(
                self.config.SYMBOL, limit=20)
            bids = sum([bid[1] for bid in orderbook['bids']])
            asks = sum([ask[1] for ask in orderbook['asks']])

            if bids + asks == 0:
                return 0.0

            imbalance = (bids - asks) / (bids + asks)
            return imbalance
        except:
            return 0.0

    async def get_funding_rate(self) -> float:
        """GOLD: No funding rate in Forex spot trading - return 0"""
        # Funding rate is a crypto perpetual futures concept
        # For Forex/Gold spot trading, this doesn't exist
        return 0.0

    def detect_flash_crash(self, current_price: float) -> bool:
        """Detect flash crash (>0.5% drop in 1 second for Gold)"""
        self.last_prices.append((current_price, time.time()))

        # Keep only last 2 seconds of data
        cutoff = time.time() - 2
        self.last_prices = [(p, t) for p, t in self.last_prices if t > cutoff]

        if len(self.last_prices) < 2:
            return False

        # Check for rapid drop
        recent_high = max([p for p, _ in self.last_prices[-10:]])
        drop = (current_price - recent_high) / recent_high

        return drop < -self.config.FLASH_CRASH_THRESHOLD


# ==================== EXTERNAL MARKET CORRELATIONS ====================
class ExternalMarkets:
    """Fetch SPX and DXY data for correlation analysis - CRITICAL FOR GOLD"""

    @staticmethod
    async def get_spx_dxy_data() -> Dict[str, float]:
        """Get S&P 500 and Dollar Index data
        
        GOLD-SPECIFIC NOTE:
        - DXY (Dollar Index) has STRONG INVERSE correlation with Gold
        - When USD weakens, Gold typically rises
        - SPX correlation is moderate (risk-on/risk-off dynamics)
        """
        try:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()

            spx_data = await loop.run_in_executor(
                None, lambda: yf.download("^GSPC",
                                          period="5d",
                                          interval="1d",
                                          progress=False,
                                          auto_adjust=True,
                                          multi_level_index=False))
            dxy_data = await loop.run_in_executor(
                None, lambda: yf.download("DX-Y.NYB",
                                          period="5d",
                                          interval="1d",
                                          progress=False,
                                          auto_adjust=True,
                                          multi_level_index=False))

            spx_change = 0.0
            dxy_change = 0.0

            if not spx_data.empty and len(spx_data) >= 2:
                close_prices = spx_data['Close']
                change = (close_prices.iloc[-1] / close_prices.iloc[-2] -
                          1) * 100
                spx_change = change.item() if hasattr(
                    change, 'item') else float(change)

            if not dxy_data.empty and len(dxy_data) >= 2:
                close_prices = dxy_data['Close']
                change = (close_prices.iloc[-1] / close_prices.iloc[-2] -
                          1) * 100
                dxy_change = change.item() if hasattr(
                    change, 'item') else float(change)

            return {
                'spx_change': float(spx_change),
                'dxy_change': float(dxy_change)
            }
        except Exception as e:
            logger.error(f"Error fetching external markets: {e}")
            return {'spx_change': 0.0, 'dxy_change': 0.0}


# ==================== TRINITY EXECUTION ENGINES (GOLD-OPTIMIZED) ====================
class TrinityEngines:
    """Three specialized trading engines - GOLD OPTIMIZED"""

    @staticmethod
    def engine_sniper(df: pd.DataFrame) -> Dict:
        """Engine A: TTM Squeeze + RSI Divergence + MACD + ADX + Entropy"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            score = 0
            signals = []

            # TTM Squeeze
            if latest['squeeze_on'] and not prev['squeeze_on']:
                score += 20
                signals.append("TTM SQUEEZE FIRE 🎯")

            # MACD Crossover
            if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
                macd = latest['MACD_12_26_9']
                signal = latest['MACDs_12_26_9']
                prev_macd = prev['MACD_12_26_9']
                prev_signal = prev['MACDs_12_26_9']

                if macd > signal and prev_macd <= prev_signal:
                    score += 15
                    signals.append("MACD BULLISH CROSS")

            # ADX Trend Strength (GOLD-ADJUSTED: Lower threshold)
            if 'ADX_14' in df.columns:
                adx = latest['ADX_14']
                if adx > 20:  # Lower for Gold (was 25)
                    score += 15
                    signals.append(f"ADX STRONG ({adx:.1f})")

            # RSI
            if 'RSI_14' in df.columns:
                rsi = latest['RSI_14']
                if 30 < rsi < 70:
                    score += 10
                    signals.append(f"RSI NEUTRAL ({rsi:.1f})")

            # Entropy (Chaos measurement) - GOLD-ADJUSTED threshold
            returns = df['returns'].dropna().tail(50).values
            if len(returns) > 10:
                ent = entropy(np.abs(returns) + 1e-10)
                if ent < 0.25:  # Lower for Gold (was 0.3)
                    score += 20
                    signals.append(f"LOW ENTROPY ({ent:.2f}) 🥇")

            return {'score': score, 'signals': signals, 'active': score > 30}
        except Exception as e:
            logger.error(f"Engine Sniper error: {e}")
            return {'score': 0, 'signals': [], 'active': False}

    @staticmethod
    def engine_whale(df: pd.DataFrame, session: str,
                     external_data: Dict) -> Dict:
        """Engine B: VWAP + Session + SPX/DXY Correlation
        
        GOLD-SPECIFIC: DXY correlation is STRONGER and INVERSE for Gold
        """
        try:
            latest = df.iloc[-1]
            score = 0
            signals = []

            # VWAP Reclaim
            if 'vwap' in df.columns:
                price = latest['close']
                vwap = latest['vwap']
                prev_price = df.iloc[-2]['close']

                if price > vwap and prev_price <= vwap:
                    score += 25
                    signals.append("VWAP RECLAIM 🥇")

            # Trading Session (Gold-specific: London is king)
            if session == 'london':
                score += 20  # Higher bonus for London (Gold trades most during London)
                signals.append("LONDON SESSION 🇬🇧")
            elif session == 'newyork':
                score += 15
                signals.append("NEW YORK SESSION 🇺🇸")

            # SPX Correlation (Risk-On/Risk-Off)
            spx_change = external_data.get('spx_change', 0)
            # Gold can move both ways with SPX - it's a hedge
            if abs(spx_change) > 1.0:
                score += 10
                signals.append(f"SPX VOLATILE ({spx_change:.1f}%)")

            # DXY INVERSE Correlation - CRITICAL FOR GOLD
            # When DXY drops, Gold rises (and vice versa)
            dxy_change = external_data.get('dxy_change', 0)
            if dxy_change < -0.3:  # USD weakening = GOLD bullish
                score += 20  # Higher weight for Gold (was 10)
                signals.append(f"💪 DXY WEAK ({dxy_change:.1f}%) = GOLD BULLISH")
            elif dxy_change > 0.3:  # USD strengthening = caution for longs
                score -= 10  # Penalty for Gold longs when USD is strong
                signals.append(f"⚠️ DXY STRONG ({dxy_change:.1f}%) = CAUTION")

            return {'score': score, 'signals': signals, 'active': score > 30}
        except Exception as e:
            logger.error(f"Engine Whale error: {e}")
            return {'score': 0, 'signals': [], 'active': False}

    @staticmethod
    def engine_contrarian(funding_rate: float, obi: float) -> Dict:
        """Engine C: Order Book Imbalance (No funding rate for Forex)
        
        GOLD-SPECIFIC: Funding rate doesn't exist in Forex, so we rely on OBI only
        """
        score = 0
        signals = []

        # GOLD: No funding rate in Forex spot trading
        # Skip funding rate analysis

        # Order Book Imbalance (More bids than asks)
        if obi > 0.15:  # Slightly lower threshold for Forex
            score += 20  # Higher weight since no funding rate
            signals.append(f"BID PRESSURE ({obi:.2f}) 🥇")
        elif obi < -0.15:
            score -= 10
            signals.append(f"ASK PRESSURE ({obi:.2f})")

        return {'score': score, 'signals': signals, 'active': score > 10}


# ==================== RISK MANAGER ====================
class RiskManager:
    """Advanced risk management with Kelly Criterion"""

    def __init__(self, config: Config):
        self.config = config
        self.win_rate = 0.60  # Initial assumption
        self.avg_win = 0.03  # 3% for Gold (lower than BTC)
        self.avg_loss = 0.01  # 1% (3:1 RR)

    def calculate_position_size(self, balance: float, score: int,
                                stop_loss_pct: float) -> float:
        """Calculate position size using Kelly Criterion"""

        # Kelly Formula: f = (p*b - q) / b
        # where p = win rate, q = lose rate, b = win/loss ratio
        b = self.avg_win / self.avg_loss
        p = self.win_rate
        q = 1 - p

        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

        # Base risk
        base_risk_amount = balance * (self.config.ACCOUNT_RISK_PERCENT / 100)

        # Apply Kelly
        risk_amount = base_risk_amount * kelly_fraction

        # TERMINATOR MODE: 2.5x if score > 80
        if score >= self.config.TERMINATOR_SCORE:
            risk_amount *= self.config.TERMINATOR_MULTIPLIER
            logger.warning("🥇💀 TERMINATOR MODE ACTIVATED - GOLD ASSAULT 💀🥇")

        # Calculate position size based on stop loss
        position_size = risk_amount / stop_loss_pct

        # Cap at max position size
        max_position = balance * self.config.MAX_POSITION_SIZE
        position_size = min(position_size, max_position)

        return position_size

    def update_statistics(self, win: bool, profit_pct: float):
        """Update win rate and average returns"""
        # Exponential moving average
        alpha = 0.1

        self.win_rate = self.win_rate * (1 - alpha) + (1 if win else 0) * alpha

        if win:
            self.avg_win = self.avg_win * (1 - alpha) + abs(profit_pct) * alpha
        else:
            self.avg_loss = self.avg_loss * (1 -
                                             alpha) + abs(profit_pct) * alpha


# ==================== TELEGRAM NOTIFIER (GOLD EDITION) ====================
class TelegramNotifier:
    """Send formatted alerts to Telegram - GOLD BRANDING"""

    def __init__(self, config: Config):
        self.bot = Bot(token=config.TELEGRAM_TOKEN)
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.config = config

    async def send_signal(self, signal_data: Dict):
        """Send trade signal with risk percentage and amounts"""
        try:
            # --- ADATOK ELŐKÉSZÍTÉSE ---
            score = signal_data['score']
            direction = signal_data['direction']
            emoji = "🟢" if direction == "LONG" else "🔴"

            # 1. Kockázati % kiszámítása
            if score >= 80:
                risk_pct = self.config.ACCOUNT_RISK_PERCENT * self.config.TERMINATOR_MULTIPLIER
                risk_mode_text = f"⚠️ DUPLA KOCKÁZAT ({risk_pct}%)"
                prob_text = "🔥 NAGYON MAGAS (Terminator Mód)"
            elif score >= 70:
                risk_pct = self.config.ACCOUNT_RISK_PERCENT
                risk_mode_text = f"🛡️ NORMÁL MÉRET ({risk_pct}%)"
                prob_text = "💎 ERŐS (High Probability)"
            else:
                risk_pct = self.config.ACCOUNT_RISK_PERCENT
                risk_mode_text = f"🛡️ NORMÁL MÉRET ({risk_pct}%)"
                prob_text = "⚖️ KÖZEPES (Standard Trade)"

            # 2. Motorok állapotának kijelzése
            engines = signal_data.get('engines', {})
            sniper_active = engines.get('sniper', {}).get('active', False)
            whale_active = engines.get('whale', {}).get('active', False)
            contra_active = engines.get('contrarian', {}).get('active', False)

            e_sniper = "✅ AKTÍV" if sniper_active else "⚪ inaktív"
            e_whale = "✅ AKTÍV" if whale_active else "⚪ inaktív"
            e_contra = "✅ AKTÍV" if contra_active else "⚪ inaktív"

            # 3. Kockázat dollárban ($)
            pos_size = float(signal_data.get('position_size', 0))
            entry = float(signal_data['entry_price'])
            sl = float(signal_data['stop_loss'])

            # SL távolság és dollár kockázat számítása
            if entry > 0:
                sl_pct = abs(entry - sl) / entry
                risk_usd = pos_size * sl_pct
            else:
                risk_usd = 0.0

            # Get RR ratio and TP3 price
            rr_ratio = signal_data.get('rr_ratio', 8)
            rr_tier = signal_data.get('rr_tier', 'STANDARD')
            tp3_value = signal_data.get('tp3')
            
            # Handle TP3 display for different tiers
            if tp3_value is None:
                tp3_display = "N/A (SCALP)"
                tp3_rr_text = "N/A"
            else:
                tp3_display = f"${tp3_value:.2f}"
                if rr_tier == 'SWING':
                    tp3_rr_text = "1:10 RR"
                elif rr_tier == 'STANDARD':
                    tp3_rr_text = "1:8 RR"
                else:
                    tp3_rr_text = "1:5 RR"
            
            # Get trade type
            trade_type = signal_data.get('trade_type', 'TRADE')
            active_engines = signal_data.get('active_engines', 'N/A')
            
            # Get SMC data
            smc_bias = signal_data.get('smc_bias', {})
            smc_reason = smc_bias.get('reason', 'N/A')
            smc_confidence = smc_bias.get('confidence', 0)
            sl_distance = signal_data.get('sl_distance', abs(entry - sl))
            swing_high = signal_data.get('swing_high', 'N/A')
            swing_low = signal_data.get('swing_low', 'N/A')
            
            # Format swing levels
            swing_high_str = f"${swing_high:.2f}" if isinstance(swing_high, (int, float)) else "N/A"
            swing_low_str = f"${swing_low:.2f}" if isinstance(swing_low, (int, float)) else "N/A"
            
            # --- ÜZENET ÖSSZEÁLLÍTÁSA (GOLD BRANDING + SMC) ---
            message = f"""
🥇 **TERMINATOR GOLD SIGNAL** 🥇
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 **Symbol / Pár:** {signal_data['symbol']}
🎯 **Direction / Irány:** {emoji} **{direction}** {emoji}
🏷️ **Trade típus:** **{trade_type}**
📊 **RR Tier:** **{rr_tier}** (Max 1:{rr_ratio})
💪 **Strength / Erősség:** **{score}/100** {prob_text}
⚡ **Mode / Mód:** {risk_mode_text}
🤖 **Active Engines:** **{active_engines}/3**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 **SMC ANALYSIS / STRUKTÚRA:**
• 🎯 SMC Bias: **{smc_bias.get('bias', 'N/A')}** ({smc_confidence}%)
• 📍 Reason: *{smc_reason}*
• 📈 Swing High: {swing_high_str}
• 📉 Swing Low: {swing_low_str}

🤖 **STRATEGY ANALYSIS / STRATÉGIA:**
• 🎯 Sniper Engine: {e_sniper}
• 🥇 Whale Engine (DXY Focus): {e_whale}
• 📊 Contrarian Engine: {e_contra}

🧠 **AI METRICS / MÉRŐSZÁMOK:**
• Oracle (HMM) Bull: **{signal_data.get('hmm_bull_prob', 'N/A')}%**
• AI Confidence: **{signal_data.get('ai_confidence', 'N/A')}%**
• Market Chaos: **{signal_data.get('entropy', 'N/A')}**

💰 **TRADE LEVELS / SZINTEK:**
• Entry / Belépő: **${entry:.2f}**
• Stop Loss: **${sl:.2f}** (${sl_distance:.2f} SL)
• Target 1 (TP1): **${signal_data['tp1']:.2f}** *(1:3 RR)*
• Target 2 (TP2): **${signal_data['tp2']:.2f}** *(1:5 RR)*
• Target 3 (TP3): **{tp3_display}** *({tp3_rr_text})*

📈 **RISK MANAGEMENT / KOCKÁZAT:**
• Account Risk: **{risk_pct}%**
• Position Size: ${pos_size:.2f} USD
▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬
🥇 *GOLD Edition + SMC Strategy* 🥇
"""
            await self.bot.send_message(chat_id=self.chat_id,
                                        text=message,
                                        parse_mode='Markdown')
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")

    async def send_alert(self, message: str):
        """Send general alert"""
        try:
            await self.bot.send_message(chat_id=self.chat_id,
                                        text=f"🥇 {message}",
                                        parse_mode='Markdown')
        except TelegramError as e:
            logger.error(f"Telegram error: {e}")


# ==================== DATABASE MANAGER ====================
class DatabaseManager:
    """Store trade history and patterns"""

    def __init__(self, config: Config):
        self.db_path = config.DB_PATH

    async def init_db(self):
        """Initialize database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    score INTEGER,
                    result TEXT,
                    profit_pct REAL,
                    pattern_hash TEXT
                )
            """)
            await db.commit()

    async def save_trade(self, trade_data: Dict):
        """Save trade to database"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT INTO trades (timestamp, symbol, direction, entry_price, 
                                  stop_loss, take_profit, score, pattern_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (datetime.now().isoformat(), trade_data['symbol'],
                  trade_data['direction'], trade_data['entry_price'],
                  trade_data['stop_loss'], trade_data['take_profit'],
                  trade_data['score'], trade_data.get('pattern_hash', '')))
            await db.commit()


# ==================== MAIN TRADING ENGINE ====================
class TerminatorEngine:
    """The God-Mode Trading System - GOLD EDITION with SMC (Smart Money Concepts)"""

    def __init__(self, config: Config):
        self.config = config
        self.exchange = None
        self.hmm = HiddenMarkovModel()
        self.smc = SMCEngine(config)  # NEW: Smart Money Concepts Engine
        self.pattern_ai = PatternRecognition(config)
        self.risk_manager = RiskManager(config)
        self.telegram = TelegramNotifier(config)
        self.db = DatabaseManager(config)
        self.market_data = None
        self.running = False
        self.current_position = None
        self.paper_balance = 10000.0  # Simulated balance for paper trading
        self.paper_trades = []  # Track paper trades
        self.last_candle_timestamps = {}  # W4It: Track candle close times

    async def initialize(self):
        """Initialize exchange and database"""
        logger.info("🥇 INITIALIZING TERMINATOR GENESIS ENGINE - GOLD EDITION 🥇")

        # GOLD-SPECIFIC: Try multiple forex-compatible exchanges
        # Vantage is not in ccxt, so we use alternatives for data
        forex_exchanges = ['oanda', 'kraken', 'bitfinex']
        exchange_connected = False

        for exchange_name in forex_exchanges:
            try:
                logger.info(f"🔄 Trying {exchange_name} for Gold data...")
                exchange_class = getattr(ccxt, exchange_name)
                self.exchange = exchange_class({
                    'apiKey': self.config.API_KEY,
                    'secret': self.config.API_SECRET,
                    'enableRateLimit': True,
                })
                await self.exchange.load_markets()
                
                # Check if XAU/USD is available
                if self.config.SYMBOL in self.exchange.symbols:
                    self.market_data = MarketDataManager(self.exchange, self.config)
                    logger.info(f"✅ Connected to {exchange_name} for Gold data")
                    exchange_connected = True
                    break
                else:
                    logger.warning(f"❌ {exchange_name} doesn't have {self.config.SYMBOL}")
                    await self.exchange.close()
            except Exception as e:
                logger.warning(f"❌ {exchange_name} failed: {str(e)[:50]}")
                try:
                    await self.exchange.close()
                except:
                    pass
                continue

        if not exchange_connected:
            # Fallback: Use yfinance for Gold data
            logger.warning("⚠️ No exchange connected - using yfinance for Gold data")
            self.config.PAPER_TRADING = True
            self.exchange = None
            logger.info(f"📝 Paper Trading | Simulated Balance: ${self.paper_balance:.2f} USD")

        # Initialize database
        await self.db.init_db()

        try:
            await self.telegram.send_alert("🥇💀 TERMINATOR GENESIS GOLD EDITION ONLINE 💀🥇")
        except Exception as e:
            logger.warning(f"Telegram notification failed (optional): {e}")

    def get_trading_session(self) -> str:
        """Determine current trading session - GOLD OPTIMIZED"""
        from datetime import timezone
        hour = datetime.now(timezone.utc).hour
        # Gold sessions (Forex hours)
        if 8 <= hour < 16:
            return "london"  # Most active for Gold
        elif 13 <= hour < 21:
            return "newyork"
        elif 0 <= hour < 8:
            return "asian"
        return "other"

    def determine_rr_tier(self, score: int, entropy: float, adx: float, session: str) -> str:
        """Determine optimal RR tier based on market conditions.
        
        Returns: 'SCALP', 'STANDARD', or 'SWING'
        
        Factors considered:
        - Score: Higher score = more aggressive targets
        - Entropy: Lower entropy (less chaos) = better for swing trades
        - ADX: Higher ADX = stronger trend = extend targets
        - Session: London/NY sessions favor swing trades (Gold)
        """
        # Base tier from score (SCALP: 55-59, STANDARD: 60-79, SWING: 80+)
        if score >= 80:
            base_tier = "SWING"
        elif score >= 60:
            base_tier = "STANDARD"
        else:
            base_tier = "SCALP"
        
        # Upgrade/downgrade based on conditions
        upgrade_points = 0
        
        # Low entropy is good for extending targets
        if entropy < 0.15:
            upgrade_points += 2
        elif entropy < 0.25:
            upgrade_points += 1
        elif entropy > 0.35:
            upgrade_points -= 1
        
        # Strong trend (high ADX) favors higher RR
        if adx > 30:
            upgrade_points += 2
        elif adx > 25:
            upgrade_points += 1
        elif adx < 18:
            upgrade_points -= 1
        
        # London session is best for Gold swings
        if session == "london":
            upgrade_points += 1
        elif session == "newyork":
            upgrade_points += 0.5
        elif session == "asian":
            upgrade_points -= 1
        
        # Determine final tier
        tier_levels = ["SCALP", "STANDARD", "SWING"]
        current_idx = tier_levels.index(base_tier)
        
        # Upgrade/downgrade based on points (need 2+ points to change tier)
        if upgrade_points >= 2 and current_idx < 2:
            final_tier = tier_levels[current_idx + 1]
        elif upgrade_points <= -2 and current_idx > 0:
            final_tier = tier_levels[current_idx - 1]
        else:
            final_tier = base_tier
        
        logger.info(f"📊 RR Tier: {final_tier} (base: {base_tier}, points: {upgrade_points:.1f})")
        return final_tier

    async def fetch_spot_gold_price(self) -> Optional[float]:
        """Fetch REAL spot XAU/USD price from multiple sources.
        
        Priority:
        1. goldprice.org - Free, no API key, most accurate real-time spot prices
        2. gold-api.com - Free, no API key, backup (slightly higher prices)  
        3. Swissquote - FALLBACK ONLY (caches data during market closures!)
        """
        import aiohttp
        
        # SOURCE 1: goldprice.org (most accurate, always fresh)
        logger.info("Trying goldprice.org API...")
        try:
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                url = "https://data-asg.goldprice.org/dbXRates/USD"
                headers = {"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10), headers=headers) as response:
                    logger.info(f"goldprice.org response status: {response.status}")
                    if response.status == 200:
                        text = await response.text()
                        import json
                        data = json.loads(text)
                        items = data.get('items', [])
                        if items and len(items) > 0:
                            price = items[0].get('xauPrice')
                            logger.info(f"goldprice.org parsed price: {price}")
                            if price and price > 1000:  # Sanity check
                                logger.info(f"SPOT XAU/USD from goldprice.org: ${price:.2f}")
                                return float(price)
                        else:
                            logger.info(f"goldprice.org no items found in response")
                    else:
                        logger.info(f"goldprice.org returned status {response.status}")
        except Exception as e:
            logger.info(f"goldprice.org EXCEPTION: {type(e).__name__}: {e}")
        
        # SOURCE 2: gold-api.com (backup, slightly higher prices)
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.gold-api.com/price/XAU"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data.get('price')
                        if price and price > 1000:  # Sanity check
                            logger.info(f"✅ SPOT XAU/USD from gold-api.com: ${price:.2f}")
                            return float(price)
        except Exception as e:
            logger.warning(f"gold-api.com failed: {e}")
        
        # SOURCE 3: Swissquote (FALLBACK - may cache stale data during market closures!)
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://forex-data-feed.swissquote.com/public-quotes/bboquotes/instrument/XAU/USD"
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and len(data) > 0:
                            spreads = data[0].get('spreadProfilePrices', [])
                            for spread in spreads:
                                if spread.get('spreadProfile') == 'elite':
                                    bid = spread.get('bid')
                                    ask = spread.get('ask')
                                    if bid and ask:
                                        mid_price = (bid + ask) / 2
                                        logger.warning(f"⚠️ SPOT XAU/USD from Swissquote (fallback): ${mid_price:.2f}")
                                        return mid_price
                            
                            if spreads:
                                bid = spreads[0].get('bid')
                                if bid:
                                    logger.warning(f"⚠️ SPOT XAU/USD from Swissquote (fallback): ${bid:.2f}")
                                    return float(bid)
                                    
        except Exception as e:
            logger.warning(f"Swissquote API failed: {e}")
        
        # LAST RESORT FALLBACK: Futures with 0.5% discount
        try:
            loop = asyncio.get_event_loop()
            gc = await loop.run_in_executor(
                None, lambda: yf.download("GC=F", period="1d", progress=False, auto_adjust=True))
            
            if not gc.empty:
                futures_price = float(gc['Close'].iloc[-1])
                if hasattr(futures_price, 'item'):
                    futures_price = futures_price.item()
                
                # Small discount (futures typically slightly higher)
                spot_estimate = futures_price * 0.995
                logger.warning(f"⚠️ Fallback: futures ${futures_price:.2f} × 0.995 = ${spot_estimate:.2f}")
                return spot_estimate
                
        except Exception as e:
            logger.error(f"All price sources failed: {e}")
        
        return None

    async def fetch_gold_data_yfinance(self, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch Gold OHLCV data with CORRECTED spot prices.
        
        Strategy:
        1. Fetch futures data (GC=F) for OHLCV history (more reliable data)
        2. Get real spot price from GLD ETF
        3. Apply adjustment ratio to convert all prices to spot equivalent
        """
        try:
            loop = asyncio.get_event_loop()
            
            tf_map = {
                "15m": ("5d", "15m"),
                "1h": ("30d", "1h"),
                "4h": ("60d", "1h"),
            }
            
            period, interval = tf_map.get(timeframe, ("30d", "1h"))
            
            # Fetch GC=F (futures) for OHLCV data (most reliable source for historical data)
            data = await loop.run_in_executor(
                None, lambda: yf.download("GC=F",
                                          period=period,
                                          interval=interval,
                                          progress=False,
                                          auto_adjust=True,
                                          multi_level_index=False))
            
            if data.empty:
                logger.error("No gold data available from yfinance")
                return pd.DataFrame()
            
            df = data.reset_index()
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Get real SPOT price and calculate adjustment ratio
            real_spot = await self.fetch_spot_gold_price()
            if real_spot and len(df) > 0:
                futures_current = float(df['close'].iloc[-1])
                
                # Calculate adjustment ratio (spot is typically lower than futures)
                adjustment_ratio = real_spot / futures_current
                logger.info(f"📊 Adjusting: futures=${futures_current:.2f} → spot=${real_spot:.2f} (ratio={adjustment_ratio:.4f})")
                
                # Apply adjustment to all OHLCV prices
                for col in ['open', 'high', 'low', 'close']:
                    df[col] = df[col] * adjustment_ratio
            
            df = df.tail(limit)
            final_price = float(df['close'].iloc[-1])
            logger.info(f"✅ Gold data ready - SPOT price: ${final_price:.2f}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching Gold data: {e}")
            return pd.DataFrame()

    async def analyze_market(self) -> Optional[Dict]:
        """
        Complete market analysis with SMC (Smart Money Concepts)
        
        Key improvements:
        - Multi-timeframe SMC structure analysis (15m, 30m, 1h, 4h, Daily)
        - CHoCH/BOS detection for direction filtering
        - W4It: Only act on CLOSED candles
        - Precise SL based on swing structure
        - SUPPORTS BOTH LONG AND SHORT!
        """
        try:
            # Fetch multi-timeframe data
            dfs = {}
            for name, tf in self.config.TIMEFRAMES.items():
                if self.exchange and self.market_data:
                    df = await self.market_data.fetch_ohlcv(tf, limit=500)
                else:
                    # Fallback to yfinance
                    df = await self.fetch_gold_data_yfinance(tf, limit=500)
                
                if df.empty:
                    logger.warning(f"No data for {tf}")
                    return None
                df = IndicatorEngine.calculate_all(df)
                dfs[name] = df
            
            # ==================== W4It: CANDLE CLOSE CONFIRMATION ====================
            # Only trade on CONFIRMED (closed) candles - not on live candle movements!
            scalp_df = dfs.get('scalp', dfs.get('medium'))
            if 'timestamp' in scalp_df.columns:
                last_candle_time = scalp_df['timestamp'].iloc[-1]
                last_seen = self.last_candle_timestamps.get('scalp')
                
                if last_seen is not None and last_candle_time == last_seen:
                    # Same candle as before - wait for it to close!
                    return None
                
                self.last_candle_timestamps['scalp'] = last_candle_time
                logger.info(f"✅ W4It: New candle confirmed at {last_candle_time}")

            # Get current price from scalp timeframe
            current_price = float(dfs['scalp']['close'].iloc[-1])

            # ==================== SMC MULTI-TIMEFRAME ANALYSIS ====================
            # Analyze market structure on each timeframe
            multi_tf_structures = {}
            for tf_name in ['daily', 'swing', 'medium', 'intraday', 'scalp']:
                if tf_name in dfs:
                    structure = self.smc.detect_market_structure(dfs[tf_name])
                    multi_tf_structures[tf_name] = structure
                    
                    # Log important structure events
                    if structure['choch']['detected']:
                        logger.warning(f"📊 {tf_name.upper()} CHoCH {structure['choch']['type']} detected!")
                    if structure['bos']['detected']:
                        logger.info(f"📈 {tf_name.upper()} BOS {structure['bos']['type']} - trend continues")
            
            # Get SMC bias (direction filter)
            smc_bias = self.smc.get_smc_bias(multi_tf_structures)
            logger.info(f"🎯 SMC Bias: {smc_bias['bias']} | Confidence: {smc_bias['confidence']}% | {smc_bias['reason']}")
            
            # Find FVG zones on scalp timeframe for entry
            fvg_zones = self.smc.find_fvg_zones(dfs['scalp'])
            
            # Flash crash detection
            if self.market_data and self.market_data.detect_flash_crash(current_price):
                logger.critical("🚨 GOLD FLASH CRASH DETECTED - EMERGENCY PROTOCOLS 🚨")
                await self.telegram.send_alert("GOLD FLASH CRASH DETECTED - ALL POSITIONS CLOSED")
                if self.current_position:
                    await self.close_position("FLASH_CRASH")
                return None

            # Get external market data (CRITICAL FOR GOLD)
            external_data = await ExternalMarkets.get_spx_dxy_data()

            # Get order book imbalance
            funding_rate = 0.0  # No funding in forex
            obi = 0.0
            if self.market_data:
                obi = await self.market_data.get_order_book_imbalance()

            # ==================== HMM ORACLE ANALYSIS ====================
            price_changes = dfs['medium']['returns'].dropna().tail(50).tolist()
            hmm_probs = self.hmm.calculate_state_probabilities(price_changes)

            # HMM direction filter
            if hmm_probs['BEAR'] > self.config.HMM_BEAR_THRESHOLD and smc_bias['bias'] == 'LONG':
                logger.warning(f"⚠️ ORACLE WARNING: {hmm_probs['BEAR']*100:.1f}% BEAR - Blocking LONG")
                return None
            if hmm_probs['BULL'] > self.config.HMM_BULL_THRESHOLD and smc_bias['bias'] == 'SHORT':
                logger.warning(f"⚠️ ORACLE WARNING: {hmm_probs['BULL']*100:.1f}% BULL - Blocking SHORT")
                return None

            # ==================== TRINITY ENGINES ANALYSIS ====================
            session = self.get_trading_session()

            engine_a = TrinityEngines.engine_sniper(dfs['scalp'])
            engine_b = TrinityEngines.engine_whale(dfs['medium'], session, external_data)
            engine_c = TrinityEngines.engine_contrarian(funding_rate, obi)

            # Calculate total score
            total_score = engine_a['score'] + engine_b['score'] + engine_c['score']
            
            # Add SMC confidence bonus
            if smc_bias['confidence'] >= 80:
                total_score += 15
                logger.info(f"💎 SMC CONFIDENCE BONUS +15 (total: {total_score})")
            elif smc_bias['confidence'] >= 70:
                total_score += 10

            # Check minimum active engines
            active_engines = sum([
                1 if engine_a['active'] else 0,
                1 if engine_b['active'] else 0,
                1 if engine_c['active'] else 0
            ])
            
            if active_engines < self.config.MIN_ACTIVE_ENGINES:
                logger.info(f"⚠️ Only {active_engines} engines active (need {self.config.MIN_ACTIVE_ENGINES}) - skipping")
                return None

            # Pattern Recognition AI
            recent_prices = dfs['medium']['close'].tail(self.config.PATTERN_HISTORY_CANDLES).values
            ai_confidence, pattern_matches = self.pattern_ai.find_similar_patterns(
                recent_prices, dfs['swing'])

            # Add AI bonus
            if ai_confidence > 70:
                total_score += 15

            # Calculate entropy
            returns = dfs['scalp']['returns'].dropna().tail(50).values
            market_entropy = entropy(np.abs(returns) + 1e-10) if len(returns) > 10 else 1.0

            # Check minimum score
            if total_score < self.config.MIN_SCORE:
                logger.info(f"📉 Score {total_score} below minimum {self.config.MIN_SCORE} - skipping")
                return None

            # ==================== DETERMINE DIRECTION FROM SMC ====================
            # Direction is now determined by SMC bias, not hardcoded LONG!
            if smc_bias['bias'] == 'NEUTRAL':
                logger.info("⚖️ SMC NEUTRAL - No clear direction, skipping")
                return None
            
            direction = smc_bias['bias']  # 'LONG' or 'SHORT'
            
            # Double-check direction is allowed
            if direction == 'LONG' and not smc_bias['long_allowed']:
                logger.warning(f"🚫 LONG FORBIDDEN by SMC: {smc_bias['reason']}")
                return None
            if direction == 'SHORT' and not smc_bias['short_allowed']:
                logger.warning(f"🚫 SHORT FORBIDDEN by SMC: {smc_bias['reason']}")
                return None

            logger.info(f"🎯 DIRECTION: {direction} | Score: {total_score}")

            # ==================== CALCULATE PRECISE SL/TP ====================
            atr = float(dfs['scalp']['ATRr_14'].iloc[-1])
            entry_price = current_price
            adx_value = float(dfs['scalp']['ADX_14'].iloc[-1]) if 'ADX_14' in dfs['scalp'].columns else 20
            
            # Determine RR tier
            rr_tier = self.determine_rr_tier(total_score, market_entropy, adx_value, session)
            tier_config = self.config.RR_TIERS[rr_tier]
            
            # Get structure for precise SL placement
            scalp_structure = multi_tf_structures.get('scalp', {})
            
            # Calculate PRECISE SL based on swing structure (not arbitrary ATR!)
            stop_loss = self.smc.calculate_structure_sl(
                direction, 
                scalp_structure, 
                current_price, 
                rr_tier
            )
            
            # Calculate SL distance in dollars
            sl_distance = abs(entry_price - stop_loss)
            max_sl_allowed = self.config.SL_LIMITS.get(rr_tier, 10)
            
            # Validate SL is within limits
            if sl_distance > max_sl_allowed:
                logger.warning(f"⚠️ SL ${sl_distance:.2f} exceeds max ${max_sl_allowed} for {rr_tier} - capping")
                if direction == 'LONG':
                    stop_loss = entry_price - max_sl_allowed
                else:
                    stop_loss = entry_price + max_sl_allowed
                sl_distance = max_sl_allowed
            
            logger.info(f"📍 PRECISE SL: ${stop_loss:.2f} (distance: ${sl_distance:.2f}, max: ${max_sl_allowed})")
            
            # Calculate TP levels based on SL distance and RR ratios
            if direction == 'LONG':
                tp1 = entry_price + (sl_distance * 3)   # 1:3 RR
                tp2 = entry_price + (sl_distance * 5)   # 1:5 RR
                tp3 = entry_price + (sl_distance * tier_config['max_rr']) if tier_config['tp3_mult'] else None
            else:  # SHORT
                tp1 = entry_price - (sl_distance * 3)   # 1:3 RR
                tp2 = entry_price - (sl_distance * 5)   # 1:5 RR
                tp3 = entry_price - (sl_distance * tier_config['max_rr']) if tier_config['tp3_mult'] else None
            
            max_rr = tier_config['max_rr']
            
            # Trade type display
            direction_emoji = "🟢" if direction == "LONG" else "🔴"
            if rr_tier == "SWING":
                trade_type = f"{direction_emoji} SWING {direction} (1:10 RR)"
            elif rr_tier == "STANDARD":
                trade_type = f"{direction_emoji} STANDARD {direction} (1:8 RR)"
            else:
                trade_type = f"{direction_emoji} SCALP {direction} (1:5 RR)"

            # Compile signal with SMC data
            signal = {
                'timestamp': datetime.now(),
                'symbol': self.config.SYMBOL,
                'direction': direction,
                'score': total_score,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'sl_distance': sl_distance,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'atr': atr,
                'rr_ratio': max_rr,
                'rr_tier': rr_tier,
                'trade_type': trade_type,
                'entropy': market_entropy,
                'adx': adx_value,
                'hmm_bull_prob': hmm_probs['BULL'] * 100,
                'hmm_bear_prob': hmm_probs['BEAR'] * 100,
                'ai_confidence': ai_confidence,
                'pattern_matches': pattern_matches,
                'engines': {
                    'sniper': engine_a,
                    'whale': engine_b,
                    'contrarian': engine_c
                },
                'active_engines': active_engines,
                'external_data': external_data,
                'funding_rate': funding_rate,
                'obi': obi,
                # SMC DATA
                'smc_bias': smc_bias,
                'smc_structures': multi_tf_structures,
                'fvg_zones': fvg_zones,
                'swing_high': scalp_structure.get('last_swing_high'),
                'swing_low': scalp_structure.get('last_swing_low')
            }

            logger.info(f"✅ SIGNAL READY: {direction} @ ${entry_price:.2f} | SL ${stop_loss:.2f} | TP1 ${tp1:.2f}")
            return signal

        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def execute_trade(self, signal: Dict):
        """Execute trade with proper risk management"""
        try:
            # Paper trading mode - simulate trades without exchange connection
            if self.config.PAPER_TRADING:
                available_balance = self.paper_balance

                # Calculate position size
                stop_loss_pct = abs(signal['entry_price'] - signal['stop_loss']
                                    ) / signal['entry_price']
                position_size_usdt = self.risk_manager.calculate_position_size(
                    available_balance, signal['score'], stop_loss_pct)

                # Convert to ounces (Gold is traded in oz)
                amount = position_size_usdt / signal['entry_price']

                logger.info(
                    f"📝 PAPER TRADE: {signal['direction']} GOLD | Size: {position_size_usdt:.2f} USD")

                # Store paper position
                self.current_position = {
                    'side': signal['direction'].lower(),
                    'entry_price': signal['entry_price'],
                    'amount': amount,
                    'stop_loss': signal['stop_loss'],
                    'tp1': signal['tp1'],
                    'tp2': signal['tp2'],
                    'tp3': signal['tp3'],
                    'score': signal['score'],
                    'rr_tier': signal.get('rr_tier', 'STANDARD'),  # Track RR tier for upgrade logic
                    'trade_type': signal.get('trade_type', 'TRADE'),
                    'entry_time': datetime.now(),
                    'breakeven_sl': signal['stop_loss'],
                    'sl_at_breakeven': False,
                    'orders': {
                        'entry': 'paper',
                        'sl': 'paper',
                        'tp1': 'paper'
                    }
                }

                self.paper_trades.append({
                    'timestamp': datetime.now(),
                    'direction': signal['direction'],
                    'entry_price': signal['entry_price'],
                    'size_usdt': position_size_usdt
                })

                logger.info("📝 Paper trade recorded successfully")
                
                # Send Telegram signal
                await self.telegram.send_signal({
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'score': signal['score'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'tp1': signal['tp1'],
                    'tp2': signal['tp2'],
                    'tp3': signal['tp3'],  # Show actual TP3 price
                    'rr_ratio': signal.get('rr_ratio', 8),  # RR ratio
                    'trade_type': signal.get('trade_type', 'TRADE'),  # SCALP or SWING
                    'entropy': f"{signal['entropy']:.2f}",
                    'hmm_bull_prob': f"{signal['hmm_bull_prob']:.1f}",
                    'ai_confidence': f"{signal['ai_confidence']:.0f}",
                    'engines': signal['engines'],
                    'position_size': f"{position_size_usdt:.2f}",
                    'latency': 'Paper-Mode'
                })
                
                # Save to database
                await self.db.save_trade({
                    'symbol': signal['symbol'],
                    'direction': signal['direction'],
                    'entry_price': signal['entry_price'],
                    'stop_loss': signal['stop_loss'],
                    'take_profit': signal['tp1'],
                    'score': signal['score'],
                    'pattern_hash': 'PAPER_TRADE'
                })
                logger.info("💾 Paper trade saved to DB")
                return

            # Live trading mode (for future Vantage integration)
            logger.warning("Live trading not implemented for Forex yet")

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            await self.telegram.send_alert(f"❌ Trade Execution Failed: {e}")

    async def close_position(self, reason: str):
        """Close current position"""
        if not self.current_position:
            return

        try:
            logger.info(f"🔴 CLOSING GOLD POSITION | Reason: {reason}")

            # Paper trading mode - just log and clear
            if self.config.PAPER_TRADING:
                logger.info(f"📝 Paper position closed: {reason}")
                self.current_position = None
                return

        except Exception as e:
            logger.error(f"Error closing position: {e}")

    async def monitor_position(self):
        """Monitor and manage active position"""
        if not self.current_position:
            return

        try:
            # Get current price
            if self.exchange:
                ticker = await self.exchange.fetch_ticker(self.config.SYMBOL)
                current_price = ticker['last']
            else:
                # Fallback: Get price from yfinance
                df = await self.fetch_gold_data_yfinance("15m", limit=1)
                if df.empty:
                    return
                current_price = df['close'].iloc[-1]

            # Flash crash check
            if self.market_data and self.market_data.detect_flash_crash(current_price):
                logger.critical("🚨 FLASH CRASH DETECTED!")
                await self.telegram.send_alert(
                    f"🚨 **EMERGENCY BRAKE ACTIVATED / VÉSZFÉK AKTIVÁLVA** 🚨\n"
                    f"📉 Sudden market crash (Flash Crash) detected.\n"
                    f"📉 Hirtelen piaci összeomlást észleltem.\n"
                    f"🛡️ All positions CLOSED immediately to protect capital.\n"
                    f"🛡️ A tőke védelme érdekében AZONNAL ZÁRTAM mindent.")
                await self.close_position("FLASH_CRASH_EMERGENCY")
                return

            entry = self.current_position['entry_price']
            is_long = self.current_position['side'] == 'long'

            # Initialize hit levels tracker
            if 'hit_levels' not in self.current_position:
                self.current_position['hit_levels'] = []

            # --- STOP LOSS HIT DETECTION ---
            sl_hit = (is_long and current_price <= self.current_position['stop_loss']) or \
                     (not is_long and current_price >= self.current_position['stop_loss'])
            
            if sl_hit:
                if 'sl_hit' not in self.current_position['hit_levels']:
                    logger.warning("⛔ STOP LOSS HIT!")
                    
                    is_breakeven = self.current_position.get('sl_at_breakeven', False)
                    direction_emoji = "📈" if is_long else "📉"
                    
                    if is_breakeven:
                        await self.telegram.send_alert(
                            f"⛔ **STOP LOSS HIT** (${current_price:.2f})\n"
                            f"🛡️ **At Breakeven level** (after TP1)\n"
                            f"🛡️ **Breakeven szinten** (TP1 után)\n"
                            f"✅ No-loss exit - capital saved! / Nyereség nélküli kijárulás - tőke megmentve!\n"
                            f"{direction_emoji} Direction / Irány: {self.current_position['side'].upper()}"
                        )
                    else:
                        loss_pct = abs(current_price - entry) / entry * 100
                        await self.telegram.send_alert(
                            f"⛔ **STOP LOSS HIT** (${current_price:.2f})\n"
                            f"📉 Loss / Veszteség: **{loss_pct:.2f}%**\n"
                            f"🛡️ Position closed due to risk limits.\n"
                            f"🛡️ Pozíció lezárva a kockázat korlátok miatt.\n"
                            f"{direction_emoji} Direction / Irány: {self.current_position['side'].upper()}"
                        )
                    
                    self.current_position['hit_levels'].append('sl_hit')
                    await self.close_position("STOP_LOSS_HIT")
                    return

            # --- TP1 HIT ---
            tp1_hit = (is_long and current_price >= self.current_position['tp1']) or \
                      (not is_long and current_price <= self.current_position['tp1'])

            if tp1_hit:
                if 'tp1' not in self.current_position['hit_levels']:
                    logger.info("🎯 TP1 HIT - Moving SL to Breakeven")
                    
                    self.current_position['stop_loss'] = self.current_position['entry_price']
                    self.current_position['sl_at_breakeven'] = True
                    profit_pct = abs(current_price - entry) / entry * 100
                    trade_type = self.current_position.get('trade_type', 'TRADE')

                    direction_emoji = "📈" if is_long else "📉"
                    await self.telegram.send_alert(
                        f"💰 **TP1 REACHED / ELÉRVE** (${current_price:.2f})\n"
                        f"🏷️ {trade_type}\n"
                        f"📊 Profit / Nyereség: **+{profit_pct:.2f}%** | **1:3 RR locked / biztosítva!**\n"
                        f"🛡️ SL moved to Breakeven (${self.current_position['entry_price']:.2f})\n"
                        f"🛡️ SL húzása Breakeven szintre\n"
                        f"✅ First target achieved! / Az első célár teljesült!\n"
                        f"{direction_emoji} Direction / Irány: {self.current_position['side'].upper()}"
                    )
                    self.current_position['hit_levels'].append('tp1')
                    
                    # --- DYNAMIC TP UPGRADE LOGIC ---
                    # ONLY upgrade if current trade is NOT already SWING tier AND new conditions warrant upgrade
                    current_rr_tier = self.current_position.get('rr_tier', 'SCALP')
                    
                    # Only try to upgrade if we're not already at max tier
                    if current_rr_tier != 'SWING':
                        try:
                            upgrade_signal = await self.analyze_market()
                            if upgrade_signal and upgrade_signal.get('rr_tier') == 'SWING':
                                # Market conditions upgraded to SWING tier - this is a real improvement!
                                new_atr = upgrade_signal['atr']
                                old_tp3 = self.current_position.get('tp3')
                                new_tp3 = current_price + (new_atr * 10)  # Extend to 1:10 from current
                                
                                if old_tp3 is None or (new_tp3 and new_tp3 > old_tp3):
                                    self.current_position['tp3'] = new_tp3
                                    self.current_position['rr_tier'] = 'SWING'
                                    
                                    old_display = f"${old_tp3:.2f}" if old_tp3 else "N/A (was SCALP)"
                                    await self.telegram.send_alert(
                                        f"📈 **TP3 UPGRADED** 🚀\n"
                                        f"🔥 Market improved from {current_rr_tier} → SWING!\n"
                                        f"📊 Old TP3: {old_display}\n"
                                        f"🎯 New TP3: **${new_tp3:.2f}** (1:10 RR)\n"
                                        f"💎 Letting profits run!"
                                    )
                                    logger.info(f"📈 TP3 upgraded: {current_rr_tier} -> SWING, TP3 = ${new_tp3:.2f}")
                        except Exception as e:
                            logger.warning(f"TP upgrade check failed: {e}")

            # --- TP2 HIT ---
            tp2_hit = (is_long and current_price >= self.current_position['tp2']) or \
                      (not is_long and current_price <= self.current_position['tp2'])

            if tp2_hit:
                if 'tp2' not in self.current_position['hit_levels']:
                    logger.info("🚀 TP2 HIT")
                    profit_pct = abs(current_price - entry) / entry * 100
                    trade_type = self.current_position.get('trade_type', 'TRADE')
                    direction_emoji = "📈" if is_long else "📉"
                    
                    await self.telegram.send_alert(
                        f"🚀 **TP2 REACHED / ELÉRVE** (${current_price:.2f})\n"
                        f"🏷️ {trade_type}\n"
                        f"💰 Profit / Nyereség: **+{profit_pct:.2f}%** | **1:5 RR achieved / elérve!**\n"
                        f"📊 Excellent performance! Market moving in your favor.\n"
                        f"📊 Remek teljesítmény! A piac az irányodba szakad.\n"
                        f"{direction_emoji} Direction / Irány: {self.current_position['side'].upper()}"
                    )
                    self.current_position['hit_levels'].append('tp2')

            # --- TP3 HIT --- (only check if TP3 exists and is not None)
            tp3_value = self.current_position.get('tp3')
            if tp3_value is not None:
                tp3_hit = (is_long and current_price >= tp3_value) or \
                          (not is_long and current_price <= tp3_value)

                if tp3_hit:
                    if 'tp3' not in self.current_position['hit_levels']:
                        logger.info("🌌 TP3 HIT - GOLD RUSH!")
                        profit_pct = abs(current_price - entry) / entry * 100
                        trade_type = self.current_position.get('trade_type', 'TRADE')
                        direction_emoji = "📈" if is_long else "📉"
                        
                        await self.telegram.send_alert(
                            f"🌌 **TP3 REACHED - MAXIMUM PROFIT / ELÉRVE - MAX NYERESÉG** (${current_price:.2f})\n"
                            f"🏷️ {trade_type}\n"
                            f"💎 Super profit / Szupernyereség: **+{profit_pct:.2f}%** | **1:8 RR MAXIMUM!**\n"
                            f"🥇 GOLD RUSH! Final target achieved! / Végső cél elérve!\n"
                            f"🥂 Only elite traders reach this level!\n"
                            f"🥂 Ezt az eredményt csak a legmagasabb szintű kereskedők érik el!\n"
                            f"{direction_emoji} Direction / Irány: {self.current_position['side'].upper()}\n"
                            f"✅ Position CLOSED - Final profit target reached!\n"
                            f"✅ Pozíció LEZÁRVA - Végső profit target elérve!"
                        )
                        self.current_position['hit_levels'].append('tp3')
                        
                        # TP3 is the final target - CLOSE THE POSITION!
                        await self.close_position("TP3_MAX_PROFIT_REACHED")
                        return

            # Stalling detection - SMART VERSION
            time_in_trade = (datetime.now() -
                             self.current_position['entry_time']).seconds / 60
            
            # Calculate current profit
            if is_long:
                current_pnl_pct = (current_price - entry) / entry
            else:
                current_pnl_pct = (entry - current_price) / entry
            
            # Check if we hit any TP levels (profitable trade)
            hit_tp1 = 'tp1' in self.current_position.get('hit_levels', [])
            hit_tp2 = 'tp2' in self.current_position.get('hit_levels', [])
            
            # SMART STALLING: Don't close profitable high-RR trades!
            # If TP1 hit (3:1 RR) - extend to 8 hours
            # If TP2 hit (5:1 RR) - extend to 12 hours, let it run for TP3
            if hit_tp2:
                stall_timeout = 720  # 12 hours for TP2+ trades
            elif hit_tp1:
                stall_timeout = 480  # 8 hours for TP1+ trades  
            else:
                stall_timeout = 240  # 4 hours for trades that haven't hit any TP
            
            if time_in_trade > stall_timeout:
                signal = await self.analyze_market()
                # Only close if market turned against us AND we're not in profit
                if (signal is None or signal['score'] < 40) and current_pnl_pct <= 0:
                    logger.info("⏳ STALLING - Market not moving, closing position.")
                    await self.telegram.send_alert(
                        f"⏳ **TIMEOUT / IDŐTÚLLÉPÉS (STALLING)**\n"
                        f"😴 Market hasn't moved in expected direction for {stall_timeout//60} hours.\n"
                        f"😴 A piac {stall_timeout//60} órája nem indult el a várt irányba.\n"
                        f"📉 Closing position to free up capital.\n"
                        f"📉 A tőke felszabadítása érdekében zárom a pozíciót.\n"
                        f"No risk on sideways market. / Nem kockáztatunk oldalazó piacon.")
                    await self.close_position("STALLING_DETECTED")
                elif current_pnl_pct > 0:
                    logger.info(f"⏳ Timeout reached but trade is profitable (+{current_pnl_pct*100:.2f}%), letting it run...")

        except Exception as e:
            logger.error(f"Position monitoring error: {e}")

    async def run(self):
        """Main trading loop"""
        self.running = True
        logger.info("🥇💀 TERMINATOR GOLD ENGINE ACTIVATED 💀🥇")

        await self.initialize()

        while self.running:
            try:
                # Monitor existing position
                await self.monitor_position()

                # Only analyze for new trades if no position
                if not self.current_position:
                    signal = await self.analyze_market()

                    if signal and signal['score'] >= self.config.MIN_SCORE:
                        logger.info(
                            f"🎯 GOLD SIGNAL DETECTED | Score: {signal['score']}")
                        await self.execute_trade(signal)

                # Wait before next iteration
                await asyncio.sleep(15)  # Check every 15 seconds (Gold moves slower)

            except KeyboardInterrupt:
                logger.info("⚠️ Shutdown requested")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(30)

        # Cleanup
        if self.current_position:
            await self.close_position("SHUTDOWN")

        if self.exchange:
            try:
                await self.exchange.close()
            except Exception as e:
                logger.warning(f"Error closing exchange connection: {e}")
        logger.info("🥇💀 TERMINATOR GOLD ENGINE OFFLINE 💀🥇")


# ==================== MAIN ENTRY POINT ====================
async def main():
    """Launch the Terminator Genesis Engine - GOLD EDITION"""

    # Start keep-alive server for Replit
    keep_alive()

    # Initialize config
    config = Config()

    # Create and run engine
    engine = TerminatorEngine(config)

    try:
        await engine.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Terminator Genesis Gold shutting down...")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🥇💀 TERMINATOR GENESIS TRADING ENGINE 💀🥇          ║
    ║              ** GOLD EDITION **                           ║
    ║                                                           ║
    ║          Advanced HFT Algorithm v1.0                      ║
    ║          Optimized for XAU/USD (Gold)                     ║
    ║          Python 3.11 | Fully Autonomous                   ║
    ║                                                           ║
    ║          ⚠️  HIGH RISK - USE AT YOUR OWN RISK  ⚠️        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    asyncio.run(main())
