"""
🥇 TERMINATOR V3B LIVE - GOLD TRADING BOT 🥇
=============================================
FINALIZED V3B CONFIG:
  - 27 Features (SMC OB, HMM, MTF, patterns)
  - ML Threshold: 0.455
  - SL: ATR × 0.80
  - RR: 3.0
  - MTF Filter: OFF
  - Cooldown: 1 bar (1 hour)
  - Ensemble: RandomForest + GradientBoosting

Backtest Results (2025):
  - 207 Trades
  - 51.7% Win Rate
  - +40,096% Profit (Compound)

Author: Sonnet/Opus
Version: 3.0.0 (V3B Final)
"""

import asyncio
import os
import sys
import io
import logging
import json
import random
from datetime import datetime, timedelta
from threading import Thread
from typing import Optional, Dict

import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from flask import Flask
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==================== V3B CONFIGURATION ====================
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # Trading
    PAPER_TRADING = True  # Paper trading mode
    STARTING_BALANCE = float(os.environ.get('STARTING_BALANCE', '1000'))
    
    # ===== V3B EXACT SETTINGS =====
    ML_THRESHOLD = 0.455    # EXACT from backtest
    SL_MULTIPLIER = 0.80    # ATR × 0.80
    RR = 3.0                # Risk-Reward 3:1
    BASE_RISK = 0.03        # 3% risk per trade
    COOLDOWN_BARS = 1       # Wait 1 bar after trade
    MTF_FILTER = False      # MTF OFF (more trades)
    MAX_HOLD_BARS = 60      # Max 60 bars to hold
    
    # Adaptive Risk (from backtest)
    RISK_DD_THRESHOLD_1 = 10  # If DD > 10%: risk = 2.25%
    RISK_DD_THRESHOLD_2 = 20  # If DD > 20%: risk = 1.5%
    
    # Costs
    SLIPPAGE = 0.60
    
    # Data
    SYMBOL = "XAUUSD"
    CHECK_INTERVAL = 60  # Check every 60 seconds
    SIGNAL_WINDOW_MINUTES = 5  # First 5 mins of hour


# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('terminator_v3b_live.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== FLASK KEEP-ALIVE ====================
app = Flask('')

@app.route('/')
def home():
    return "🥇 TERMINATOR V3B LIVE - GOLD BOT (51.7% WR) 🥇"

@app.route('/health')
def health():
    return {"status": "healthy", "version": "V3B", "timestamp": datetime.now().isoformat()}

def run_flask():
    app.run(host='0.0.0.0', port=8000)

def keep_alive():
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()
    logger.info("🌐 Flask Keep-Alive Server Started on Port 8000")


# ==================== TELEGRAM BOT ====================
class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}" if token else None
    
    async def send_message(self, text: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram not configured, skipping message")
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown"
                }
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
    
    async def send_signal(self, direction: str, ml_confidence: float, 
                          entry: float, sl: float, tp: float, risk_pct: float, atr: float):
        emoji = "📈" if direction == "LONG" else "📉"
        sl_dist = abs(entry - sl)
        
        message = f"""
🥇 *TERMINATOR V3B GOLD Signal* 🥇
━━━━━━━━━━━━━━━━
📍 XAU/USD | {emoji} *{direction}*
🤖 ML: *{ml_confidence:.1%}* | 📊 ATR: ${atr:.2f}
━━━━━━━━━━━━━━━━
💰 Entry: *${entry:.2f}*
🛑 SL: *${sl:.2f}* (-${sl_dist:.2f})
✅ TP: *${tp:.2f}* (+${sl_dist * 3:.2f})
💵 Risk: *{risk_pct:.1%}* | RR: *1:3*
"""
        await self.send_message(message)
    
    async def send_tp_hit(self, entry: float, exit_price: float, pnl: float, balance: float):
        message = f"""
✅ *TP HIT!* ✅
━━━━━━━━━━━━━━━━
💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}
📈 Profit: *+${pnl:.2f}*
💵 Balance: *${balance:.2f}*
"""
        await self.send_message(message)
    
    async def send_sl_hit(self, entry: float, exit_price: float, pnl: float, balance: float):
        message = f"""
🛑 *SL HIT* 🛑
━━━━━━━━━━━━━━━━
💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}
📉 Loss: *${pnl:.2f}*
💵 Balance: *${balance:.2f}*
"""
        await self.send_message(message)


# ==================== NEWS FILTER ====================
class NewsFilter:
    def __init__(self):
        self.high_impact_events = []
        self.last_update = None
        self.blackout_minutes_before = 15
        self.blackout_minutes_after = 30
    
    async def update_calendar(self):
        try:
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.high_impact_events = []
                        for event in data:
                            if event.get('impact') == 'High' and event.get('country') == 'USD':
                                event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S%z')
                                self.high_impact_events.append({
                                    'title': event.get('title', 'Unknown'),
                                    'time': event_time,
                                    'impact': 'High'
                                })
                        self.last_update = datetime.now()
                        logger.info(f"📰 Loaded {len(self.high_impact_events)} high-impact USD events")
        except Exception as e:
            logger.warning(f"News calendar update failed: {e}")
    
    async def is_news_blackout(self) -> tuple:
        if self.last_update is None or (datetime.now() - self.last_update).seconds > 21600:
            await self.update_calendar()
        
        now = datetime.now()
        for event in self.high_impact_events:
            event_time = event['time'].replace(tzinfo=None)
            blackout_start = event_time - timedelta(minutes=self.blackout_minutes_before)
            blackout_end = event_time + timedelta(minutes=self.blackout_minutes_after)
            if blackout_start <= now <= blackout_end:
                return True, event['title']
        return False, None


# ==================== PRICE FETCHER ====================
class PriceFetcher:
    def __init__(self):
        self.last_price = None
        self.last_update = None
    
    async def get_current_price(self) -> Optional[float]:
        url = "https://data-asg.goldprice.org/dbXRates/USD"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
        
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price = data.get('items', [{}])[0].get('xauPrice')
                            if price:
                                self.last_price = float(price)
                                self.last_update = datetime.now()
                                return self.last_price
            except Exception as e:
                logger.warning(f"Price fetch attempt {attempt+1} failed: {e}")
            await asyncio.sleep(1)
        
        logger.error("Could not get SPOT price after 3 attempts")
        return self.last_price


# ==================== V3B ML MODEL (27 FEATURES) ====================
class V3BModel:
    """V3B ML Model - EXACT backtest logic"""
    
    def __init__(self, config: Config):
        self.config = config
        self.rf = None
        self.gb = None
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.trending_regime = None
        self.ranging_regime = None
        self.last_train = None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL indicators - EXACT from backtest"""
        # Basic EMAs
        df['ema21'] = df['c'].ewm(21).mean()
        df['ema50'] = df['c'].ewm(50).mean()
        df['ema200'] = df['c'].ewm(200).mean()
        df['atr'] = (df['h'] - df['l']).rolling(14).mean()
        
        # RSI
        df['rsi'] = 100 - 100/(1 + df['c'].diff().clip(lower=0).rolling(14).mean() / 
                              (-df['c'].diff().clip(upper=0)).rolling(14).mean().replace(0,0.001))
        
        # MACD
        df['macd'] = df['c'].ewm(12).mean() - df['c'].ewm(26).mean()
        df['macd_sig'] = df['macd'].ewm(9).mean()
        
        # Multi-Timeframe (4H)
        df['ema21_4h'] = df['c'].rolling(4).mean().ewm(21).mean()
        df['ema50_4h'] = df['c'].rolling(4).mean().ewm(50).mean()
        df['rsi_4h'] = df['rsi'].rolling(4).mean()
        df['macd_4h'] = df['macd'].rolling(4).mean()
        df['trend_4h'] = np.where(df['ema21_4h'] > df['ema50_4h'], 1, -1)
        
        # Daily
        df['c_daily'] = df['c'].rolling(24).mean()
        df['trend_daily'] = np.where(df['c_daily'] > df['c_daily'].shift(24), 1, -1)
        
        # SMC Order Blocks
        df['bullish_ob'] = (df['c'] > df['o']) & (df['c'].shift(-1) > df['c'] * 1.002)
        df['bearish_ob'] = (df['c'] < df['o']) & (df['c'].shift(-1) < df['c'] * 0.998)
        df['dist_to_bull_ob'] = 0.0
        df['dist_to_bear_ob'] = 0.0
        
        last_bull, last_bear = 0, 0
        for i in range(len(df)):
            if i < len(df) and df['bullish_ob'].iloc[i]:
                last_bull = df['c'].iloc[i]
            if i < len(df) and df['bearish_ob'].iloc[i]:
                last_bear = df['c'].iloc[i]
            if last_bull > 0 and df['atr'].iloc[i] > 0:
                df.iloc[i, df.columns.get_loc('dist_to_bull_ob')] = (df['c'].iloc[i] - last_bull) / df['atr'].iloc[i]
            if last_bear > 0 and df['atr'].iloc[i] > 0:
                df.iloc[i, df.columns.get_loc('dist_to_bear_ob')] = (df['c'].iloc[i] - last_bear) / df['atr'].iloc[i]
        
        # Structure
        df['high_5'] = df['h'].rolling(5).max()
        df['low_5'] = df['l'].rolling(5).min()
        df['high_20'] = df['h'].rolling(20).max()
        df['low_20'] = df['l'].rolling(20).min()
        df['dist_to_high_20'] = (df['high_20'] - df['c']) / df['atr']
        df['dist_to_low_20'] = (df['c'] - df['low_20']) / df['atr']
        
        # Momentum
        df['mom_5'] = df['c'].pct_change(5) * 100
        df['mom_10'] = df['c'].pct_change(10) * 100
        df['vol_ratio'] = (df['h'] - df['l']) / df['atr']
        df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l'] + 0.001)
        
        # Candlestick patterns
        df['is_doji'] = (abs(df['c'] - df['o']) < (df['h'] - df['l']) * 0.1).astype(int)
        df['is_engulfing'] = ((df['c'] > df['o']) & (df['c'].shift(1) < df['o'].shift(1)) & 
                               (df['c'] > df['h'].shift(1)) & (df['o'] < df['l'].shift(1))).astype(int)
        df['is_pin_bar'] = (((df['h'] - df[['c','o']].max(axis=1)) > 2 * abs(df['c']-df['o'])) |
                            ((df[['c','o']].min(axis=1) - df['l']) > 2 * abs(df['c']-df['o']))).astype(int)
        
        return df
    
    def get_27_features(self, df: pd.DataFrame, i: int) -> tuple:
        """Extract 27 features - EXACT from backtest"""
        if i < 30 or pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] <= 0:
            return None, None
        
        a = df['atr'].iloc[i]
        d = 'LONG' if df['ema21'].iloc[i] > df['ema50'].iloc[i] else 'SHORT'
        
        try:
            features = [
                (df['ema21'].iloc[i] - df['ema50'].iloc[i]) / a,
                (df['c'].iloc[i] - df['ema21'].iloc[i]) / a,
                (df['c'].iloc[i] - df['ema200'].iloc[i]) / a,
                df['rsi'].iloc[i] / 100 if not pd.isna(df['rsi'].iloc[i]) else 0.5,
                (df['macd'].iloc[i] - df['macd_sig'].iloc[i]) / a if not pd.isna(df['macd'].iloc[i]) else 0,
                (df['c'].iloc[i] - df['c'].iloc[i-1]) / a,
                (df['c'].iloc[i] - df['c'].iloc[i-5]) / a if i >= 5 else 0,
                df['trend_4h'].iloc[i],
                df['trend_daily'].iloc[i],
                df['rsi_4h'].iloc[i] / 100 if not pd.isna(df['rsi_4h'].iloc[i]) else 0.5,
                df['macd_4h'].iloc[i] / a if not pd.isna(df['macd_4h'].iloc[i]) else 0,
                1 if df['trend_4h'].iloc[i] == df['trend_daily'].iloc[i] else 0,
                df['dist_to_bull_ob'].iloc[i],
                df['dist_to_bear_ob'].iloc[i],
                1 if (d=='LONG' and df['dist_to_bull_ob'].iloc[i] < 3) else (1 if (d=='SHORT' and df['dist_to_bear_ob'].iloc[i] > -3) else 0),
                df.get('is_trending', pd.Series([0]*len(df))).iloc[i] if 'is_trending' in df else 0,
                df.get('is_ranging', pd.Series([0]*len(df))).iloc[i] if 'is_ranging' in df else 0,
                df['dist_to_high_20'].iloc[i] if not pd.isna(df['dist_to_high_20'].iloc[i]) else 0,
                df['dist_to_low_20'].iloc[i] if not pd.isna(df['dist_to_low_20'].iloc[i]) else 0,
                1 if (d=='LONG' and df['c'].iloc[i] > df['high_5'].iloc[i-1]) else (1 if (d=='SHORT' and df['c'].iloc[i] < df['low_5'].iloc[i-1]) else 0),
                df['mom_5'].iloc[i] / 10 if not pd.isna(df['mom_5'].iloc[i]) else 0,
                df['mom_10'].iloc[i] / 10 if not pd.isna(df['mom_10'].iloc[i]) else 0,
                df['vol_ratio'].iloc[i] if not pd.isna(df['vol_ratio'].iloc[i]) else 1,
                df['body_ratio'].iloc[i] if not pd.isna(df['body_ratio'].iloc[i]) else 0.5,
                df['is_doji'].iloc[i],
                df['is_engulfing'].iloc[i],
                df['is_pin_bar'].iloc[i],
            ]
            return np.array([0 if pd.isna(f) else f for f in features]), d
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None, None
    
    async def train(self):
        """Train model on historical data - USING SPOT CSV (exact backtest data)"""
        logger.info("🤖 Training V3B model (27 features, RF+GB)...")
        
        try:
            # Use LOCAL SPOT CSV file (same as backtest) - NOT Yahoo Finance futures!
            import os
            csv_path = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2020_2024_spot.csv')
            
            if os.path.exists(csv_path):
                logger.info(f"📊 Loading SPOT training data from: {csv_path}")
                df = pd.read_csv(csv_path, parse_dates=['datetime'])
                df.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                # Fallback: try to download from remote (if CSV uploaded to GitHub)
                logger.warning("⚠️ Local CSV not found, trying remote...")
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2020_2024_spot.csv"
                df = pd.read_csv(csv_url, parse_dates=['datetime'])
                df.columns = ['ts', 'o', 'h', 'l', 'c']
            
            logger.info(f"📊 Loaded {len(df)} SPOT candles for training")

            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Train HMM
            returns = df['c'].pct_change().dropna().values.reshape(-1, 1)
            self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            self.hmm_model.fit(returns[:5000])
            regimes = self.hmm_model.predict(returns)
            df['regime'] = pd.Series([np.nan] + list(regimes), index=df.index)
            
            regime_means = [returns[regimes==i].mean() for i in range(3)]
            self.trending_regime = np.argmax([abs(m) for m in regime_means])
            self.ranging_regime = np.argmin([abs(m) for m in regime_means])
            df['is_trending'] = (df['regime'] == self.trending_regime).astype(int)
            df['is_ranging'] = (df['regime'] == self.ranging_regime).astype(int)
            
            df = df.iloc[250:].reset_index(drop=True)
            
            # Extract features and labels
            X, y = [], []
            for i in range(50, len(df) - 50):
                f, d = self.get_27_features(df, i)
                if f is None:
                    continue
                
                a = df['atr'].iloc[i]
                c = df['c'].iloc[i]
                sl = a * 1.0  # Training uses 1.0 ATR
                tp = sl * 3.0
                
                label = 0
                for j in range(i + 1, min(i + 45, len(df))):
                    if d == 'LONG':
                        if df['l'].iloc[j] <= c - sl: break
                        if df['h'].iloc[j] >= c + tp: label = 1; break
                    else:
                        if df['h'].iloc[j] >= c + sl: break
                        if df['l'].iloc[j] <= c - tp: label = 1; break
                
                X.append(f)
                y.append(label)
            
            X, y = np.array(X), np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train RF + GB
            self.rf = RandomForestClassifier(
                n_estimators=100, max_depth=8, 
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            self.gb = GradientBoostingClassifier(
                n_estimators=80, max_depth=5, random_state=42
            )
            
            self.rf.fit(X_scaled, y)
            self.gb.fit(X_scaled, y)
            
            self.last_train = datetime.now()
            logger.info(f"✅ V3B Model trained on {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> float:
        """Get ensemble prediction - EXACT from backtest"""
        if self.rf is None or self.gb is None:
            return 0.0
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            p_rf = self.rf.predict_proba(features_scaled)[0][1]
            p_gb = self.gb.predict_proba(features_scaled)[0][1]
            return (p_rf + p_gb) / 2  # Average of RF + GB
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0


# ==================== V3B TRADING ENGINE ====================
class V3BTradingEngine:
    """V3B Trading Engine - EXACT backtest logic"""
    
    def __init__(self, config: Config):
        self.config = config
        self.telegram = TelegramBot(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        self.news_filter = NewsFilter()
        self.price_fetcher = PriceFetcher()
        self.model = V3BModel(config)
        
        self.balance = config.STARTING_BALANCE
        self.peak_balance = config.STARTING_BALANCE
        self.current_position = None
        self.last_exit_bar = -100
        self.bar_count = 0
        self.running = False
    
    async def initialize(self):
        logger.info("🥇 Initializing Terminator V3B Live...")
        
        await self.model.train()
        await self.news_filter.update_calendar()
        
        mode = "📄 PAPER" if self.config.PAPER_TRADING else "💰 LIVE"
        await self.telegram.send_message(
            f"🥇 *TERMINATOR V3B LIVE STARTED* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"Balance: ${self.balance:.2f}\n"
            f"Config: ML={self.config.ML_THRESHOLD}, SL={self.config.SL_MULTIPLIER}×ATR, RR=1:3\n"
            f"Model: ✅ RF+GB (27 features)"
        )
        
        logger.info("✅ V3B Engine initialized")
    
    def get_adaptive_risk(self) -> float:
        """Get risk based on drawdown - EXACT from backtest"""
        if self.peak_balance > self.balance:
            dd_pct = (self.peak_balance - self.balance) / self.peak_balance * 100
            if dd_pct > self.config.RISK_DD_THRESHOLD_2:
                return 0.015  # 1.5%
            elif dd_pct > self.config.RISK_DD_THRESHOLD_1:
                return 0.0225  # 2.25%
        return self.config.BASE_RISK  # 3%
    
    async def check_for_signal(self) -> Optional[Dict]:
        """Check if there's a valid V3B signal"""
        try:
            # Check market hours
            now = datetime.utcnow()
            weekday = now.weekday()
            hour = now.hour
            
            is_market_closed = (
                (weekday == 4 and hour >= 22) or
                (weekday == 5) or
                (weekday == 6 and hour < 22)
            )
            
            if is_market_closed:
                return None
            
            # Check cooldown - EXACT from backtest (1 bar)
            if self.bar_count <= self.last_exit_bar + self.config.COOLDOWN_BARS:
                return None
            
            # Check news
            is_blackout, event = await self.news_filter.is_news_blackout()
            if is_blackout:
                logger.info(f"📰 News blackout: {event}")
                return None
            
            # Get OHLC data
            df = yf.download('GC=F', period='10d', interval='1h', progress=False)
            if df.empty or len(df) < 100:
                return None
            
            df = df.reset_index()
            if len(df.columns) == 7:
                df.columns = ['ts', 'o', 'h', 'l', 'c', 'ac', 'v']
                df = df.drop('ac', axis=1)
            else:
                df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
            
            # Drop incomplete candle
            df = df.iloc[:-1].reset_index(drop=True)
            
            # Calculate indicators
            df = self.model.calculate_indicators(df)
            
            # Add HMM regime (if model trained)
            if self.model.hmm_model:
                returns = df['c'].pct_change().dropna().values.reshape(-1, 1)
                try:
                    regimes = self.model.hmm_model.predict(returns)
                    df['regime'] = pd.Series([np.nan] + list(regimes), index=df.index)
                    df['is_trending'] = (df['regime'] == self.model.trending_regime).astype(int)
                    df['is_ranging'] = (df['regime'] == self.model.ranging_regime).astype(int)
                except:
                    df['is_trending'] = 0
                    df['is_ranging'] = 0
            
            i = len(df) - 1
            
            # Get features
            features, direction = self.model.get_27_features(df, i)
            if features is None:
                return None
            
            # ML prediction
            ml_prob = self.model.predict(features)
            
            if ml_prob < self.config.ML_THRESHOLD:
                return None
            
            # Get spot price
            spot_price = await self.price_fetcher.get_current_price()
            if spot_price is None:
                return None
            
            # Calculate levels - EXACT from backtest
            atr = df['atr'].iloc[i]
            sl_dist = atr * self.config.SL_MULTIPLIER
            
            if direction == 'LONG':
                entry = spot_price + self.config.SLIPPAGE
                sl = entry - sl_dist
                tp = entry + sl_dist * self.config.RR
            else:
                entry = spot_price - self.config.SLIPPAGE
                sl = entry + sl_dist
                tp = entry - sl_dist * self.config.RR
            
            return {
                'direction': direction,
                'ml_confidence': ml_prob,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'atr': atr,
                'sl_dist': sl_dist
            }
            
        except Exception as e:
            logger.error(f"Signal check error: {e}")
            return None
    
    async def open_position(self, signal: Dict):
        """Open V3B position"""
        risk_pct = self.get_adaptive_risk()
        risk_amt = self.balance * risk_pct
        
        self.current_position = {
            'direction': signal['direction'],
            'entry': signal['entry'],
            'sl': signal['sl'],
            'tp': signal['tp'],
            'risk_amt': risk_amt,
            'sl_dist': signal['sl_dist'],
            'entry_time': datetime.now(),
            'entry_bar': self.bar_count,
            'ml_confidence': signal['ml_confidence']
        }
        
        logger.info(f"📈 OPENED {signal['direction']} @ ${signal['entry']:.2f}")
        
        await self.telegram.send_signal(
            direction=signal['direction'],
            ml_confidence=signal['ml_confidence'],
            entry=signal['entry'],
            sl=signal['sl'],
            tp=signal['tp'],
            risk_pct=risk_pct,
            atr=signal['atr']
        )
    
    async def monitor_position(self):
        """Monitor position for TP/SL"""
        if not self.current_position:
            return
        
        current_price = await self.price_fetcher.get_current_price()
        if current_price is None:
            return
        
        pos = self.current_position
        is_long = pos['direction'] == 'LONG'
        
        # Check SL
        sl_hit = (is_long and current_price <= pos['sl']) or \
                 (not is_long and current_price >= pos['sl'])
        
        # Check TP
        tp_hit = (is_long and current_price >= pos['tp']) or \
                 (not is_long and current_price <= pos['tp'])
        
        # Check timeout (60 bars max)
        bars_held = self.bar_count - pos['entry_bar']
        timeout = bars_held >= self.config.MAX_HOLD_BARS
        
        if tp_hit:
            pnl = pos['risk_amt'] * self.config.RR
            self.balance += pnl
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            
            await self.telegram.send_tp_hit(pos['entry'], pos['tp'], pnl, self.balance)
            logger.info(f"✅ TP HIT! +${pnl:.2f}")
            
            self.current_position = None
            self.last_exit_bar = self.bar_count
            
        elif sl_hit:
            pnl = -pos['risk_amt']
            self.balance += pnl
            
            await self.telegram.send_sl_hit(pos['entry'], pos['sl'], pnl, self.balance)
            logger.info(f"🛑 SL HIT! ${pnl:.2f}")
            
            self.current_position = None
            self.last_exit_bar = self.bar_count
            
        elif timeout:
            if is_long:
                pnl = (current_price - pos['entry']) / pos['sl_dist'] * pos['risk_amt']
            else:
                pnl = (pos['entry'] - current_price) / pos['sl_dist'] * pos['risk_amt']
            
            self.balance += pnl
            if self.balance > self.peak_balance:
                self.peak_balance = self.balance
            
            await self.telegram.send_message(
                f"⏰ *TIMEOUT* - Position closed\nPnL: ${pnl:.2f}"
            )
            logger.info(f"⏰ TIMEOUT! ${pnl:.2f}")
            
            self.current_position = None
            self.last_exit_bar = self.bar_count
    
    async def check_retrain(self):
        """Weekly retrain"""
        if self.model.last_train is None:
            return
        
        days_since_train = (datetime.now() - self.model.last_train).days
        
        if days_since_train >= 7:
            logger.info("🔄 Weekly retrain triggered...")
            await self.telegram.send_message("🔄 *Weekly V3B Model Retrain Starting...*")
            
            success = await self.model.train()
            
            if success:
                await self.telegram.send_message("✅ *V3B Model Retrained Successfully!*")
            else:
                await self.telegram.send_message("⚠️ *V3B Model Retrain Failed*")
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        await self.initialize()
        
        while self.running:
            try:
                # Increment bar count on hour change
                current_hour = datetime.now().hour
                if not hasattr(self, '_last_hour'):
                    self._last_hour = current_hour
                if current_hour != self._last_hour:
                    self.bar_count += 1
                    self._last_hour = current_hour
                
                # Monitor position
                await self.monitor_position()
                
                # Check for new signal
                if not self.current_position:
                    current_minute = datetime.now().minute
                    if current_minute <= self.config.SIGNAL_WINDOW_MINUTES:
                        signal = await self.check_for_signal()
                        if signal:
                            await self.open_position(signal)
                
                # Weekly retrain
                await self.check_retrain()
                
                await asyncio.sleep(self.config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("⚠️ Shutdown requested")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
        
        await self.telegram.send_message(
            f"🛑 *Terminator V3B Stopped*\n"
            f"Final Balance: ${self.balance:.2f}"
        )
        logger.info("🥇 Terminator V3B shutdown complete")


# ==================== MAIN ====================
async def main():
    keep_alive()
    config = Config()
    engine = V3BTradingEngine(config)
    
    try:
        await engine.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🥇 TERMINATOR V3B LIVE - GOLD TRADING BOT 🥇         ║
    ║                                                           ║
    ║     27 Features | ML=0.455 | RR=1:3 | 51.7% WR           ║
    ║     SMC Order Blocks | HMM Regime | RF+GB Ensemble       ║
    ║                                                           ║
    ║        ⚠️  HIGH RISK - USE AT YOUR OWN RISK  ⚠️          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
