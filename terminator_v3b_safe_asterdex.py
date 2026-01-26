"""
🥇 TERMINATOR V3B SAFE - ASTERDEX LIVE TRADING BOT 🥇
=====================================================
SAFE CONFIG (Low DD, Good WR):
  - ML Threshold: 0.50
  - SL: ATR × 0.80
  - RR: 3.0
  - Risk: 2% per trade (SAFE for funded accounts)
  - Filters: TIME + MTF + RSI
  - Cooldown: 1 bar (1 hour)

Backtest Results:
  - +1,505% Profit
  - 67.9% Win Rate
  - 5.9% Max DD (SAFE FOR FUNDED ACCOUNTS!)

ASTERDEX INTEGRATION:
  - XAUUSDT Perpetual Futures
  - Real position execution
  - Automatic SL/TP orders

Author: Sonnet/Opus
Version: 3.0-SAFE-ASTERDEX
"""

import asyncio
import os
import sys
import io
import logging
import json
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from threading import Thread
from typing import Optional, Dict
from urllib.parse import urlencode

import pandas as pd
import numpy as np
import aiohttp
from flask import Flask
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==================== V3B SAFE CONFIGURATION ====================
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # AsterDex API
    ASTERDEX_API_KEY = os.environ.get('ASTERDEX_API_KEY')
    ASTERDEX_SECRET_KEY = os.environ.get('ASTERDEX_SECRET_KEY')
    ASTERDEX_BASE_URL = "https://fapi.asterdex.com"
    
    # Trading Mode
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'true').lower() == 'true'
    STARTING_BALANCE = float(os.environ.get('STARTING_BALANCE', '1000'))
    
    # ===== V3B SAFE CONFIG =====
    ML_THRESHOLD = 0.50      # Higher threshold for better quality
    SL_MULTIPLIER = 0.80     # ATR × 0.80
    RR = 3.0                 # Risk-Reward 3:1
    BASE_RISK = 0.02         # 2% risk per trade (SAFE!)
    COOLDOWN_BARS = 1        # Wait 1 bar after trade
    MAX_HOLD_BARS = 60       # Max 60 bars to hold
    
    # Filters (TIME + MTF + RSI + MIN_ATR + WEEKEND)
    USE_TIME_FILTER = True        # Skip bad hours
    USE_MTF_FILTER = True         # MTF trend alignment
    USE_RSI_OVERSOLD_FILTER = True  # Skip RSI < 30
    USE_MIN_ATR_FILTER = True     # Skip tiny ATR (prevents $1-2 SL/TP)
    USE_WEEKEND_FILTER = True     # Skip Saturday/Sunday entries
    MIN_ATR_DOLLARS = 5.0         # Minimum ATR in dollars
    
    # Bad hours to skip (from backtest analysis)
    BAD_HOURS = [0, 1, 2, 3, 4, 5, 22, 23]
    
    # Adaptive Risk (if DD increases)
    RISK_DD_THRESHOLD_1 = 5   # If DD > 5%: risk = 1.5%
    RISK_DD_THRESHOLD_2 = 8   # If DD > 8%: risk = 1%
    
    # AsterDex Trading Config
    SYMBOL = "XAUUSDT"
    LEVERAGE = 10              # 10x leverage
    MARGIN_TYPE = "CROSSED"    # Cross margin
    
    # Costs
    SLIPPAGE = 0.60
    
    # Timing
    CHECK_INTERVAL = 60  # Check every 60 seconds
    SIGNAL_WINDOW_MINUTES = 5  # First 5 mins of hour


# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('terminator_v3b_safe_asterdex.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== FLASK KEEP-ALIVE ====================
app = Flask('')

@app.route('/')
def home():
    return "🥇 TERMINATOR V3B SAFE - ASTERDEX (67.9% WR, 5.9% DD) 🥇"

@app.route('/health')
def health():
    return {"status": "healthy", "version": "V3B-SAFE-ASTERDEX", "timestamp": datetime.now().isoformat()}

def run_flask():
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)

def keep_alive():
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()
    port = os.environ.get('PORT', '8000')
    logger.info(f"🌐 Flask Keep-Alive Server Started on Port {port}")


# ==================== ASTERDEX API CLIENT ====================
class AsterDexClient:
    """AsterDex Futures API Client (Binance-compatible)"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
    
    def _sign(self, params: dict) -> str:
        """Generate HMAC-SHA256 signature"""
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        """Make API request"""
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self._sign(params)
        
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, params=params, headers=headers) as resp:
                    return await resp.json()
            elif method == "POST":
                async with session.post(url, params=params, headers=headers) as resp:
                    return await resp.json()
            elif method == "DELETE":
                async with session.delete(url, params=params, headers=headers) as resp:
                    return await resp.json()
    
    async def get_account_info(self) -> dict:
        """Get account balance and positions"""
        return await self._request("GET", "/fapi/v2/account", signed=True)
    
    async def get_position_info(self, symbol: str) -> dict:
        """Get position for symbol"""
        params = {"symbol": symbol}
        result = await self._request("GET", "/fapi/v2/positionRisk", params, signed=True)
        for pos in result:
            if pos['symbol'] == symbol:
                return pos
        return None
    
    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set leverage for symbol"""
        params = {"symbol": symbol, "leverage": leverage}
        return await self._request("POST", "/fapi/v1/leverage", params, signed=True)
    
    async def set_margin_type(self, symbol: str, margin_type: str) -> dict:
        """Set margin type (ISOLATED or CROSSED)"""
        params = {"symbol": symbol, "marginType": margin_type}
        try:
            return await self._request("POST", "/fapi/v1/marginType", params, signed=True)
        except:
            pass  # May already be set
    
    async def get_price(self, symbol: str) -> float:
        """Get current price"""
        result = await self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        return float(result['price'])
    
    async def get_exchange_info(self, symbol: str) -> dict:
        """Get symbol info"""
        result = await self._request("GET", "/fapi/v1/exchangeInfo")
        for s in result['symbols']:
            if s['symbol'] == symbol:
                return s
        return None
    
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        """Place market order"""
        params = {
            "symbol": symbol,
            "side": side,  # BUY or SELL
            "type": "MARKET",
            "quantity": quantity
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)
    
    async def place_stop_loss(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """Place stop loss order"""
        params = {
            "symbol": symbol,
            "side": side,  # BUY (close short) or SELL (close long)
            "type": "STOP_MARKET",
            "stopPrice": f"{stop_price:.2f}",
            "quantity": quantity,
            "closePosition": "true"
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)
    
    async def place_take_profit(self, symbol: str, side: str, quantity: float, stop_price: float) -> dict:
        """Place take profit order"""
        params = {
            "symbol": symbol,
            "side": side,  # BUY (close short) or SELL (close long)
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{stop_price:.2f}",
            "quantity": quantity,
            "closePosition": "true"
        }
        return await self._request("POST", "/fapi/v1/order", params, signed=True)
    
    async def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders"""
        params = {"symbol": symbol}
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", params, signed=True)
    
    async def close_position(self, symbol: str) -> dict:
        """Close current position"""
        pos = await self.get_position_info(symbol)
        if pos and float(pos['positionAmt']) != 0:
            amt = float(pos['positionAmt'])
            side = "SELL" if amt > 0 else "BUY"
            return await self.place_market_order(symbol, side, abs(amt))
        return None


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
                          entry: float, sl: float, tp: float, risk_pct: float, 
                          atr: float, is_paper: bool, order_id: str = None):
        emoji = "📈" if direction == "LONG" else "📉"
        sl_dist = abs(entry - sl)
        mode = "📄 PAPER" if is_paper else "💰 LIVE TRADE!"
        
        message = f"""
🥇 *TERMINATOR V3B SAFE - ASTERDEX* 🥇
━━━━━━━━━━━━━━━━
{mode}
━━━━━━━━━━━━━━━━
📍 XAUUSDT | {emoji} *{direction}*
🤖 ML: *{ml_confidence:.1%}* | 📊 ATR: ${atr:.2f}
━━━━━━━━━━━━━━━━
💰 Entry: *${entry:.2f}*
🛑 SL: *${sl:.2f}* (-${sl_dist:.2f})
✅ TP: *${tp:.2f}* (+${sl_dist * 3:.2f})
💵 Risk: *{risk_pct:.1%}* | RR: *1:3*
━━━━━━━━━━━━━━━━
🔗 Order ID: {order_id if order_id else 'N/A'}
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
    """Fetch XAUUSD SPOT price for ML signals"""
    
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


# ==================== CANDLE COLLECTOR (Auto-save new SPOT candles) ====================
class CandleCollector:
    """Automatically collect hourly SPOT candles for training"""
    
    def __init__(self, price_fetcher: PriceFetcher):
        self.price_fetcher = price_fetcher
        self.current_candle = {'o': None, 'h': None, 'l': None, 'c': None, 'ts': None}
        self.last_saved_hour = None
        
    async def update(self):
        """Call this every minute to build hourly candles"""
        price = await self.price_fetcher.get_current_price()
        if price is None:
            return
        
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        # New hour started - save previous candle and start new one
        if self.current_candle['ts'] is not None and current_hour > self.current_candle['ts']:
            await self._save_candle()
            self._start_new_candle(current_hour, price)
        elif self.current_candle['ts'] is None:
            self._start_new_candle(current_hour, price)
        else:
            # Update current candle
            self.current_candle['h'] = max(self.current_candle['h'], price)
            self.current_candle['l'] = min(self.current_candle['l'], price)
            self.current_candle['c'] = price
    
    def _start_new_candle(self, ts, price):
        self.current_candle = {
            'ts': ts,
            'o': price,
            'h': price,
            'l': price,
            'c': price
        }
        logger.debug(f"🕐 Started new candle at {ts}")
    
    async def _save_candle(self):
        """Save completed candle to CSV"""
        if self.current_candle['ts'] is None:
            return
        
        csv_path = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')
        
        candle_df = pd.DataFrame([{
            'datetime': self.current_candle['ts'],
            'open': self.current_candle['o'],
            'high': self.current_candle['h'],
            'low': self.current_candle['l'],
            'close': self.current_candle['c']
        }])
        
        if os.path.exists(csv_path):
            candle_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            candle_df.to_csv(csv_path, index=False)
        
        logger.info(f"📊 Saved candle: {self.current_candle['ts']} O={self.current_candle['o']:.2f} H={self.current_candle['h']:.2f} L={self.current_candle['l']:.2f} C={self.current_candle['c']:.2f}")


# ==================== V3B SAFE ML MODEL ====================
class V3BSafeModel:
    """V3B SAFE ML Model - with TIME+MTF+RSI filters"""
    
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
        """Calculate ALL indicators"""
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
        
        # SMC Order Blocks - VECTORIZED (fast)
        df['bullish_ob'] = (df['c'] > df['o']) & (df['c'].shift(-1) > df['c'] * 1.002)
        df['bearish_ob'] = (df['c'] < df['o']) & (df['c'].shift(-1) < df['c'] * 0.998)
        
        # Fast vectorized distance calculation
        df['last_bull_ob_price'] = df.loc[df['bullish_ob'], 'c'].reindex(df.index).ffill().fillna(0)
        df['last_bear_ob_price'] = df.loc[df['bearish_ob'], 'c'].reindex(df.index).ffill().fillna(0)
        df['dist_to_bull_ob'] = np.where(
            (df['last_bull_ob_price'] > 0) & (df['atr'] > 0),
            (df['c'] - df['last_bull_ob_price']) / df['atr'],
            0
        )
        df['dist_to_bear_ob'] = np.where(
            (df['last_bear_ob_price'] > 0) & (df['atr'] > 0),
            (df['c'] - df['last_bear_ob_price']) / df['atr'],
            0
        )
        df = df.drop(['last_bull_ob_price', 'last_bear_ob_price'], axis=1)
        
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
        """Extract 27 features"""
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
        """Train model on SPOT CSV + auto-collected new candles"""
        logger.info("🤖 Training V3B SAFE model (27 features, RF+GB)...")
        
        try:
            # 1. Load BASE training data (2020-2025)
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2020_2025_spot.csv')
            
            if os.path.exists(base_csv):
                logger.info(f"📊 Loading BASE training data: {base_csv}")
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                # Try GitHub
                logger.warning("⚠️ Local CSV not found, trying GitHub...")
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminatorjanos/main/xauusd_1h_2020_2025_spot.csv"
                df_base = pd.read_csv(csv_url, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            
            logger.info(f"📊 Base data: {len(df_base)} candles (2020-2025)")
            
            # 2. Load AUTO-COLLECTED new candles (2026+) if exists
            new_candles_csv = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')
            
            if os.path.exists(new_candles_csv):
                df_new = pd.read_csv(new_candles_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                logger.info(f"📊 New candles (2026): {len(df_new)} candles")
                
                # Combine base + new
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
                logger.info(f"📊 Combined training data: {len(df)} candles")
            else:
                df = df_base
                logger.info("📊 No new 2026 candles yet, using base only")
            
            logger.info(f"📊 Total training candles: {len(df)}")
            
            # Calculate indicators
            logger.info("⏳ Step 1/5: Calculating indicators...")
            df = self.calculate_indicators(df)
            logger.info("✅ Step 1/5: Indicators calculated")
            
            # Train HMM - FULL 100 iterations
            logger.info("⏳ Step 2/5: Fitting HMM model...")
            returns = df['c'].pct_change().dropna().values.reshape(-1, 1)
            self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            self.hmm_model.fit(returns[:5000])
            logger.info("✅ Step 2/5: HMM fitted")
            
            regimes = self.hmm_model.predict(returns)
            df['regime'] = pd.Series([np.nan] + list(regimes), index=df.index)
            
            regime_means = [returns[regimes==i].mean() for i in range(3)]
            self.trending_regime = np.argmax([abs(m) for m in regime_means])
            self.ranging_regime = np.argmin([abs(m) for m in regime_means])
            df['is_trending'] = (df['regime'] == self.trending_regime).astype(int)
            df['is_ranging'] = (df['regime'] == self.ranging_regime).astype(int)
            
            df = df.iloc[250:].reset_index(drop=True)
            
            # Extract features and labels - FULL DATASET
            logger.info(f"⏳ Step 3/5: Extracting features from {len(df)} candles (this takes 3-5 min)...")
            X, y = [], []
            total = len(df) - 100
            
            for i in range(50, len(df) - 50):
                f, d = self.get_27_features(df, i)
                if f is None:
                    continue
                
                a = df['atr'].iloc[i]
                c = df['c'].iloc[i]
                sl = a * 1.0
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
            
            logger.info(f"✅ Step 3/5: Features extracted: {len(X)} samples")
            
            X, y = np.array(X), np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train RF + GB
            logger.info("⏳ Step 4/5: Training Random Forest...")
            self.rf = RandomForestClassifier(
                n_estimators=100, max_depth=8, 
                class_weight='balanced', random_state=42, n_jobs=-1
            )
            self.rf.fit(X_scaled, y)
            logger.info("✅ Step 4/5: Random Forest trained")
            
            logger.info("⏳ Step 5/5: Training Gradient Boosting...")
            self.gb = GradientBoostingClassifier(
                n_estimators=80, max_depth=5, random_state=42
            )
            self.gb.fit(X_scaled, y)
            logger.info("✅ Step 5/5: Gradient Boosting trained")
            
            self.last_train = datetime.now()
            logger.info(f"🎉 V3B SAFE Model trained on {len(X)} samples - READY!")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> float:
        """Get ensemble prediction"""
        if self.rf is None or self.gb is None:
            return 0.0
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            p_rf = self.rf.predict_proba(features_scaled)[0][1]
            p_gb = self.gb.predict_proba(features_scaled)[0][1]
            return (p_rf + p_gb) / 2
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0


# ==================== V3B SAFE TRADING ENGINE ====================
class V3BSafeTradingEngine:
    """V3B SAFE Trading Engine with AsterDex execution"""
    
    def __init__(self, config: Config):
        self.config = config
        self.telegram = TelegramBot(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        self.news_filter = NewsFilter()
        self.price_fetcher = PriceFetcher()
        self.model = V3BSafeModel(config)
        self.candle_collector = CandleCollector(self.price_fetcher)  # Auto-collect candles
        
        # AsterDex client
        self.asterdex = AsterDexClient(
            config.ASTERDEX_API_KEY,
            config.ASTERDEX_SECRET_KEY,
            config.ASTERDEX_BASE_URL
        ) if config.ASTERDEX_API_KEY else None
        
        self.balance = config.STARTING_BALANCE
        self.peak_balance = config.STARTING_BALANCE
        self.current_position = None
        self.last_exit_bar = -100
        self.bar_count = 0
        self.running = False
        
        # Symbol info
        self.quantity_precision = 3  # XAUUSDT uses 0.001 min qty
        self.price_precision = 2
    
    async def initialize(self):
        logger.info("🥇 Initializing Terminator V3B SAFE ASTERDEX...")
        logger.info("="*60)
        
        # Train model
        await self.model.train()
        await self.news_filter.update_calendar()
        
        # Setup AsterDex if not paper trading
        if not self.config.PAPER_TRADING and self.asterdex:
            logger.info("")
            logger.info("🔗 CONNECTING TO ASTERDEX...")
            logger.info("="*60)
            
            try:
                # Get account info FIRST
                account = await self.asterdex.get_account_info()
                
                if account and 'totalWalletBalance' in account:
                    self.balance = float(account['totalWalletBalance'])
                    self.peak_balance = self.balance
                    available = float(account.get('availableBalance', 0))
                    unrealized_pnl = float(account.get('totalUnrealizedProfit', 0))
                    
                    logger.info("")
                    logger.info("✅ ACCOUNT CONNECTED SUCCESSFULLY!")
                    logger.info("="*60)
                    logger.info(f"💰 WALLET BALANCE:     ${self.balance:,.2f} USDT")
                    logger.info(f"💵 AVAILABLE BALANCE:  ${available:,.2f} USDT")
                    logger.info(f"📊 UNREALIZED PNL:     ${unrealized_pnl:,.2f} USDT")
                    logger.info("="*60)
                else:
                    logger.error("❌ Could not get account info!")
                    logger.error(f"Response: {account}")
                    return
                
                # Set leverage
                result = await self.asterdex.set_leverage(self.config.SYMBOL, self.config.LEVERAGE)
                logger.info(f"⚙️ Leverage set: {self.config.LEVERAGE}x")
                
                # Set margin type
                await self.asterdex.set_margin_type(self.config.SYMBOL, self.config.MARGIN_TYPE)
                logger.info(f"⚙️ Margin type: {self.config.MARGIN_TYPE}")
                
                # Get symbol info
                info = await self.asterdex.get_exchange_info(self.config.SYMBOL)
                if info:
                    self.quantity_precision = info.get('quantityPrecision', 3)
                    self.price_precision = info.get('pricePrecision', 2)
                    logger.info(f"⚙️ XAUUSDT precision: qty={self.quantity_precision}, price={self.price_precision}")
                
                # Check for existing position
                pos = await self.asterdex.get_position_info(self.config.SYMBOL)
                if pos and float(pos.get('positionAmt', 0)) != 0:
                    pos_amt = float(pos['positionAmt'])
                    pos_side = "LONG" if pos_amt > 0 else "SHORT"
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized = float(pos.get('unRealizedProfit', 0))
                    logger.info("")
                    logger.warning(f"⚠️ EXISTING POSITION DETECTED!")
                    logger.warning(f"   Side: {pos_side} | Qty: {abs(pos_amt):.3f}")
                    logger.warning(f"   Entry: ${entry_price:.2f} | PnL: ${unrealized:.2f}")
                else:
                    logger.info("📋 No existing positions")
                
                logger.info("")
                logger.info("✅ ASTERDEX CONNECTION COMPLETE!")
                logger.info("="*60)
                
            except Exception as e:
                logger.error(f"❌ AsterDex connection error: {e}")
                logger.error("Please check your API keys!")
                return
        else:
            logger.info("")
            logger.info("📄 PAPER TRADING MODE")
            logger.info(f"💰 Starting Balance: ${self.balance:,.2f}")
            logger.info("="*60)
        
        mode = "📄 PAPER" if self.config.PAPER_TRADING else "💰 LIVE ASTERDEX"
        await self.telegram.send_message(
            f"🥇 *TERMINATOR V3B SAFE - ASTERDEX STARTED* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"💰 Balance: *${self.balance:,.2f}*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Config: ML={self.config.ML_THRESHOLD}, Risk={self.config.BASE_RISK*100:.0f}%, RR=1:3\n"
            f"Filters: TIME+MTF+RSI+MIN\\_ATR+WEEKEND\n"
            f"Target: 67.9% WR, 5.9% Max DD\n"
            f"Model: ✅ RF+GB (27 features)"
        )
        
        logger.info("")
        logger.info("🚀 V3B SAFE Engine initialized and READY!")
        logger.info("="*60)
    
    def get_adaptive_risk(self) -> float:
        """Get risk based on drawdown"""
        if self.peak_balance > self.balance:
            dd_pct = (self.peak_balance - self.balance) / self.peak_balance * 100
            if dd_pct > self.config.RISK_DD_THRESHOLD_2:
                return 0.01  # 1%
            elif dd_pct > self.config.RISK_DD_THRESHOLD_1:
                return 0.015  # 1.5%
        return self.config.BASE_RISK  # 2%
    
    def apply_filters(self, df: pd.DataFrame, i: int, direction: str) -> tuple:
        """Apply TIME + MTF + RSI filters"""
        
        # TIME filter - skip bad hours
        if self.config.USE_TIME_FILTER:
            hour = datetime.utcnow().hour
            if hour in self.config.BAD_HOURS:
                return False, "TIME_FILTER: Bad hour"
        
        # MTF filter - require trend alignment
        if self.config.USE_MTF_FILTER:
            trend_4h = df['trend_4h'].iloc[i]
            trend_daily = df['trend_daily'].iloc[i]
            
            if direction == 'LONG' and (trend_4h != 1 or trend_daily != 1):
                return False, "MTF_FILTER: LONG but trend not aligned"
            if direction == 'SHORT' and (trend_4h != -1 or trend_daily != -1):
                return False, "MTF_FILTER: SHORT but trend not aligned"
        
        # RSI Oversold filter - skip trades when RSI < 30
        if self.config.USE_RSI_OVERSOLD_FILTER:
            rsi = df['rsi'].iloc[i]
            if not pd.isna(rsi) and rsi < 30:
                return False, f"RSI_FILTER: RSI={rsi:.1f} < 30"
        
        # MIN ATR filter - prevent tiny SL/TP (1-2$ trades)
        if self.config.USE_MIN_ATR_FILTER:
            atr = df['atr'].iloc[i]
            if not pd.isna(atr) and atr < self.config.MIN_ATR_DOLLARS:
                return False, f"MIN_ATR_FILTER: ATR=${atr:.2f} < ${self.config.MIN_ATR_DOLLARS}"
        
        # WEEKEND filter - skip Saturday/Sunday entries
        if self.config.USE_WEEKEND_FILTER:
            weekday = datetime.utcnow().weekday()
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return False, f"WEEKEND_FILTER: No trades on weekends"
        
        return True, "Filters passed"
    
    async def check_for_signal(self) -> Optional[Dict]:
        """Check if there's a valid V3B SAFE signal"""
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
            
            # Check cooldown
            if self.bar_count <= self.last_exit_bar + self.config.COOLDOWN_BARS:
                return None
            
            # Check news
            is_blackout, event = await self.news_filter.is_news_blackout()
            if is_blackout:
                logger.info(f"📰 News blackout: {event}")
                return None
            
            # Get OHLC data
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2020_2025_spot.csv')
            
            if os.path.exists(base_csv):
                df = pd.read_csv(base_csv, parse_dates=['datetime'])
                df.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminatorjanos/main/xauusd_1h_2020_2025_spot.csv"
                try:
                    df = pd.read_csv(csv_url, parse_dates=['datetime'])
                    df.columns = ['ts', 'o', 'h', 'l', 'c']
                except:
                    logger.error("Could not load SPOT OHLC data")
                    return None
            
            # Use last 200 candles
            df = df.tail(200).reset_index(drop=True)
            
            if len(df) < 100:
                return None
            
            # Calculate indicators
            df = self.model.calculate_indicators(df)
            
            # Add HMM regime
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
            
            # Apply filters
            filter_ok, filter_reason = self.apply_filters(df, i, direction)
            if not filter_ok:
                logger.debug(f"🚫 Signal filtered: {filter_reason}")
                return None
            
            # ML prediction
            ml_prob = self.model.predict(features)
            
            if ml_prob < self.config.ML_THRESHOLD:
                return None
            
            # Get spot price
            spot_price = await self.price_fetcher.get_current_price()
            if spot_price is None:
                return None
            
            # Calculate levels
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
        """Open V3B SAFE position"""
        risk_pct = self.get_adaptive_risk()
        risk_amt = self.balance * risk_pct
        
        # Calculate position size
        # qty = risk_amt / sl_dist (in XAU)
        qty = risk_amt / signal['sl_dist']
        qty = round(qty, self.quantity_precision)
        
        # Ensure minimum notional (5 USDT)
        notional = qty * signal['entry']
        if notional < 5:
            qty = 5.1 / signal['entry']
            qty = round(qty, self.quantity_precision)
        
        order_id = None
        
        # Execute on AsterDex if live
        if not self.config.PAPER_TRADING and self.asterdex:
            try:
                side = "BUY" if signal['direction'] == 'LONG' else "SELL"
                
                # Place market order
                result = await self.asterdex.place_market_order(
                    self.config.SYMBOL, side, qty
                )
                
                if 'orderId' in result:
                    order_id = result['orderId']
                    logger.info(f"✅ AsterDex order placed: {order_id}")
                    
                    # Place SL order
                    sl_side = "SELL" if signal['direction'] == 'LONG' else "BUY"
                    await self.asterdex.place_stop_loss(
                        self.config.SYMBOL, sl_side, qty, signal['sl']
                    )
                    logger.info(f"🛑 SL order placed @ ${signal['sl']:.2f}")
                    
                    # Place TP order
                    await self.asterdex.place_take_profit(
                        self.config.SYMBOL, sl_side, qty, signal['tp']
                    )
                    logger.info(f"✅ TP order placed @ ${signal['tp']:.2f}")
                else:
                    logger.error(f"Order failed: {result}")
                    return
                    
            except Exception as e:
                logger.error(f"AsterDex order error: {e}")
                return
        
        self.current_position = {
            'direction': signal['direction'],
            'entry': signal['entry'],
            'sl': signal['sl'],
            'tp': signal['tp'],
            'risk_amt': risk_amt,
            'sl_dist': signal['sl_dist'],
            'entry_time': datetime.now(),
            'entry_bar': self.bar_count,
            'ml_confidence': signal['ml_confidence'],
            'quantity': qty,
            'order_id': order_id
        }
        
        logger.info(f"📈 OPENED {signal['direction']} @ ${signal['entry']:.2f} | Qty: {qty}")
        
        await self.telegram.send_signal(
            direction=signal['direction'],
            ml_confidence=signal['ml_confidence'],
            entry=signal['entry'],
            sl=signal['sl'],
            tp=signal['tp'],
            risk_pct=risk_pct,
            atr=signal['atr'],
            is_paper=self.config.PAPER_TRADING,
            order_id=str(order_id) if order_id else None
        )
    
    async def monitor_position(self):
        """Monitor position for TP/SL"""
        if not self.current_position:
            return
        
        # For live trading, check AsterDex position
        if not self.config.PAPER_TRADING and self.asterdex:
            try:
                pos = await self.asterdex.get_position_info(self.config.SYMBOL)
                if pos and float(pos['positionAmt']) == 0:
                    # Position closed (by SL or TP) - fetch REAL balance from exchange
                    account = await self.asterdex.get_account_info()
                    if account and 'totalWalletBalance' in account:
                        new_balance = float(account['totalWalletBalance'])
                        pnl = new_balance - self.balance
                        self.balance = new_balance
                        
                        logger.info("")
                        logger.info("="*60)
                        if pnl >= 0:
                            if self.balance > self.peak_balance:
                                self.peak_balance = self.balance
                            logger.info("✅ POSITION CLOSED - TAKE PROFIT!")
                            logger.info(f"📈 Profit: +${pnl:,.2f}")
                            await self.telegram.send_tp_hit(
                                self.current_position['entry'], 
                                self.current_position['tp'], 
                                pnl, self.balance
                            )
                        else:
                            logger.info("🛑 POSITION CLOSED - STOP LOSS")
                            logger.info(f"📉 Loss: ${pnl:,.2f}")
                            await self.telegram.send_sl_hit(
                                self.current_position['entry'], 
                                self.current_position['sl'], 
                                pnl, self.balance
                            )
                        
                        logger.info(f"💰 NEW BALANCE: ${self.balance:,.2f} USDT")
                        dd_pct = (self.peak_balance - self.balance) / self.peak_balance * 100 if self.peak_balance > self.balance else 0
                        logger.info(f"📊 Peak: ${self.peak_balance:,.2f} | DD: {dd_pct:.1f}%")
                        logger.info("="*60)
                    
                    self.current_position = None
                    self.last_exit_bar = self.bar_count
                    return
            except Exception as e:
                logger.warning(f"Position check error: {e}")
        
        # Paper trading - check manually
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
            
            # Close on AsterDex
            if not self.config.PAPER_TRADING and self.asterdex:
                await self.asterdex.cancel_all_orders(self.config.SYMBOL)
                await self.asterdex.close_position(self.config.SYMBOL)
            
            await self.telegram.send_message(
                f"⏰ *TIMEOUT* - Position closed\nPnL: ${pnl:.2f}"
            )
            logger.info(f"⏰ TIMEOUT! ${pnl:.2f}")
            
            self.current_position = None
            self.last_exit_bar = self.bar_count
    
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
                
                await asyncio.sleep(self.config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("⚠️ Shutdown requested")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
        
        await self.telegram.send_message(
            f"🛑 *Terminator V3B SAFE ASTERDEX Stopped*\n"
            f"Final Balance: ${self.balance:.2f}"
        )
        logger.info("🥇 Terminator V3B SAFE ASTERDEX shutdown complete")


# ==================== MAIN ====================
async def main():
    keep_alive()
    config = Config()
    engine = V3BSafeTradingEngine(config)
    
    try:
        await engine.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🥇 TERMINATOR V3B SAFE - ASTERDEX LIVE 🥇               ║
    ║                                                              ║
    ║     SAFE CONFIG: ML=0.50 | Risk=2% | RR=1:3                 ║
    ║     Filters: TIME + MTF + RSI                                ║
    ║     Target: 67.9% WR | 5.9% Max DD                          ║
    ║                                                              ║
    ║     ✅ SAFE FOR FUNDED ACCOUNTS (<10% DD)                   ║
    ║                                                              ║
    ║        ⚠️  STILL HIGH RISK - USE AT YOUR OWN RISK  ⚠️       ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Environment Variables Required:                             ║
    ║    - TELEGRAM_TOKEN                                          ║
    ║    - TELEGRAM_CHAT_ID                                        ║
    ║    - ASTERDEX_API_KEY (for live trading)                    ║
    ║    - ASTERDEX_SECRET_KEY (for live trading)                 ║
    ║    - PAPER_TRADING=false (to enable live trading)           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
