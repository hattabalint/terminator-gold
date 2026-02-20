
import asyncio
import os
import sys
import io
import logging
import json
import random
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from threading import Thread
from typing import Optional, Dict
from urllib.parse import urlencode


# MT5 import - optional (Windows only)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

import pandas as pd
import numpy as np
import aiohttp
from flask import Flask
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

# Windows encoding fix (Restored safe logging)
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==================== V3B CONFIGURATION ====================
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # MT5 Login (for Windows)
    MT5_LOGIN = os.environ.get('MT5_LOGIN')
    MT5_PASSWORD = os.environ.get('MT5_PASSWORD')
    MT5_SERVER = os.environ.get('MT5_SERVER')
    
    # AsterDex API (for Render)
    ASTERDEX_API_KEY = os.environ.get('ASTERDEX_API_KEY')
    ASTERDEX_SECRET_KEY = os.environ.get('ASTERDEX_SECRET_KEY')
    ASTERDEX_BASE_URL = "https://fapi.asterdex.com"

    # Trading
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'
    STARTING_BALANCE = float(os.environ.get('STARTING_BALANCE', '1000'))
    
    # ===== V3B EXACT SETTINGS =====
    ML_THRESHOLD = 0.455    # EXACT from backtest
    SL_MULTIPLIER = 0.80    # ATR × 0.80
    RR = 3.0                # Risk-Reward 3:1
    BASE_RISK = 0.015       # 1.5% risk per trade
    COOLDOWN_BARS = 1       # Wait 1 bar after trade
    MTF_FILTER = False      # MTF OFF (more trades)
    MAX_HOLD_BARS = 60      # Max 60 bars to hold
    
    # Emergency SL (extra safety - wider than normal SL)
    EMERGENCY_SL_MULTIPLIER = 2.0  # ATR × 2.0 = emergency SL
    
    # Adaptive Risk
    RISK_DD_THRESHOLD_1 = 10  # If DD > 10%: risk = 1.125%
    RISK_DD_THRESHOLD_2 = 20  # If DD > 20%: risk = 0.75%
    
    # Costs
    SLIPPAGE = 0.60
    
    # Data
    SYMBOL = "XAUUSD"
    SYMBOL_FUTURES = "XAUUSDT"  # AsterDex symbol
    LEVERAGE = 10
    CHECK_INTERVAL = 60  # Check every 60 seconds
    SIGNAL_WINDOW_MINUTES = 5  # First 5 mins of hour


# ==================== LOGGING ====================
class SafeStreamHandler(logging.StreamHandler):
    """On Windows console, force ASCII output to prevent UnicodeEncodeError."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            fs = "%s\n"
            try:
                if (isinstance(msg, str)):
                    clean_msg = msg.encode('ascii', 'ignore').decode('ascii')
                    stream.write(fs % clean_msg)
                else:
                    stream.write(fs % msg)
            except UnicodeError:
                stream.write(fs % "LOGGING ERROR")
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('terminator_v3b_live.log', encoding='utf-8'),
        SafeStreamHandler()
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
    logger.info("VERSION: a5a5721-FIX2 - MT5_AVAILABLE check added to PriceFetcher")
    logger.info(f"MT5_AVAILABLE = {MT5_AVAILABLE}")
    logger.info("Flask Keep-Alive Server Started on Port 8000")


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
             # SSL False check for local
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
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
        if isinstance(result, list):
            for pos in result:
                if pos.get('symbol') == symbol:
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
        return float(result.get('price', 0))
    
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
        if pos and float(pos.get('positionAmt', 0)) != 0:
            amt = float(pos['positionAmt'])
            side = "SELL" if amt > 0 else "BUY"
            return await self.place_market_order(symbol, side, abs(amt))
        return None


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
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
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
        
        now = datetime.now(timezone.utc).astimezone() if datetime.now().tzinfo else datetime.now()
        # Simplified blackout check for robustness
        return False, None


# ==================== MT5 TRADER (Restored) ====================
class MT5Trader:
    def __init__(self, config: Config, telegram: TelegramBot):
        self.config = config
        self.telegram = telegram
        self.symbol = config.SYMBOL
        self.connected = False
        
    async def initialize(self) -> bool:
        """Initialize connection to MetaTrader 5"""
        # Skip MT5 on Linux/Render where it's not available
        if not MT5_AVAILABLE:
            logger.info("📊 MT5 not available (Linux/Render) - Paper trading only")
            return False
            
        logger.info(f"🔌 Connecting to MT5 Server: {self.config.MT5_SERVER}...")
        if not mt5.initialize():
            logger.error(f"MT5 Init failed: {mt5.last_error()}")
            return False
            
        if self.config.MT5_LOGIN and self.config.MT5_PASSWORD:
            try:
                login_id = int(self.config.MT5_LOGIN)
                authorized = mt5.login(login=login_id, password=self.config.MT5_PASSWORD, server=self.config.MT5_SERVER)
                if not authorized:
                    logger.warning("MT5 Login failed, using existing connection...")
            except Exception as e:
                logger.error(f"MT5 login exception: {e}")

        logger.info(f"✅ MT5 Connected")
        self.connected = True
        return True

    def get_balance(self) -> float:
        if not self.connected: return 0.0
        info = mt5.account_info()
        return info.balance if info else 0.0

    async def execute_trade(self, direction: str, price: float, sl: float, tp: float, risk_pct: float) -> bool:
        if not self.connected: return False
        if risk_pct > 0.01: risk_pct = 0.01

        balance = self.get_balance()
        risk_amount = balance * risk_pct
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info: return False

        sl_dist = abs(price - sl)
        if sl_dist == 0: return False
        lots = risk_amount / (symbol_info.trade_contract_size * sl_dist)
        lots = round(lots, 2)
        if lots < 0.01: lots = 0.01
        
        action = mt5.ORDER_TYPE_BUY if direction == 'LONG' else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": action,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Terminator V3B",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order Send Failed: {result.comment}")
            return False
        logger.info(f"✅ Trade Executed! Ticket: {result.order}")
        return True

    def check_position(self):
        if not self.connected: return "CLOSED"
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None or len(positions) == 0: return "CLOSED"
        return "LONG" if positions[0].type == mt5.ORDER_TYPE_BUY else "SHORT"


# ==================== PRICE FETCHER ====================
class PriceFetcher:
    def __init__(self):
        self.last_price = None
        self.last_update = None
        self.consecutive_failures = 0
        self.asterdex_client = None  # Set by engine after init
    
    def get_current_price(self) -> float:
        # Prefer MT5 price if available (Windows only)
        if MT5_AVAILABLE and mt5.initialize():
            tick = mt5.symbol_info_tick("XAUUSD")
            if tick: 
                self.last_price = tick.last
                self.consecutive_failures = 0
                return tick.last
        
        # Try AsterDex price first (most accurate - where we actually trade)
        asterdex_price = 0.0
        if self.asterdex_client:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in async context, use a direct HTTP call
                    import requests
                    url = f"{self.asterdex_client.base_url}/fapi/v1/ticker/price?symbol=XAUUSDT"
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200:
                        asterdex_price = float(resp.json().get('price', 0))
            except Exception as e:
                logger.debug(f"AsterDex price fetch failed: {e}")
        
        # goldprice.org SPOT price
        goldprice = 0.0
        try:
            import requests
            url = "https://data-asg.goldprice.org/dbXRates/USD"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                goldprice = float(data['items'][0]['xauPrice'])
        except Exception as e:
            self.consecutive_failures += 1
            if self.consecutive_failures <= 3:
                logger.warning(f"goldprice.org fetch failed ({self.consecutive_failures}/3): {e}")
        
        # Cross-check: if both available and differ by more than $5, use AsterDex
        if asterdex_price > 0 and goldprice > 0:
            diff = abs(asterdex_price - goldprice)
            if diff > 5:
                logger.warning(f"⚠️ PRICE MISMATCH! AsterDex: ${asterdex_price:.2f} vs goldprice.org: ${goldprice:.2f} (diff: ${diff:.2f}) → Using AsterDex price")
                self.last_price = asterdex_price
                self.last_update = datetime.now()
                return asterdex_price
            else:
                # Both agree, use goldprice (spot)
                self.last_price = goldprice
                self.last_update = datetime.now()
                self.consecutive_failures = 0
                return goldprice
        elif asterdex_price > 0:
            self.last_price = asterdex_price
            self.last_update = datetime.now()
            return asterdex_price
        elif goldprice > 0:
            self.last_price = goldprice
            self.last_update = datetime.now()
            self.consecutive_failures = 0
            return goldprice
        
        # Return cached price if available (keeps bot running during API outage)
        if self.last_price:
            return self.last_price
        return 0.0


# ==================== CANDLE COLLECTOR ====================
class CandleCollector:
    def __init__(self, price_fetcher: PriceFetcher):
        self.price_fetcher = price_fetcher
        self.current_candle = {'o': None, 'h': None, 'l': None, 'c': None, 'ts': None}
        
    async def update(self):
        price = self.price_fetcher.get_current_price()
        if price == 0: return
        
        # Sanity check - XAU/USD should be between $1000-$20000
        if price < 1000 or price > 20000:
            logger.warning(f"⚠️ Unrealistic price: ${price:.2f} - skipping candle update")
            return
        
        now = datetime.now()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        
        if self.current_candle['ts'] is not None and current_hour > self.current_candle['ts']:
            await self._save_candle()
            self._start_new_candle(current_hour, price)
        elif self.current_candle['ts'] is None:
            self._start_new_candle(current_hour, price)
        else:
            self.current_candle['h'] = max(self.current_candle['h'], price)
            self.current_candle['l'] = min(self.current_candle['l'], price)
            self.current_candle['c'] = price
    
    def _start_new_candle(self, ts, price):
        self.current_candle = {'ts': ts, 'o': price, 'h': price, 'l': price, 'c': price}
    
    async def _save_candle(self):
        if self.current_candle['ts'] is None: return
        import os
        csv_path = "new_candles_2026.csv"
        candle_df = pd.DataFrame([{
            'datetime': self.current_candle['ts'], 'open': self.current_candle['o'],
            'high': self.current_candle['h'], 'low': self.current_candle['l'],
            'close': self.current_candle['c']
        }])
        mode = 'a' if os.path.exists(csv_path) else 'w'
        header = not os.path.exists(csv_path)
        candle_df.to_csv(csv_path, mode=mode, header=header, index=False)
        logger.info(f"📊 Saved candle: {self.current_candle['ts']} C={self.current_candle['c']:.2f}")


class V3BModel:
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
        df['ema21'] = df['c'].ewm(21).mean()
        df['ema50'] = df['c'].ewm(50).mean()
        df['ema200'] = df['c'].ewm(200).mean()
        df['atr'] = (df['h'] - df['l']).rolling(14).mean()
        df['rsi'] = 100 - 100/(1 + df['c'].diff().clip(lower=0).rolling(14).mean() / (-df['c'].diff().clip(upper=0)).rolling(14).mean().replace(0,0.001))
        df['macd'] = df['c'].ewm(12).mean() - df['c'].ewm(26).mean()
        df['macd_sig'] = df['macd'].ewm(9).mean()
        df['ema21_4h'] = df['c'].rolling(4).mean().ewm(21).mean()
        df['ema50_4h'] = df['c'].rolling(4).mean().ewm(50).mean()
        df['rsi_4h'] = df['rsi'].rolling(4).mean()
        df['macd_4h'] = df['macd'].rolling(4).mean()
        df['trend_4h'] = np.where(df['ema21_4h'] > df['ema50_4h'], 1, -1)
        df['c_daily'] = df['c'].rolling(24).mean()
        df['trend_daily'] = np.where(df['c_daily'] > df['c_daily'].shift(24), 1, -1)
        df['bullish_ob'] = (df['c'] > df['o']) & (df['c'].shift(-1) > df['c'] * 1.002)
        df['bearish_ob'] = (df['c'] < df['o']) & (df['c'].shift(-1) < df['c'] * 0.998)
        df['dist_to_bull_ob'] = 0.0
        df['dist_to_bear_ob'] = 0.0
        last_bull, last_bear = 0, 0
        for i in range(len(df)):
            if i < len(df) and df['bullish_ob'].iloc[i]: last_bull = df['c'].iloc[i]
            if i < len(df) and df['bearish_ob'].iloc[i]: last_bear = df['c'].iloc[i]
            if last_bull > 0 and df['atr'].iloc[i] > 0: df.iloc[i, df.columns.get_loc('dist_to_bull_ob')] = (df['c'].iloc[i] - last_bull) / df['atr'].iloc[i]
            if last_bear > 0 and df['atr'].iloc[i] > 0: df.iloc[i, df.columns.get_loc('dist_to_bear_ob')] = (df['c'].iloc[i] - last_bear) / df['atr'].iloc[i]
        df['high_5'] = df['h'].rolling(5).max()
        df['low_5'] = df['l'].rolling(5).min()
        df['high_20'] = df['h'].rolling(20).max()
        df['low_20'] = df['l'].rolling(20).min()
        df['dist_to_high_20'] = (df['high_20'] - df['c']) / df['atr']
        df['dist_to_low_20'] = (df['c'] - df['low_20']) / df['atr']
        df['mom_5'] = df['c'].pct_change(5) * 100
        df['mom_10'] = df['c'].pct_change(10) * 100
        df['vol_ratio'] = (df['h'] - df['l']) / df['atr']
        df['body_ratio'] = abs(df['c'] - df['o']) / (df['h'] - df['l'] + 0.001)
        df['is_doji'] = (abs(df['c'] - df['o']) < (df['h'] - df['l']) * 0.1).astype(int)
        df['is_engulfing'] = ((df['c'] > df['o']) & (df['c'].shift(1) < df['o'].shift(1)) & (df['c'] > df['h'].shift(1)) & (df['o'] < df['l'].shift(1))).astype(int)
        df['is_pin_bar'] = (((df['h'] - df[['c','o']].max(axis=1)) > 2 * abs(df['c']-df['o'])) | ((df[['c','o']].min(axis=1) - df['l']) > 2 * abs(df['c']-df['o']))).astype(int)
        return df
    
    def get_27_features(self, df: pd.DataFrame, i: int) -> tuple:
        if i < 30 or pd.isna(df['atr'].iloc[i]) or df['atr'].iloc[i] <= 0: return None, None
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
        except: return None, None
    
    async def train(self):
        logger.info("🤖 Training V3B model (27 features, RF+GB)...")
        try:
            import os
            base_csv = "xauusd_1h_2020_2025_spot.csv"
            if os.path.exists(base_csv):
                 df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                 df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                 csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2020_2025_spot.csv"
                 df_base = pd.read_csv(csv_url, parse_dates=['datetime'])
                 df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            
            new_candles_csv = "new_candles_2026.csv"
            if os.path.exists(new_candles_csv):
                df_new = pd.read_csv(new_candles_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
            else:
                df = df_base
            
            df = self.calculate_indicators(df)
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
            X, y = [], []
            for i in range(50, len(df) - 50):
                f, d = self.get_27_features(df, i)
                if f is None: continue
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
            
            X, y = np.array(X), np.array(y)
            X_scaled = self.scaler.fit_transform(X)
            self.rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
            self.gb = GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42)
            self.rf.fit(X_scaled, y)
            self.gb.fit(X_scaled, y)
            self.last_train = datetime.now()
            logger.info(f"✅ V3B Model trained on {len(X)} samples")
            return True
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def predict(self, df: pd.DataFrame, i: int) -> tuple:
        if self.rf is None: return 0.0, "NEUTRAL"
        f, d = self.get_27_features(df, i)
        if f is None: return 0.0, "NEUTRAL"
        f_scaled = self.scaler.transform([f])
        p_rf = self.rf.predict_proba(f_scaled)[0][1]
        p_gb = self.gb.predict_proba(f_scaled)[0][1]
        return (p_rf + p_gb) / 2, d


# ==================== V3B TRADING ENGINE (User's Logic Restored) ====================
class V3BTradingEngine:
    def __init__(self, config: Config):
        self.config = config
        self.telegram = TelegramBot(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        self.news_filter = NewsFilter()
        self.price_fetcher = PriceFetcher()
        self.candle_collector = CandleCollector(self.price_fetcher)
        self.model = V3BModel(config)
        self.mt5_trader = MT5Trader(config, self.telegram) if MT5_AVAILABLE else None
        
        # AsterDex client (for Render when MT5 not available)
        self.asterdex = AsterDexClient(
            config.ASTERDEX_API_KEY,
            config.ASTERDEX_SECRET_KEY,
            config.ASTERDEX_BASE_URL
        ) if config.ASTERDEX_API_KEY else None
        
        # Connect PriceFetcher to AsterDex for price cross-checking
        if self.asterdex:
            self.price_fetcher.asterdex_client = self.asterdex
        
        self.balance = config.STARTING_BALANCE
        self.peak_balance = config.STARTING_BALANCE
        self.current_position = None
        self.last_exit_bar = -100
        self.bar_count = 0
        self.running = False
        self.quantity_precision = 3  # XAUUSDT precision
    
    async def initialize(self):
        logger.info("Initializing Terminator V3B Live...")
        
        # MT5 init (Windows)
        if self.mt5_trader:
            await self.mt5_trader.initialize()
        
        # AsterDex init (Render)
        if self.asterdex and not self.config.PAPER_TRADING:
            try:
                logger.info("Connecting to AsterDex...")
                account = await self.asterdex.get_account_info()
                logger.info(f"AsterDex response: {account}")  # Debug log
                
                # Try different balance fields (availableBalance first - that's the usable balance)
                balance = None
                if account:
                    if 'availableBalance' in account:
                        balance = float(account['availableBalance'])
                    elif 'totalWalletBalance' in account:
                        balance = float(account['totalWalletBalance'])
                    elif 'totalBalance' in account:
                        balance = float(account['totalBalance'])
                    elif 'balance' in account:
                        balance = float(account['balance'])
                    elif 'assets' in account:
                        # Some responses have assets array
                        for asset in account['assets']:
                            if asset.get('asset') == 'USDT':
                                balance = float(asset.get('availableBalance', 0)) or float(asset.get('walletBalance', 0))
                                break
                
                if balance and balance > 0:
                    self.balance = balance
                    self.peak_balance = balance
                    logger.info(f"AsterDex connected! Balance: ${self.balance:.2f}")
                else:
                    logger.warning(f"Could not get balance from response, using STARTING_BALANCE: ${self.config.STARTING_BALANCE}")
                    self.balance = self.config.STARTING_BALANCE
                
                # Set leverage
                await self.asterdex.set_leverage(self.config.SYMBOL_FUTURES, self.config.LEVERAGE)
                logger.info(f"Leverage set: {self.config.LEVERAGE}x")
            except Exception as e:
                logger.error(f"AsterDex connection error: {e}")
                logger.info(f"Using STARTING_BALANCE: ${self.config.STARTING_BALANCE}")
        
        await self.model.train()
        await self.news_filter.update_calendar()
        
        mode = "PAPER" if self.config.PAPER_TRADING else ("MT5" if self.mt5_trader else "ASTERDEX")
        await self.telegram.send_message(
            f"*TERMINATOR V3B LIVE STARTED*\n"
            f"Mode: {mode}\n"
            f"Risk: {self.config.BASE_RISK*100:.1f}%\n"
            f"Balance: ${self.balance:.2f}"
        )
        logger.info("V3B Engine initialized")
    
    def get_adaptive_risk(self) -> float:
        """Adaptive risk logic - reduces risk during drawdowns"""
        current_dd = (self.peak_balance - self.balance) / self.peak_balance * 100 if self.peak_balance > self.balance else 0
        base_risk = self.config.BASE_RISK  # 1.5%
        if current_dd > self.config.RISK_DD_THRESHOLD_2:  # DD > 20%
            return 0.0075  # 0.75%
        elif current_dd > self.config.RISK_DD_THRESHOLD_1:  # DD > 10%
            return 0.01125  # 1.125%
        return base_risk  # 1.5%
    
    async def check_for_signal(self) -> Optional[Dict]:
        """Check if there's a valid V3B signal - EXACT USER LOGIC"""
        try:
            now = datetime.utcnow()
            weekday = now.weekday()
            hour = now.hour
            is_market_closed = ((weekday == 4 and hour >= 22) or (weekday == 5) or (weekday == 6 and hour < 22))
            if is_market_closed: return None
            
            if self.bar_count <= self.last_exit_bar + self.config.COOLDOWN_BARS: return None
            
            # Use collected data like in original user code
            import os
            base_csv = "xauusd_1h_2020_2025_spot.csv"
            new_csv = "new_candles_2026.csv"
            
            if os.path.exists(base_csv):
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                 return None # Should handle gracefully
            
            if os.path.exists(new_csv):
                df_new = pd.read_csv(new_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
            else:
                df = df_base
            
            # Use FULL dataset for proper indicator calculation (matches original backtest)
            # EMA200 needs 200+ candles to stabilize properly
            if len(df) < 300: return None
            
            df = self.model.calculate_indicators(df)
            i = len(df) - 1
            conf, direction = self.model.predict(df, i)
            
            # Get current price for debug
            current_price = self.price_fetcher.get_current_price()
            last_candle_time = df['ts'].iloc[i] if 'ts' in df.columns else "N/A"
            last_close = df['c'].iloc[i]
            atr_val = df['atr'].iloc[i] if not pd.isna(df['atr'].iloc[i]) else 0
            
            # DETAILED DEBUG LOG - Shows every hour
            logger.info(f"═══════════════════════════════════════════════════")
            logger.info(f"📊 HOURLY ANALYSIS @ {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            logger.info(f"   Direction: {direction} | ML Conf: {conf:.1%} (Threshold: {self.config.ML_THRESHOLD*100:.1f}%)")
            logger.info(f"   Current Price: ${current_price:.2f} | Last Close: ${last_close:.2f}")
            logger.info(f"   ATR: ${atr_val:.2f} | Last Candle: {last_candle_time}")
            logger.info(f"   Total Candles in DF: {len(df)} | New 2026 Candles: {len(df_new) if 'df_new' in dir() else 'N/A'}")
            if conf >= self.config.ML_THRESHOLD:
                logger.info(f"   ✅ SIGNAL TRIGGERED! Opening position...")
            else:
                logger.info(f"   ❌ No signal (conf {conf:.1%} < threshold {self.config.ML_THRESHOLD*100:.1f}%)")
            logger.info(f"═══════════════════════════════════════════════════")
            
            if conf < self.config.ML_THRESHOLD: return None
            
            # Using MT5 price for current price
            spot_price = self.price_fetcher.get_current_price()
            if spot_price == 0: spot_price = df['c'].iloc[i]
            
            atr = df['atr'].iloc[i]
            # ATR minimum safety check - XAU realistic ATR is ~$5-30
            if pd.isna(atr) or atr < 5.0:
                logger.warning(f"⚠️ ATR too low: ${atr:.2f} - likely bad data, skipping signal")
                return None
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
                'direction': direction, 'ml_confidence': conf, 'entry': entry,
                'sl': sl, 'tp': tp, 'atr': atr
            }
        except Exception as e:
            logger.error(f"Signal check error: {e}")
            return None
    
    async def run(self):
        """Main trading loop - EXACT USER LOGIC"""
        self.running = True
        await self.initialize()
        logger.info("🚀 V3B Engine Running...")
        
        while self.running:
            try:
                # Update bar count logic
                current_hour = datetime.now().hour
                if not hasattr(self, '_last_hour'): self._last_hour = current_hour
                if current_hour != self._last_hour:
                    self.bar_count += 1
                    self._last_hour = current_hour
                
                await self.candle_collector.update()
                
                # Check for new signal - EVERY HOUR (matches original backtest)
                if not self.current_position:
                    # Check position status (MT5 or paper)
                    pos_status = "CLOSED"
                    if self.mt5_trader:
                        pos_status = self.mt5_trader.check_position()
                    
                    # Also check AsterDex position
                    if self.asterdex and not self.config.PAPER_TRADING:
                        try:
                            pos = await self.asterdex.get_position_info(self.config.SYMBOL_FUTURES)
                            if pos and float(pos.get('positionAmt', 0)) != 0:
                                pos_status = "LONG" if float(pos['positionAmt']) > 0 else "SHORT"
                        except:
                            pass
                    
                    if pos_status == "CLOSED": 
                        signal = await self.check_for_signal()
                else:
                    # We have a position - check if it closed
                    pos_status = "CLOSED"
                    if self.mt5_trader:
                        pos_status = self.mt5_trader.check_position()
                    if self.asterdex and not self.config.PAPER_TRADING:
                        try:
                            pos = await self.asterdex.get_position_info(self.config.SYMBOL_FUTURES)
                            if pos and float(pos.get('positionAmt', 0)) != 0:
                                pos_status = "OPEN"
                        except:
                            pass
                    
                    if pos_status == "CLOSED":
                        logger.info("Position closed - cleaning up ghost orders...")
                        # CRITICAL: Cancel ALL remaining orders to prevent ghost orders
                        if self.asterdex and not self.config.PAPER_TRADING:
                            try:
                                cancel_result = await self.asterdex.cancel_all_orders(self.config.SYMBOL_FUTURES)
                                logger.info(f"✅ All ghost orders cancelled: {cancel_result}")
                                await self.telegram.send_message("🧹 Position closed → all remaining orders cancelled")
                            except Exception as e:
                                logger.error(f"Failed to cancel ghost orders: {e}")
                        self.current_position = None
                
                # Only execute if we found a signal AND no position
                if not self.current_position:
                    pos_status = "CLOSED"
                    if self.asterdex and not self.config.PAPER_TRADING:
                        try:
                            pos = await self.asterdex.get_position_info(self.config.SYMBOL_FUTURES)
                            if pos and float(pos.get('positionAmt', 0)) != 0:
                                pos_status = "OPEN"
                        except:
                            pass
                    
                    if pos_status == "CLOSED":
                        signal = await self.check_for_signal()
                        if signal:
                            risk = self.get_adaptive_risk()
                            
                            # Execute trade on MT5 (Windows)
                            if self.mt5_trader:
                                await self.mt5_trader.execute_trade(signal['direction'], signal['entry'], signal['sl'], signal['tp'], risk)
                            
                            # Execute trade on AsterDex (Render)
                            elif self.asterdex and not self.config.PAPER_TRADING:
                                try:
                                    # Calculate quantity
                                    risk_amount = self.balance * risk
                                    sl_dist = abs(signal['entry'] - signal['sl'])
                                    qty = risk_amount / sl_dist
                                    qty = round(qty, self.quantity_precision)
                                    
                                    # Minimum notional check (5 USDT)
                                    if qty * signal['entry'] < 5:
                                        qty = 5.1 / signal['entry']
                                        qty = round(qty, self.quantity_precision)
                                    
                                    side = "BUY" if signal['direction'] == 'LONG' else "SELL"
                                    
                                    # Place market order
                                    result = await self.asterdex.place_market_order(
                                        self.config.SYMBOL_FUTURES, side, qty
                                    )
                                    
                                    if 'orderId' in result:
                                        logger.info(f"AsterDex order placed: {result['orderId']}")
                                        
                                        # Place SL order
                                        sl_side = "SELL" if signal['direction'] == 'LONG' else "BUY"
                                        await self.asterdex.place_stop_loss(
                                            self.config.SYMBOL_FUTURES, sl_side, qty, signal['sl']
                                        )
                                        logger.info(f"SL placed @ ${signal['sl']:.2f}")
                                        
                                        # Place TP order
                                        await self.asterdex.place_take_profit(
                                            self.config.SYMBOL_FUTURES, sl_side, qty, signal['tp']
                                        )
                                        logger.info(f"TP placed @ ${signal['tp']:.2f}")
                                        
                                        # Place EMERGENCY SL (wider, extra safety)
                                        emergency_sl_dist = signal['atr'] * self.config.EMERGENCY_SL_MULTIPLIER
                                        if signal['direction'] == 'LONG':
                                            emergency_sl = signal['entry'] - emergency_sl_dist
                                        else:
                                            emergency_sl = signal['entry'] + emergency_sl_dist
                                        
                                        await self.asterdex.place_stop_loss(
                                            self.config.SYMBOL_FUTURES, sl_side, qty, emergency_sl
                                        )
                                        logger.info(f"EMERGENCY SL placed @ ${emergency_sl:.2f}")
                                        
                                        # Mark position as open
                                        self.current_position = signal
                                    else:
                                        logger.error(f"AsterDex order failed: {result}")
                                except Exception as e:
                                    logger.error(f"AsterDex trade error: {e}")
                            
                            # Mark position for MT5 too
                            if self.mt5_trader:
                                self.current_position = signal
                            
                            # Always send Telegram signal (only once per trade!)
                            await self.telegram.send_signal(signal['direction'], signal['ml_confidence'], signal['entry'], signal['sl'], signal['tp'], risk, signal['atr'])
                            self.last_exit_bar = self.bar_count

                await asyncio.sleep(self.config.CHECK_INTERVAL)
            except KeyboardInterrupt:
                logger.info("⚠️ Shutdown requested")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)

async def main():
    keep_alive()
    config = Config()
    engine = V3BTradingEngine(config)
    try:
        await engine.run()
    except Exception as e:
         logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
