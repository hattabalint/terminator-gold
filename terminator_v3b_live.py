# -*- coding: utf-8 -*-
"""
🥇 TERMINATOR V3B + SCALPER V6 LIVE - GOLD TRADING BOT 🥇
============================================================
V3B CONFIG (UNCHANGED):
  - 27 Features (SMC OB, HMM, MTF, patterns)
  - ML Threshold: 0.455
  - SL: ATR × 0.80, RR: 3.0
  - Ensemble: RandomForest + GradientBoosting

SCALPER V6 CONFIG:
  - 41 Features (BB, ADX, RSI div, price_accel, vol_squeeze, etc.)
  - 5 Models: TrendScalper, RangeScalper, FakeBreak, MomBurst, DivHunter
  - HMM 5-State regime detection
  - Stacked Ensemble: RF + GB + LGB + XGB -> LR meta
  - SC Threshold: 0.42 (adaptive), SC RR: 1:5, SC Risk: 2%
  - Trailing Stop: lock +1R profit at +2R
  - SC fires ONLY when V3B does NOT fire

Backtest Results (2025):
  - V3B: 184T / 56.5% WR
  - SC:  111T / 50.5% WR / RR=1:5
  - Combined: 295T / 54.2% WR / +188,847% Profit

Exchange: AsterDex (XAUUSDT Futures)
Author: TradeVersum
Version: 4.0.0 (V3B + Scalper V6)
"""

import asyncio
import hashlib
import hmac as hmac_lib
import io
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from threading import Thread
from typing import Optional, Dict
from urllib.parse import urlencode

import aiohttp
import numpy as np
import pandas as pd
from flask import Flask
from hmmlearn import hmm
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


# ==================== V3B CONFIGURATION ====================
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

    # AsterDex
    ASTERDEX_API_KEY = os.environ.get('ASTERDEX_API_KEY')
    ASTERDEX_SECRET_KEY = os.environ.get('ASTERDEX_SECRET_KEY')
    ASTERDEX_BASE_URL = "https://fapi.asterdex.com"
    ASTERDEX_SYMBOL = "XAUUSDT"
    LEVERAGE = 10

    # Trading
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'
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
    RISK_DD_THRESHOLD_1 = 10   # If DD > 10%: risk = 2.25%
    RISK_DD_THRESHOLD_2 = 20   # If DD > 20%: risk = 1.5%

    # ===== SCALPER V6 SETTINGS =====
    SC_THRESHOLD = 0.42
    SC_RR = 5.0
    SC_RISK = 0.02          # 2% risk per SC trade
    SC_ADAPTIVE_TH = True
    SC_MODEL_SET = 'trend_range'
    SC_LABEL_RR = 2.0       # Training label RR (best AUC)
    SC_MAX_HOLD = 48         # min(RR*12, 48)
    SC_TRAILING_TRIGGER = 2.0  # Lock trailing at +2R
    SC_TRAILING_LOCK = 1.0     # SL moves to +1R profit
    SC_SL_MULT = 1.0          # SL = 1.0 x ATR for SC trades
    SC_ENABLED = True

    # Costs
    SLIPPAGE = 0.60

    # Data
    SYMBOL = "XAUUSD"
    CHECK_INTERVAL = 60         # Check every 60 seconds
    SIGNAL_WINDOW_MINUTES = 5   # First 5 mins of hour

    # Price sanity: max allowed spread between SPOT and AsterDex (in $)
    MAX_PRICE_SPREAD = 20.0


# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('terminator_v3b_live.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== FLASK KEEP-ALIVE ====================
app = Flask('')


@app.route('/')
def home():
    return "🥇 TRADEVERSUM V3B + SCALPER V6 LIVE - GOLD BOT 🥇"


@app.route('/health')
def health():
    return {"status": "healthy", "version": "V3B-3.1+SC-V6", "timestamp": datetime.now().isoformat()}


def run_flask():
    app.run(host='0.0.0.0', port=8000)


def keep_alive():
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()
    logger.info("Flask Keep-Alive Server Started on Port 8000")


# ==================== TELEGRAM BOT ====================
class TelegramBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}" if token else None

    async def send_message(self, text: str):
        if not self.token or not self.chat_id:
            return
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.base_url}/sendMessage"
                payload = {"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"}
                async with session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        logger.error(f"Telegram error: {await resp.text()}")
        except Exception as e:
            logger.error(f"Telegram send error: {e}")

    async def send_signal(self, direction: str, ml_confidence: float,
                          entry: float, sl: float, tp: float, risk_pct: float, atr: float,
                          spot_price: float, asterdex_price: float):
        emoji = "📈" if direction == "LONG" else "📉"
        sl_dist = abs(entry - sl)
        spread = abs(spot_price - asterdex_price)

        message = (
            f"🥇 *TRADEVERSUM V3B Signal* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📍 XAU/USD | {emoji} *{direction}*\n"
            f"🤖 ML: *{ml_confidence:.1%}* | 📊 ATR: ${atr:.2f}\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Entry: *${entry:.2f}*\n"
            f"🛑 SL: *${sl:.2f}* (-${sl_dist:.2f})\n"
            f"✅ TP: *${tp:.2f}* (+${sl_dist * 3:.2f})\n"
            f"💵 Risk: *{risk_pct:.1%}* | RR: *1:3*\n"
            f"📡 SPOT: ${spot_price:.2f} | AsterDex: ${asterdex_price:.2f} (Δ${spread:.2f})"
        )
        await self.send_message(message)

    async def send_tp_hit(self, entry: float, exit_price: float, pnl: float, balance: float):
        message = (
            f"✅ *TP HIT!* ✅\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}\n"
            f"📈 Profit: *+${pnl:.2f}*\n"
            f"💵 Balance: *${balance:.2f}*"
        )
        await self.send_message(message)

    async def send_sl_hit(self, entry: float, exit_price: float, pnl: float, balance: float):
        message = (
            f"🛑 *SL HIT* 🛑\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}\n"
            f"📉 Loss: *${pnl:.2f}*\n"
            f"💵 Balance: *${balance:.2f}*"
        )
        await self.send_message(message)

    async def send_sc_signal(self, direction: str, model_name: str, prob: float,
                              entry: float, sl: float, tp: float, risk_pct: float,
                              atr: float, rr: float):
        emoji = "📈" if direction == "LONG" else "📉"
        sl_dist = abs(entry - sl)
        message = (
            f"⚡ *SCALPER V6 Signal* ⚡\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📍 XAU/USD | {emoji} *{direction}*\n"
            f"🤖 Model: *{model_name}* | Prob: *{prob:.1%}*\n"
            f"📊 ATR: ${atr:.2f} | RR: *1:{rr:.0f}*\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 Entry: *${entry:.2f}*\n"
            f"🛑 SL: *${sl:.2f}* (-${sl_dist:.2f})\n"
            f"✅ TP: *${tp:.2f}* (+${sl_dist * rr:.2f})\n"
            f"💵 Risk: *{risk_pct:.1%}*\n"
            f"🔒 Trailing: lock +1R at +2R"
        )
        await self.send_message(message)

    async def send_sc_trailing_lock(self, direction: str, entry: float, new_sl: float):
        message = (
            f"🔒 *TRAILING STOP LOCKED* 🔒\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚡ Scalper trade reached +2R!\n"
            f"💰 Entry: ${entry:.2f}\n"
            f"🛑 New SL: *${new_sl:.2f}* (+1R profit locked)"
        )
        await self.send_message(message)

    async def send_sc_be_exit(self, entry: float, exit_price: float, pnl: float, balance: float):
        message = (
            f"🔒 *TRAILING STOP HIT* 🔒\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⚡ Scalper +1R profit locked!\n"
            f"💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}\n"
            f"📈 Profit: *+${pnl:.2f}*\n"
            f"💵 Balance: *${balance:.2f}*"
        )
        await self.send_message(message)


# ==================== ASTERDEX API CLIENT ====================
class AsterDexClient:
    """AsterDex Futures API Client (Binance-compatible)"""

    def __init__(self, api_key: str, secret_key: str, base_url: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url

    def _sign(self, params: dict) -> str:
        query_string = urlencode(sorted(params.items()))
        signature = hmac_lib.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        url = f"{self.base_url}{endpoint}"
        headers = {"X-MBX-APIKEY": self.api_key}
        if params is None:
            params = {}
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 5000
            params['signature'] = self._sign(params)
        async with aiohttp.ClientSession() as session:
            try:
                if method == "GET":
                    async with session.get(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        return await resp.json(content_type=None)
                elif method == "POST":
                    async with session.post(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        return await resp.json(content_type=None)
                elif method == "DELETE":
                    async with session.delete(url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        return await resp.json(content_type=None)
            except Exception as e:
                logger.error(f"AsterDex API error [{method} {endpoint}]: {e}")
                return {}

    async def get_account_balance(self) -> float:
        """Get USDT available balance"""
        result = await self._request("GET", "/fapi/v2/account", signed=True)
        if isinstance(result, dict):
            for field in ['availableBalance', 'totalWalletBalance', 'totalBalance']:
                if field in result:
                    return float(result[field])
            for asset in result.get('assets', []):
                if asset.get('asset') == 'USDT':
                    return float(asset.get('availableBalance', 0) or asset.get('walletBalance', 0))
        return 0.0

    async def get_asterdex_price(self) -> Optional[float]:
        """Get current XAUUSDT price from AsterDex"""
        result = await self._request("GET", "/fapi/v1/ticker/price", {"symbol": "XAUUSDT"})
        price = result.get('price')
        if price:
            return float(price)
        return None

    async def set_leverage(self, symbol: str, leverage: int):
        await self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        try:
            await self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": margin_type}, signed=True)
        except Exception:
            pass  # May already be set

    async def get_position(self, symbol: str) -> Optional[dict]:
        result = await self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        if isinstance(result, list):
            for pos in result:
                if pos.get('symbol') == symbol:
                    return pos
        return None

    async def cancel_all_orders(self, symbol: str):
        await self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)

    async def close_position(self, symbol: str):
        """Close any open position on symbol"""
        pos = await self.get_position(symbol)
        if pos:
            amt = float(pos.get('positionAmt', 0))
            if amt != 0:
                side = "SELL" if amt > 0 else "BUY"
                await self._request("POST", "/fapi/v1/order", {
                    "symbol": symbol, "side": side, "type": "MARKET",
                    "quantity": abs(amt), "reduceOnly": "true"
                }, signed=True)
                logger.info(f"Closed position: {amt} {symbol}")

    async def place_market_order_with_sl_tp(
        self, symbol: str, direction: str, quantity: float,
        sl_price: float, tp_price: float
    ) -> bool:
        """Place market order + SL + TP orders"""
        side = "BUY" if direction == "LONG" else "SELL"
        close_side = "SELL" if direction == "LONG" else "BUY"

        # 1. Market entry
        entry_result = await self._request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side,
            "type": "MARKET", "quantity": quantity
        }, signed=True)
        logger.info(f"Entry order result: {entry_result}")

        if 'orderId' not in entry_result and 'code' in entry_result:
            logger.error(f"Entry order failed: {entry_result}")
            return False

        await asyncio.sleep(0.5)

        # 2. Stop Loss
        sl_result = await self._request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": close_side,
            "type": "STOP_MARKET",
            "stopPrice": f"{sl_price:.2f}",
            "closePosition": "true"
        }, signed=True)
        logger.info(f"SL order result: {sl_result}")

        # 3. Take Profit
        tp_result = await self._request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": close_side,
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": f"{tp_price:.2f}",
            "closePosition": "true"
        }, signed=True)
        logger.info(f"TP order result: {tp_result}")

        return True


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
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.high_impact_events = []
                        for event in data:
                            if event.get('impact') == 'High' and event.get('country') == 'USD':
                                event_time = datetime.strptime(event['date'], '%Y-%m-%dT%H:%M:%S%z')
                                self.high_impact_events.append({
                                    'title': event.get('title', 'Unknown'),
                                    'time': event_time,
                                })
                        self.last_update = datetime.now()
                        logger.info(f"Loaded {len(self.high_impact_events)} high-impact USD events")
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
        self.last_spot_price = None
        self.last_update = None

    async def get_spot_price(self) -> Optional[float]:
        """Get gold price - tries 3 sources in order:
        1. AsterDex XAUUSDT (most reliable from Render servers)
        2. goldprice.org (SPOT)
        3. Frankfurter API (SPOT fallback)
        """

        # --- SOURCE 1: AsterDex (no IP blocking, always accessible) ---
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://fapi.asterdex.com/fapi/v1/ticker/price?symbol=XAUUSDT"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        price = data.get('price')
                        if price and float(price) > 100:
                            p = float(price)
                            self.last_spot_price = p
                            self.last_update = datetime.now()
                            logger.debug(f"Price from AsterDex: ${p:.2f}")
                            return p
        except Exception as e:
            logger.warning(f"AsterDex price fetch failed: {e}")

        # --- SOURCE 2: goldprice.org ---
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://data-asg.goldprice.org/dbXRates/USD"
                headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        price = data.get('items', [{}])[0].get('xauPrice')
                        if price:
                            p = float(price)
                            self.last_spot_price = p
                            self.last_update = datetime.now()
                            logger.debug(f"Price from goldprice.org: ${p:.2f}")
                            return p
        except Exception as e:
            logger.warning(f"goldprice.org fetch failed: {e}")

        # --- SOURCE 3: Frankfurter API ---
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.frankfurter.app/latest?from=XAU&to=USD"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        price = data.get('rates', {}).get('USD')
                        if price and float(price) > 100:
                            p = float(price)
                            self.last_spot_price = p
                            self.last_update = datetime.now()
                            logger.debug(f"Price from Frankfurter: ${p:.2f}")
                            return p
        except Exception as e:
            logger.warning(f"Frankfurter fetch failed: {e}")

        # All sources failed - use cache
        if self.last_spot_price:
            logger.warning(f"All price sources failed - using cached price: ${self.last_spot_price:.2f}")
        else:
            logger.error("All price sources failed and no cached price available")
        return self.last_spot_price


# ==================== CANDLE COLLECTOR ====================
class CandleCollector:
    """Collect hourly SPOT candles for training"""

    def __init__(self, price_fetcher: PriceFetcher):
        self.price_fetcher = price_fetcher
        self.current_candle = {'o': None, 'h': None, 'l': None, 'c': None, 'ts': None}

    async def update(self):
        price = await self.price_fetcher.get_spot_price()
        if price is None:
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
        mode = 'a' if os.path.exists(csv_path) else 'w'
        candle_df.to_csv(csv_path, mode=mode, header=not os.path.exists(csv_path), index=False)
        logger.info(f"Saved candle: {self.current_candle['ts']} C={self.current_candle['c']:.2f}")


# ==================== V3B ML MODEL ====================
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
        df['rsi'] = 100 - 100 / (1 + df['c'].diff().clip(lower=0).rolling(14).mean() /
                                  (-df['c'].diff().clip(upper=0)).rolling(14).mean().replace(0, 0.001))
        df['macd'] = df['c'].ewm(12).mean() - df['c'].ewm(26).mean()
        df['macd_sig'] = df['macd'].ewm(9).mean()
        df['ema21_4h'] = df['c'].rolling(4).mean().ewm(21).mean()
        df['ema50_4h'] = df['c'].rolling(4).mean().ewm(50).mean()
        df['rsi_4h'] = df['rsi'].rolling(4).mean()
        df['macd_4h'] = df['macd'].rolling(4).mean()
        df['trend_4h'] = np.where(df['ema21_4h'] > df['ema50_4h'], 1, -1)
        df['c_daily'] = df['c'].rolling(24).mean()
        df['trend_daily'] = np.where(df['c_daily'] > df['c_daily'].shift(24), 1, -1)

        # SMC Order Blocks - EXACT backtest logic (shift(-1) look-ahead)
        df['bullish_ob'] = (df['c'] > df['o']) & (df['c'].shift(-1) > df['c'] * 1.002)
        df['bearish_ob'] = (df['c'] < df['o']) & (df['c'].shift(-1) < df['c'] * 0.998)
        df['dist_to_bull_ob'] = 0.0
        df['dist_to_bear_ob'] = 0.0
        last_bull, last_bear = 0, 0
        for i in range(len(df)):
            if df['bullish_ob'].iloc[i]:
                last_bull = df['c'].iloc[i]
            if df['bearish_ob'].iloc[i]:
                last_bear = df['c'].iloc[i]
            if last_bull > 0 and df['atr'].iloc[i] > 0:
                df.iloc[i, df.columns.get_loc('dist_to_bull_ob')] = (df['c'].iloc[i] - last_bull) / df['atr'].iloc[i]
            if last_bear > 0 and df['atr'].iloc[i] > 0:
                df.iloc[i, df.columns.get_loc('dist_to_bear_ob')] = (df['c'].iloc[i] - last_bear) / df['atr'].iloc[i]

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
        df['is_engulfing'] = (
            (df['c'] > df['o']) & (df['c'].shift(1) < df['o'].shift(1)) &
            (df['c'] > df['h'].shift(1)) & (df['o'] < df['l'].shift(1))
        ).astype(int)
        df['is_pin_bar'] = (
            ((df['h'] - df[['c', 'o']].max(axis=1)) > 2 * abs(df['c'] - df['o'])) |
            ((df[['c', 'o']].min(axis=1) - df['l']) > 2 * abs(df['c'] - df['o']))
        ).astype(int)
        return df

    def get_27_features(self, df: pd.DataFrame, i: int) -> tuple:
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
                (df['c'].iloc[i] - df['c'].iloc[i - 1]) / a,
                (df['c'].iloc[i] - df['c'].iloc[i - 5]) / a if i >= 5 else 0,
                df['trend_4h'].iloc[i],
                df['trend_daily'].iloc[i],
                df['rsi_4h'].iloc[i] / 100 if not pd.isna(df['rsi_4h'].iloc[i]) else 0.5,
                df['macd_4h'].iloc[i] / a if not pd.isna(df['macd_4h'].iloc[i]) else 0,
                1 if df['trend_4h'].iloc[i] == df['trend_daily'].iloc[i] else 0,
                df['dist_to_bull_ob'].iloc[i],
                df['dist_to_bear_ob'].iloc[i],
                1 if (d == 'LONG' and df['dist_to_bull_ob'].iloc[i] < 3) else (
                    1 if (d == 'SHORT' and df['dist_to_bear_ob'].iloc[i] > -3) else 0),
                df['is_trending'].iloc[i] if 'is_trending' in df else 0,
                df['is_ranging'].iloc[i] if 'is_ranging' in df else 0,
                df['dist_to_high_20'].iloc[i] if not pd.isna(df['dist_to_high_20'].iloc[i]) else 0,
                df['dist_to_low_20'].iloc[i] if not pd.isna(df['dist_to_low_20'].iloc[i]) else 0,
                1 if (d == 'LONG' and df['c'].iloc[i] > df['high_5'].iloc[i - 1]) else (
                    1 if (d == 'SHORT' and df['c'].iloc[i] < df['low_5'].iloc[i - 1]) else 0),
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
        logger.info("Training V3B model (27 features, RF+GB)...")
        try:
            # Load base training data
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2020_2025_spot.csv')
            if os.path.exists(base_csv):
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
                logger.info(f"Loaded base data: {len(df_base)} candles")
            else:
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2020_2025_spot.csv"
                logger.warning("Local CSV not found, trying GitHub...")
                df_base = pd.read_csv(csv_url, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']

            # Add 2026 candles if available
            new_csv = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')
            if os.path.exists(new_csv):
                df_new = pd.read_csv(new_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
                logger.info(f"Combined training data: {len(df)} candles (with 2026)")
            else:
                df = df_base

            df = self.calculate_indicators(df)

            # Train HMM
            returns = df['c'].pct_change().dropna().values.reshape(-1, 1)
            self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
            self.hmm_model.fit(returns[:5000])
            regimes = self.hmm_model.predict(returns)
            df['regime'] = pd.Series([np.nan] + list(regimes), index=df.index)
            regime_means = [returns[regimes == i].mean() for i in range(3)]
            self.trending_regime = np.argmax([abs(m) for m in regime_means])
            self.ranging_regime = np.argmin([abs(m) for m in regime_means])
            df['is_trending'] = (df['regime'] == self.trending_regime).astype(int)
            df['is_ranging'] = (df['regime'] == self.ranging_regime).astype(int)

            df = df.iloc[250:].reset_index(drop=True)

            # Build labels (using 1.0 ATR SL like original backtest)
            X, y = [], []
            for i in range(50, len(df) - 50):
                f, d = self.get_27_features(df, i)
                if f is None:
                    continue
                a = df['atr'].iloc[i]
                c = df['c'].iloc[i]
                sl = a * 1.0   # Training label uses 1.0 ATR (original backtest)
                tp = sl * 3.0
                label = 0
                for j in range(i + 1, min(i + 45, len(df))):
                    if d == 'LONG':
                        if df['l'].iloc[j] <= c - sl:
                            break
                        if df['h'].iloc[j] >= c + tp:
                            label = 1
                            break
                    else:
                        if df['h'].iloc[j] >= c + sl:
                            break
                        if df['l'].iloc[j] <= c - tp:
                            label = 1
                            break
                X.append(f)
                y.append(label)

            X, y = np.array(X), np.array(y)
            X_scaled = self.scaler.fit_transform(X)

            self.rf = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced',
                                             random_state=42, n_jobs=-1)
            self.gb = GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42)
            self.rf.fit(X_scaled, y)
            self.gb.fit(X_scaled, y)
            self.last_train = datetime.now()
            logger.info(f"V3B Model trained on {len(X)} samples | pos_rate={y.mean():.2%}")
            return True
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def predict(self, features: np.ndarray) -> float:
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


# ==================== SCALPER V6 MODEL ====================
ACTIVE_SC_MODELS = {
    'all5':           ['TrendScalper', 'RangeScalper', 'FakeBreak', 'MomBurst', 'DivHunter'],
    'trend_range':    ['TrendScalper', 'RangeScalper'],
    'range_fake_div': ['RangeScalper', 'FakeBreak', 'DivHunter'],
    'mom_burst_only': ['MomBurst', 'DivHunter'],
}

REGIME_TO_MODELS = {
    'TREND_UP':    ['TrendScalper', 'MomBurst'],
    'TREND_DOWN':  ['TrendScalper', 'MomBurst'],
    'RANGE_TIGHT': ['RangeScalper', 'FakeBreak', 'DivHunter'],
    'RANGE_WIDE':  ['FakeBreak', 'MomBurst', 'DivHunter'],
    'UNCERTAIN':   [],
}

SC_FEATURE_NAMES = [
    'ema_spread_n', 'c_vs_ema21_n', 'rsi14_n', 'macd_hist_n', 'trend_4h', 'trend_daily', 'tf_align',
    'bb_pct', 'bb_width', 'atr_rank', 'vol_spike', 'vol_ma_ratio',
    'roc5', 'roc14', 'body_ratio', 'upper_wick', 'lower_wick', 'is_doji', 'is_pin_bar', 'is_engulfing',
    'london', 'ny_session', 'overlap', 'hour_n', 'day_of_week_n',
    'tf4h_dist_high', 'tf4h_dist_low', 'tf4h_rsi_n', 'tf4h_trend',
    'dist_to_high_20', 'dist_to_low_20', 'dist_bull_ob', 'dist_bear_ob',
    'bull_div', 'bear_div',
    'hmm_confidence',
    'adx14_n', 'mom_5_n', 'mom_10_n',
    'price_accel', 'vol_squeeze',
]


class ScalperModel:
    """Scalper V6 – 5 specialized models with HMM 5-state regime detection."""

    def __init__(self, config: Config):
        self.config = config
        self.models = {}       # {name: (clf, scaler)}
        self.hmm5_model = None
        self.hmm5_labels = {}  # {state_int: label_str}
        self.trained = False

    @staticmethod
    def add_scalper_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add scalper-specific indicators on top of V3B indicators."""
        # Bollinger Bands
        df['bb_mid'] = df['c'].rolling(20).mean()
        bb_std = df['c'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'].replace(0, 0.001)
        df['bb_pct'] = (df['c'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 0.001)

        # ADX
        plus_dm = df['h'].diff().clip(lower=0)
        minus_dm = (-df['l'].diff()).clip(lower=0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        df['adx14'] = (100 * (plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()).abs() /
                        (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean()).replace(0, 0.001)).rolling(14).mean()

        # ATR percentile rank
        df['atr_rank'] = df['atr'].rolling(200).rank(pct=True)

        # Volume proxy
        hl_range = df['h'] - df['l']
        df['vol_spike'] = (hl_range > hl_range.rolling(20).mean() * 1.5).astype(int)
        df['vol_ma_ratio'] = hl_range / hl_range.rolling(20).mean().replace(0, 0.001)

        # Rate of change
        df['roc5'] = df['c'].pct_change(5) * 100
        df['roc14'] = df['c'].pct_change(14) * 100

        # Session flags (UTC)
        hr = pd.to_datetime(df['ts']).dt.hour
        df['hour'] = hr
        df['london'] = ((hr >= 7) & (hr < 16)).astype(int)
        df['ny_session'] = ((hr >= 13) & (hr < 22)).astype(int)
        df['overlap'] = ((hr >= 13) & (hr < 16)).astype(int)
        df['day_of_week'] = pd.to_datetime(df['ts']).dt.dayofweek

        # Wick ratios
        df['upper_wick'] = (df['h'] - df[['c', 'o']].max(axis=1)) / df['atr'].replace(0, 0.001)
        df['lower_wick'] = (df[['c', 'o']].min(axis=1) - df['l']) / df['atr'].replace(0, 0.001)

        # 4H MTF features
        df['tf4h_pivot_high'] = df['h'].rolling(4).max().shift(1)
        df['tf4h_pivot_low'] = df['l'].rolling(4).min().shift(1)
        a_safe = df['atr'].replace(0, 0.001)
        df['tf4h_dist_high'] = (df['tf4h_pivot_high'] - df['c']) / a_safe
        df['tf4h_dist_low'] = (df['c'] - df['tf4h_pivot_low']) / a_safe
        df['tf4h_trend'] = df['trend_4h']
        df['tf4h_rsi'] = df['rsi'].rolling(4).mean()

        # RSI Divergence
        price_low_5 = df['l'].rolling(5).min()
        rsi_at_low = df['rsi'].rolling(5).min()
        prev_price_low = price_low_5.shift(5)
        prev_rsi_low = rsi_at_low.shift(5)
        df['bull_div'] = ((df['l'] <= price_low_5) &
                          (df['rsi'] > prev_rsi_low) &
                          (df['l'] < prev_price_low)).astype(int)
        price_high_5 = df['h'].rolling(5).max()
        rsi_at_high = df['rsi'].rolling(5).max()
        prev_ph = price_high_5.shift(5)
        prev_rh = rsi_at_high.shift(5)
        df['bear_div'] = ((df['h'] >= price_high_5) &
                          (df['rsi'] < prev_rh) &
                          (df['h'] > prev_ph)).astype(int)

        # V6: Price acceleration & Volatility squeeze
        roc3 = df['c'].pct_change(3)
        df['price_accel'] = roc3 - roc3.shift(3)
        df['vol_squeeze'] = (df['bb_width'] == df['bb_width'].rolling(20).min()).astype(int)

        return df

    def _train_hmm5(self, df: pd.DataFrame):
        """Train 5-state HMM for regime detection."""
        logger.info("[SC-HMM5] Training 5-state HMM...")
        ret = df['c'].pct_change().fillna(0).values
        atr_r = df['atr_rank'].fillna(0.5).values
        vol = df['vol_ma_ratio'].fillna(1.0).clip(0, 5).values
        bb = df['bb_width'].fillna(0.02).clip(0, 0.2).values

        X = np.column_stack([ret, atr_r, vol, bb])
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

        self.hmm5_model = hmm.GaussianHMM(n_components=5, covariance_type='full',
                                            n_iter=200, random_state=42)
        self.hmm5_model.fit(X[:6000])

        states = self.hmm5_model.predict(X)
        state_means = []
        for s in range(5):
            mask = states == s
            if mask.sum() > 0:
                state_means.append((s, abs(ret[mask].mean()), atr_r[mask].mean(), vol[mask].mean()))
            else:
                state_means.append((s, 0, 0, 0))

        sorted_by_ret = sorted(state_means, key=lambda x: x[1], reverse=True)
        sorted_by_atr = sorted(state_means, key=lambda x: x[2])

        self.hmm5_labels = {}
        assigned = set()

        trend_up = sorted_by_ret[0][0]
        trend_down = sorted_by_ret[1][0]
        self.hmm5_labels[trend_up] = 'TREND_UP'
        self.hmm5_labels[trend_down] = 'TREND_DOWN'
        assigned.update([trend_up, trend_down])

        for s, _, atr_m, _ in sorted_by_atr:
            if s not in assigned:
                self.hmm5_labels[s] = 'RANGE_TIGHT'
                assigned.add(s)
                break

        remaining = [s for s in range(5) if s not in assigned]
        if len(remaining) >= 2:
            self.hmm5_labels[remaining[0]] = 'RANGE_WIDE'
            self.hmm5_labels[remaining[1]] = 'UNCERTAIN'
        elif len(remaining) == 1:
            self.hmm5_labels[remaining[0]] = 'RANGE_WIDE'

        logger.info(f"[SC-HMM5] State mapping: {self.hmm5_labels}")

    def add_hmm5_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply HMM5 to dataframe, add hmm_label and hmm_confidence columns."""
        if self.hmm5_model is None:
            df['hmm_label'] = 'UNCERTAIN'
            df['hmm_confidence'] = 0.5
            return df
        ret = df['c'].pct_change().fillna(0).values
        atr_r = df['atr_rank'].fillna(0.5).values
        vol = df['vol_ma_ratio'].fillna(1.0).clip(0, 5).values
        bb = df['bb_width'].fillna(0.02).clip(0, 0.2).values
        X = np.column_stack([ret, atr_r, vol, bb])
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        try:
            raw_states = self.hmm5_model.predict(X)
            proba = self.hmm5_model.predict_proba(X)
            df['hmm_state'] = raw_states
            df['hmm_confidence'] = proba[np.arange(len(proba)), raw_states]
            df['hmm_label'] = [self.hmm5_labels.get(s, 'UNCERTAIN') for s in raw_states]
        except Exception as e:
            logger.error(f"[SC-HMM5] predict error: {e}")
            df['hmm_state'] = 2
            df['hmm_confidence'] = 0.5
            df['hmm_label'] = 'UNCERTAIN'
        return df

    @staticmethod
    def get_sc_features(df, i, direction='LONG'):
        """Extract 41 scalper features at bar i."""
        if i < 30:
            return None
        a = df['atr'].iloc[i]
        if pd.isna(a) or a <= 0:
            return None
        try:
            feats = [
                (df['ema21'].iloc[i] - df['ema50'].iloc[i]) / a,
                (df['c'].iloc[i] - df['ema21'].iloc[i]) / a,
                df['rsi'].iloc[i] / 100 if not pd.isna(df['rsi'].iloc[i]) else 0.5,
                (df['macd'].iloc[i] - df['macd_sig'].iloc[i]) / a if not pd.isna(df['macd'].iloc[i]) else 0,
                float(df['trend_4h'].iloc[i]),
                float(df['trend_daily'].iloc[i]),
                1.0 if df['trend_4h'].iloc[i] == df['trend_daily'].iloc[i] else 0.0,
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
                float(df['price_accel'].iloc[i]) * 100 if not pd.isna(df['price_accel'].iloc[i]) else 0,
                float(df['vol_squeeze'].iloc[i]),
            ]
            return np.array([0.0 if pd.isna(x) else float(x) for x in feats])
        except Exception as e:
            logger.error(f"[SC] Feature extraction error at bar {i}: {e}")
            return None

    @staticmethod
    def triple_barrier_label(df, i, direction, rr, sl_mult=1.0, timeout=12):
        """Triple barrier labeling for scalper training."""
        a = df['atr'].iloc[i]
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
        # Timeout soft label
        exit_c = df['c'].iloc[min(end - 1, len(df) - 1)]
        if direction == 'LONG':
            pnl_r = (exit_c - c) / sl_dist
        else:
            pnl_r = (c - exit_c) / sl_dist
        pnl_r = max(-1.0, min(float(rr), pnl_r))
        timeout_label = 0.30 + 0.20 * (pnl_r + 1.0) / (float(rr) + 1.0)
        return round(timeout_label, 4)

    @staticmethod
    def _make_stacked_model():
        """Build stacked ensemble: RF + GB + optional LGB/XGB -> LR meta."""
        base = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=7,
                                           class_weight='balanced', random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=80, max_depth=4, random_state=42)),
        ]
        if HAS_LGB:
            base.append(('lgb', lgb.LGBMClassifier(n_estimators=100, max_depth=6,
                                                     class_weight='balanced', random_state=42, verbose=-1)))
        if HAS_XGB:
            base.append(('xgb', XGBClassifier(n_estimators=80, max_depth=5,
                                                scale_pos_weight=2, random_state=42,
                                                eval_metric='logloss', verbosity=0)))
        meta = LogisticRegression(C=1.0, max_iter=500)
        return StackingClassifier(estimators=base, final_estimator=meta,
                                   cv=3, passthrough=False, n_jobs=-1)

    def _train_one_model(self, df, state_mask_col, label_rr, model_name, min_samples=80):
        """Train a single scalper model for a given state subset."""
        if state_mask_col and state_mask_col in df.columns:
            subset = df[df[state_mask_col] == 1].reset_index(drop=True)
            if len(subset) < min_samples:
                subset = df.reset_index(drop=True)
        else:
            subset = df.reset_index(drop=True)

        X_list, y_list = [], []
        for i in range(50, len(subset) - 15):
            direction = 'LONG' if subset['ema21'].iloc[i] > subset['ema50'].iloc[i] else 'SHORT'
            feats = self.get_sc_features(subset, i, direction)
            if feats is None:
                continue

            # Per-model pre-filter
            if model_name == 'FakeBreak':
                bb_p = subset['bb_pct'].iloc[i] if not pd.isna(subset['bb_pct'].iloc[i]) else 0.5
                if not (bb_p < 0.15 or bb_p > 0.85):
                    continue
            elif model_name == 'MomBurst':
                if subset['vol_spike'].iloc[i] == 0:
                    continue
            elif model_name == 'DivHunter':
                has_div = (subset['bull_div'].iloc[i] == 1) or (subset['bear_div'].iloc[i] == 1)
                if not has_div:
                    continue

            tb_timeout = min(int(label_rr * 12), 48)
            label = self.triple_barrier_label(subset, i, direction, label_rr, timeout=tb_timeout)
            X_list.append(feats)
            y_list.append(label)

        if len(X_list) < min_samples:
            logger.info(f"  [{model_name}] Not enough samples ({len(X_list)}) - skipped")
            return None, None

        X = np.array(X_list)
        y_hard = (np.array(y_list) >= 0.5).astype(int)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pos_rate = y_hard.mean()
        logger.info(f"  [{model_name}] {len(X)} samples | pos rate {pos_rate:.1%}")

        if len(set(y_hard)) < 2:
            logger.info(f"  [{model_name}] Only one class - skipped")
            return None, None

        # Time-based validation split
        split = int(len(Xs) * 0.8)
        if split > 20 and (len(Xs) - split) > 10:
            Xtr, Xval = Xs[:split], Xs[split:]
            ytr, yval = y_hard[:split], y_hard[split:]
            if len(set(ytr)) < 2:
                Xtr, ytr = Xs, y_hard
            clf = self._make_stacked_model()
            try:
                clf.fit(Xtr, ytr)
                if len(set(yval)) > 1:
                    auc = roc_auc_score(yval, clf.predict_proba(Xval)[:, 1])
                    logger.info(f"  [{model_name}] Val AUC: {auc:.3f}")
                clf.fit(Xs, y_hard)
            except Exception as e:
                logger.warning(f"  [{model_name}] stacking failed ({e}), using RF only")
                clf = RandomForestClassifier(n_estimators=150, max_depth=7,
                                              class_weight='balanced', random_state=42, n_jobs=-1)
                clf.fit(Xs, y_hard)
        else:
            clf = RandomForestClassifier(n_estimators=150, max_depth=7,
                                          class_weight='balanced', random_state=42, n_jobs=-1)
            clf.fit(Xs, y_hard)

        return clf, scaler

    async def train(self, v3b_model: V3BModel):
        """Train all 5 scalper models. Uses V3B's indicator data + scalper extras."""
        logger.info("[SCALPER V6] Training scalper models...")
        try:
            # Load training data (same source as V3B)
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2020_2025_spot.csv')
            if os.path.exists(base_csv):
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2020_2025_spot.csv"
                df_base = pd.read_csv(csv_url, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']

            new_csv = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')
            if os.path.exists(new_csv):
                df_new = pd.read_csv(new_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
            else:
                df = df_base

            # V3B indicators first (reuse existing method)
            df = v3b_model.calculate_indicators(df)
            # Scalper extras on top
            df = self.add_scalper_indicators(df)

            # HMM 5-state
            self._train_hmm5(df)
            df = self.add_hmm5_states(df)

            df = df.iloc[250:].reset_index(drop=True)
            label_rr = self.config.SC_LABEL_RR

            logger.info(f"[SCALPER V6] Training 5 models with label_rr={label_rr}")

            # 1. TrendScalper
            df_t = df.copy()
            df_t['_use'] = df_t['hmm_label'].isin(['TREND_UP', 'TREND_DOWN']).astype(int)
            m, s = self._train_one_model(df_t, '_use', label_rr, 'TrendScalper')
            self.models['TrendScalper'] = (m, s)

            # 2. RangeScalper
            df_r = df.copy()
            df_r['_use'] = (df_r['hmm_label'] == 'RANGE_TIGHT').astype(int)
            m, s = self._train_one_model(df_r, '_use', label_rr, 'RangeScalper')
            self.models['RangeScalper'] = (m, s)

            # 3. FakeBreak
            df_f = df.copy()
            df_f['_use'] = df_f['hmm_label'].isin(['RANGE_WIDE', 'RANGE_TIGHT']).astype(int)
            m, s = self._train_one_model(df_f, '_use', label_rr, 'FakeBreak')
            self.models['FakeBreak'] = (m, s)

            # 4. MomBurst
            m, s = self._train_one_model(df, None, label_rr, 'MomBurst')
            self.models['MomBurst'] = (m, s)

            # 5. DivHunter
            m, s = self._train_one_model(df, None, label_rr, 'DivHunter')
            self.models['DivHunter'] = (m, s)

            trained = [k for k, v in self.models.items() if v[0] is not None]
            logger.info(f"[SCALPER V6] Trained models: {trained}")
            self.trained = len(trained) > 0
            return self.trained

        except Exception as e:
            logger.error(f"[SCALPER V6] Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_adaptive_th(self, atr_rank_val):
        """Adaptive threshold based on volatility regime."""
        base_th = self.config.SC_THRESHOLD
        if not self.config.SC_ADAPTIVE_TH:
            return base_th
        if pd.isna(atr_rank_val):
            return base_th
        if atr_rank_val < 0.40:
            return max(base_th - 0.05, 0.35)
        elif atr_rank_val > 0.70:
            return min(base_th + 0.05, 0.90)
        return base_th

    def get_adaptive_sc_risk(self, balance, peak):
        """Reduce SC risk during drawdown."""
        base = self.config.SC_RISK
        if peak <= 0:
            return base
        dd = (peak - balance) / peak
        if dd > 0.15:
            return base * 0.5
        elif dd > 0.05:
            return base * 0.75
        return base

    def predict_best(self, df, i, direction):
        """Get best model prediction for bar i. Returns (prob, model_name) or (0, None)."""
        if not self.trained:
            return 0.0, None

        model_set = self.config.SC_MODEL_SET
        active = ACTIVE_SC_MODELS.get(model_set, list(self.models.keys()))

        # Regime filtering
        hmm_lbl = str(df['hmm_label'].iloc[i]) if 'hmm_label' in df.columns else 'UNCERTAIN'
        regime_models = REGIME_TO_MODELS.get(hmm_lbl, [])
        candidates = [m for m in active if m in regime_models or model_set == 'mom_burst_only']
        if not candidates:
            candidates = active

        feats = self.get_sc_features(df, i, direction)
        if feats is None:
            return 0.0, None

        best_prob = 0.0
        best_model = None
        for mname in candidates:
            if mname not in self.models:
                continue
            clf, scl = self.models[mname]
            if clf is None or scl is None:
                continue
            try:
                fs = scl.transform(feats.reshape(1, -1))
                prob = clf.predict_proba(fs)[0][1]
                if prob > best_prob:
                    best_prob = prob
                    best_model = mname
            except Exception:
                continue

        return best_prob, best_model


# ==================== V3B TRADING ENGINE ====================
class V3BTradingEngine:
    def __init__(self, config: Config):
        self.config = config
        self.telegram = TelegramBot(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        self.news_filter = NewsFilter()
        self.price_fetcher = PriceFetcher()
        self.candle_collector = CandleCollector(self.price_fetcher)
        self.model = V3BModel(config)
        self.asterdex = AsterDexClient(
            config.ASTERDEX_API_KEY or '',
            config.ASTERDEX_SECRET_KEY or '',
            config.ASTERDEX_BASE_URL
        ) if config.ASTERDEX_API_KEY else None

        self.balance = config.STARTING_BALANCE
        self.peak_balance = config.STARTING_BALANCE
        self.current_position = None
        self.last_exit_bar = -100
        self.bar_count = 0
        self.running = False

        # Scalper V6
        self.scalper = ScalperModel(config) if config.SC_ENABLED else None
        self.sc_stats = {'trades': 0, 'wins': 0, 'trailing_locks': 0}

    async def initialize(self):
        logger.info("Initializing TradeVersum V3B Live...")

        ok = await self.model.train()
        await self.news_filter.update_calendar()

        # Setup AsterDex
        if self.asterdex and not self.config.PAPER_TRADING:
            try:
                live_balance = await self.asterdex.get_account_balance()
                if live_balance > 0:
                    self.balance = live_balance
                    self.peak_balance = live_balance
                    logger.info(f"AsterDex USDT Balance: ${live_balance:.2f}")
                await self.asterdex.set_leverage(self.config.ASTERDEX_SYMBOL, self.config.LEVERAGE)
                await self.asterdex.set_margin_type(self.config.ASTERDEX_SYMBOL, "ISOLATED")
                logger.info("AsterDex configured: leverage set")
            except Exception as e:
                logger.error(f"AsterDex init error: {e}")

        # Train Scalper V6 models
        sc_ok = False
        if self.scalper:
            sc_ok = await self.scalper.train(self.model)
            logger.info(f"Scalper V6: {'✅ trained' if sc_ok else '❌ failed'}")

        mode = "PAPER" if self.config.PAPER_TRADING else "LIVE (AsterDex)"
        sc_status = ""
        if self.scalper:
            sc_status = (
                f"\n⚡ *Scalper V6*: {'✅' if sc_ok else '❌'}\n"
                f"SC Config: TH={self.config.SC_THRESHOLD}, RR=1:{self.config.SC_RR:.0f}, "
                f"Risk={self.config.SC_RISK:.0%}\n"
                f"Trailing: lock +{self.config.SC_TRAILING_LOCK:.0f}R at +{self.config.SC_TRAILING_TRIGGER:.0f}R"
            )
        await self.telegram.send_message(
            f"🥇 *TRADEVERSUM V3B + SCALPER V6 STARTED* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"Balance: ${self.balance:.2f}\n"
            f"V3B: ML={self.config.ML_THRESHOLD}, SL={self.config.SL_MULTIPLIER}×ATR, RR=1:3\n"
            f"Model: {'✅' if ok else '❌'} RF+GB 27 features"
            f"{sc_status}"
        )
        logger.info("V3B + Scalper V6 Engine initialized")

    def get_adaptive_risk(self) -> float:
        if self.peak_balance > self.balance:
            dd_pct = (self.peak_balance - self.balance) / self.peak_balance * 100
            if dd_pct > self.config.RISK_DD_THRESHOLD_2:
                return 0.015
            elif dd_pct > self.config.RISK_DD_THRESHOLD_1:
                return 0.0225
        return self.config.BASE_RISK

    def _calc_quantity(self, asterdex_price: float, risk_amt: float, sl_dist: float) -> float:
        """Calculate quantity for AsterDex order based on risk amount"""
        if sl_dist <= 0 or asterdex_price <= 0:
            return 0.0
        quantity = risk_amt / sl_dist
        return round(quantity, 3)

    async def check_for_signal(self) -> Optional[Dict]:
        try:
            # Market hours check (UTC)
            now = datetime.utcnow()
            weekday = now.weekday()
            hour = now.hour
            is_market_closed = (
                (weekday == 4 and hour >= 22) or
                (weekday == 5) or
                (weekday == 6 and hour < 22)
            )
            if is_market_closed:
                logger.info("Market closed - skipping signal check")
                return None

            # Cooldown check
            if self.bar_count <= self.last_exit_bar + self.config.COOLDOWN_BARS:
                logger.info(f"Cooldown active (bar {self.bar_count} <= {self.last_exit_bar + self.config.COOLDOWN_BARS})")
                return None

            # News check
            is_blackout, event = await self.news_filter.is_news_blackout()
            if is_blackout:
                logger.info(f"News blackout: {event}")
                return None

            # Load OHLC data
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2025_spot.csv')
            new_csv = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')

            if os.path.exists(base_csv):
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2025_spot.csv"
                try:
                    df_base = pd.read_csv(csv_url, parse_dates=['datetime'])
                    df_base.columns = ['ts', 'o', 'h', 'l', 'c']
                except Exception:
                    logger.error("Could not load SPOT OHLC data")
                    return None

            if os.path.exists(new_csv):
                df_new = pd.read_csv(new_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
            else:
                df = df_base

            df = df.tail(500).reset_index(drop=True)

            if len(df) < 100:
                logger.warning("Not enough OHLC data")
                return None

            df = self.model.calculate_indicators(df)

            # Apply HMM (V3B 3-state)
            if self.model.hmm_model:
                try:
                    returns = df['c'].pct_change().dropna().values.reshape(-1, 1)
                    regimes = self.model.hmm_model.predict(returns)
                    df['regime'] = pd.Series([np.nan] + list(regimes), index=df.index)
                    df['is_trending'] = (df['regime'] == self.model.trending_regime).astype(int)
                    df['is_ranging'] = (df['regime'] == self.model.ranging_regime).astype(int)
                except Exception:
                    df['is_trending'] = 0
                    df['is_ranging'] = 0

            # Add scalper indicators + HMM5 states (needed for SC signal check)
            if self.scalper and self.scalper.trained:
                df = ScalperModel.add_scalper_indicators(df)
                df = self.scalper.add_hmm5_states(df)

            i = len(df) - 1
            features, direction = self.model.get_27_features(df, i)
            if features is None:
                logger.info("Feature extraction failed for latest candle")
                return None

            ml_prob = self.model.predict(features)
            atr = df['atr'].iloc[i]

            # Verbose log for Render
            logger.info(
                f"[SIGNAL CHECK] Bar={self.bar_count} | Dir={direction} | "
                f"ML={ml_prob:.4f} ({ml_prob:.1%}) | Threshold={self.config.ML_THRESHOLD} | "
                f"ATR={atr:.2f} | EMA21={df['ema21'].iloc[i]:.2f} | EMA50={df['ema50'].iloc[i]:.2f} | "
                f"RSI={df['rsi'].iloc[i]:.1f} | "
                f"{'>>> V3B SIGNAL!' if ml_prob >= self.config.ML_THRESHOLD else 'no V3B signal'}"
            )

            # ---- V3B signal fires ----
            if ml_prob >= self.config.ML_THRESHOLD:
                spot_price = await self.price_fetcher.get_spot_price()
                if spot_price is None:
                    logger.error("Could not get SPOT price")
                    return None

                asterdex_price = spot_price
                if self.asterdex:
                    ad_price = await self.asterdex.get_asterdex_price()
                    if ad_price:
                        asterdex_price = ad_price
                        spread = abs(spot_price - asterdex_price)
                        logger.info(f"[PRICES] SPOT={spot_price:.2f} | AsterDex={asterdex_price:.2f} | Spread=${spread:.2f}")
                        if spread > self.config.MAX_PRICE_SPREAD:
                            logger.warning(
                                f"Large spread ${spread:.2f} > ${self.config.MAX_PRICE_SPREAD}. "
                                f"Proceeding with AsterDex price for order placement, SPOT ATR for SL/TP."
                            )

                sl_dist = atr * self.config.SL_MULTIPLIER
                if direction == 'LONG':
                    entry = asterdex_price + self.config.SLIPPAGE
                    sl = entry - sl_dist
                    tp = entry + sl_dist * self.config.RR
                else:
                    entry = asterdex_price - self.config.SLIPPAGE
                    sl = entry + sl_dist
                    tp = entry - sl_dist * self.config.RR

                return {
                    'type': 'V3B',
                    'direction': direction,
                    'ml_confidence': ml_prob,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'atr': atr,
                    'sl_dist': sl_dist,
                    'spot_price': spot_price,
                    'asterdex_price': asterdex_price,
                }

            # ---- SC signal check (only when V3B does NOT fire) ----
            if self.scalper and self.scalper.trained:
                sc_direction = 'LONG' if df['ema21'].iloc[i] > df['ema50'].iloc[i] else 'SHORT'
                atr_rank_val = df['atr_rank'].iloc[i] if 'atr_rank' in df.columns else 0.5
                eff_th = self.scalper.get_adaptive_th(atr_rank_val)

                sc_prob, sc_model_name = self.scalper.predict_best(df, i, sc_direction)

                logger.info(
                    f"[SC CHECK] Dir={sc_direction} | Best={sc_model_name} | "
                    f"Prob={sc_prob:.4f} ({sc_prob:.1%}) | Threshold={eff_th:.3f} | "
                    f"HMM={df['hmm_label'].iloc[i] if 'hmm_label' in df.columns else '?'} | "
                    f"{'>>> SC SIGNAL!' if sc_prob >= eff_th and sc_model_name else 'no SC signal'}"
                )

                if sc_prob >= eff_th and sc_model_name is not None:
                    spot_price = await self.price_fetcher.get_spot_price()
                    if spot_price is None:
                        logger.error("Could not get SPOT price for SC trade")
                        return None

                    asterdex_price = spot_price
                    if self.asterdex:
                        ad_price = await self.asterdex.get_asterdex_price()
                        if ad_price:
                            asterdex_price = ad_price

                    sc_sl_dist = atr * self.config.SC_SL_MULT
                    if sc_direction == 'LONG':
                        entry = asterdex_price + self.config.SLIPPAGE
                        sl = entry - sc_sl_dist
                        tp = entry + sc_sl_dist * self.config.SC_RR
                    else:
                        entry = asterdex_price - self.config.SLIPPAGE
                        sl = entry + sc_sl_dist
                        tp = entry - sc_sl_dist * self.config.SC_RR

                    return {
                        'type': 'SC',
                        'direction': sc_direction,
                        'ml_confidence': sc_prob,
                        'model_name': sc_model_name,
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'atr': atr,
                        'sl_dist': sc_sl_dist,
                        'spot_price': spot_price,
                        'asterdex_price': asterdex_price,
                    }

            return None

        except Exception as e:
            logger.error(f"Signal check error: {e}")
            return None

    async def open_position(self, signal: Dict):
        trade_type = signal.get('type', 'V3B')

        # Different risk for V3B vs SC
        if trade_type == 'SC':
            risk_pct = self.scalper.get_adaptive_sc_risk(self.balance, self.peak_balance)
        else:
            risk_pct = self.get_adaptive_risk()
        risk_amt = self.balance * risk_pct

        logger.info(
            f"[OPEN {trade_type}] {signal['direction']} | Entry={signal['entry']:.2f} | "
            f"SL={signal['sl']:.2f} | TP={signal['tp']:.2f} | "
            f"Risk={risk_pct:.1%} | RiskAmt=${risk_amt:.2f}"
            f"{' | Model=' + signal.get('model_name', '') if trade_type == 'SC' else ''}"
        )

        # Execute on AsterDex
        if self.asterdex and not self.config.PAPER_TRADING:
            try:
                quantity = self._calc_quantity(signal['asterdex_price'], risk_amt, signal['sl_dist'])
                if quantity > 0:
                    success = await self.asterdex.place_market_order_with_sl_tp(
                        symbol=self.config.ASTERDEX_SYMBOL,
                        direction=signal['direction'],
                        quantity=quantity,
                        sl_price=signal['sl'],
                        tp_price=signal['tp']
                    )
                    if not success:
                        logger.error("AsterDex order placement failed")
                        return
                    logger.info(f"AsterDex order placed: {signal['direction']} {quantity} {self.config.ASTERDEX_SYMBOL}")
                else:
                    logger.error("Quantity calculation returned 0, skipping order")
                    return
            except Exception as e:
                logger.error(f"AsterDex order error: {e}")
                return

        self.current_position = {
            'type': trade_type,
            'direction': signal['direction'],
            'entry': signal['entry'],
            'sl': signal['sl'],
            'tp': signal['tp'],
            'risk_amt': risk_amt,
            'sl_dist': signal['sl_dist'],
            'entry_time': datetime.now(),
            'entry_bar': self.bar_count,
            'ml_confidence': signal['ml_confidence'],
            'spot_price': signal['spot_price'],
            'asterdex_price': signal['asterdex_price'],
            # SC trailing stop state
            'be_done': False,
            'original_sl': signal['sl'],
            'model_name': signal.get('model_name', ''),
        }

        # Send appropriate telegram notification
        if trade_type == 'SC':
            self.sc_stats['trades'] += 1
            await self.telegram.send_sc_signal(
                direction=signal['direction'],
                model_name=signal.get('model_name', ''),
                prob=signal['ml_confidence'],
                entry=signal['entry'],
                sl=signal['sl'],
                tp=signal['tp'],
                risk_pct=risk_pct,
                atr=signal['atr'],
                rr=self.config.SC_RR,
            )
        else:
            await self.telegram.send_signal(
                direction=signal['direction'],
                ml_confidence=signal['ml_confidence'],
                entry=signal['entry'],
                sl=signal['sl'],
                tp=signal['tp'],
                risk_pct=risk_pct,
                atr=signal['atr'],
                spot_price=signal['spot_price'],
                asterdex_price=signal['asterdex_price'],
            )

    async def monitor_position(self):
        if not self.current_position:
            return

        spot_price = await self.price_fetcher.get_spot_price()
        current_price = spot_price
        if self.asterdex:
            ad_price = await self.asterdex.get_asterdex_price()
            if ad_price:
                current_price = ad_price

        if current_price is None:
            return

        pos = self.current_position
        is_long = pos['direction'] == 'LONG'
        trade_type = pos.get('type', 'V3B')

        # ---- SC Trailing Stop: lock +1R at +2R ----
        if trade_type == 'SC' and not pos.get('be_done', False) and pos['sl_dist'] > 0:
            trigger_price_long = pos['entry'] + self.config.SC_TRAILING_TRIGGER * pos['sl_dist']
            trigger_price_short = pos['entry'] - self.config.SC_TRAILING_TRIGGER * pos['sl_dist']

            if is_long and current_price >= trigger_price_long:
                new_sl = pos['entry'] + self.config.SC_TRAILING_LOCK * pos['sl_dist']
                pos['sl'] = new_sl
                pos['be_done'] = True
                self.sc_stats['trailing_locks'] += 1
                logger.info(f"[SC TRAILING] LONG +2R reached! SL moved to +1R: ${new_sl:.2f}")

                # Update SL on AsterDex
                if self.asterdex and not self.config.PAPER_TRADING:
                    try:
                        await self.asterdex.cancel_all_orders(self.config.ASTERDEX_SYMBOL)
                        close_side = "SELL"
                        await self.asterdex._request("POST", "/fapi/v1/order", {
                            "symbol": self.config.ASTERDEX_SYMBOL,
                            "side": close_side,
                            "type": "STOP_MARKET",
                            "stopPrice": f"{new_sl:.2f}",
                            "closePosition": "true"
                        }, signed=True)
                        # Re-place TP
                        await self.asterdex._request("POST", "/fapi/v1/order", {
                            "symbol": self.config.ASTERDEX_SYMBOL,
                            "side": close_side,
                            "type": "TAKE_PROFIT_MARKET",
                            "stopPrice": f"{pos['tp']:.2f}",
                            "closePosition": "true"
                        }, signed=True)
                        logger.info("[SC TRAILING] AsterDex SL/TP orders updated")
                    except Exception as e:
                        logger.error(f"[SC TRAILING] AsterDex update error: {e}")

                await self.telegram.send_sc_trailing_lock(pos['direction'], pos['entry'], new_sl)

            elif not is_long and current_price <= trigger_price_short:
                new_sl = pos['entry'] - self.config.SC_TRAILING_LOCK * pos['sl_dist']
                pos['sl'] = new_sl
                pos['be_done'] = True
                self.sc_stats['trailing_locks'] += 1
                logger.info(f"[SC TRAILING] SHORT +2R reached! SL moved to +1R: ${new_sl:.2f}")

                if self.asterdex and not self.config.PAPER_TRADING:
                    try:
                        await self.asterdex.cancel_all_orders(self.config.ASTERDEX_SYMBOL)
                        close_side = "BUY"
                        await self.asterdex._request("POST", "/fapi/v1/order", {
                            "symbol": self.config.ASTERDEX_SYMBOL,
                            "side": close_side,
                            "type": "STOP_MARKET",
                            "stopPrice": f"{new_sl:.2f}",
                            "closePosition": "true"
                        }, signed=True)
                        await self.asterdex._request("POST", "/fapi/v1/order", {
                            "symbol": self.config.ASTERDEX_SYMBOL,
                            "side": close_side,
                            "type": "TAKE_PROFIT_MARKET",
                            "stopPrice": f"{pos['tp']:.2f}",
                            "closePosition": "true"
                        }, signed=True)
                        logger.info("[SC TRAILING] AsterDex SL/TP orders updated")
                    except Exception as e:
                        logger.error(f"[SC TRAILING] AsterDex update error: {e}")

                await self.telegram.send_sc_trailing_lock(pos['direction'], pos['entry'], new_sl)

        # ---- Check exit conditions ----
        sl_hit = (is_long and current_price <= pos['sl']) or (not is_long and current_price >= pos['sl'])
        tp_hit = (is_long and current_price >= pos['tp']) or (not is_long and current_price <= pos['tp'])
        bars_held = self.bar_count - pos['entry_bar']

        # Different max_hold for V3B vs SC
        if trade_type == 'SC':
            max_hold = self.config.SC_MAX_HOLD
        else:
            max_hold = self.config.MAX_HOLD_BARS
        timeout = bars_held >= max_hold

        if tp_hit or sl_hit or timeout:
            if self.asterdex and not self.config.PAPER_TRADING:
                try:
                    await self.asterdex.cancel_all_orders(self.config.ASTERDEX_SYMBOL)
                    if timeout:
                        await self.asterdex.close_position(self.config.ASTERDEX_SYMBOL)
                except Exception as e:
                    logger.error(f"AsterDex close error: {e}")

            if tp_hit:
                # PnL based on actual RR for the trade type
                if trade_type == 'SC':
                    pnl = pos['risk_amt'] * self.config.SC_RR
                else:
                    pnl = pos['risk_amt'] * self.config.RR
                self.balance += pnl
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                await self.telegram.send_tp_hit(pos['entry'], pos['tp'], pnl, self.balance)
                logger.info(f"[{trade_type}] TP HIT! +${pnl:.2f} | Balance=${self.balance:.2f}")
                if trade_type == 'SC':
                    self.sc_stats['wins'] += 1
            elif sl_hit:
                # If SC trade with BE done, the SL is at +1R (profit!)
                if trade_type == 'SC' and pos.get('be_done', False):
                    pnl = pos['risk_amt'] * self.config.SC_TRAILING_LOCK  # +1R
                    self.balance += pnl
                    if self.balance > self.peak_balance:
                        self.peak_balance = self.balance
                    await self.telegram.send_sc_be_exit(pos['entry'], pos['sl'], pnl, self.balance)
                    logger.info(f"[SC] TRAILING STOP HIT! +${pnl:.2f} (locked +1R) | Balance=${self.balance:.2f}")
                    self.sc_stats['wins'] += 1
                else:
                    pnl = -pos['risk_amt']
                    self.balance += pnl
                    await self.telegram.send_sl_hit(pos['entry'], pos['sl'], pnl, self.balance)
                    logger.info(f"[{trade_type}] SL HIT! ${pnl:.2f} | Balance=${self.balance:.2f}")
            elif timeout:
                if is_long:
                    pnl = (current_price - pos['entry']) / pos['sl_dist'] * pos['risk_amt']
                else:
                    pnl = (pos['entry'] - current_price) / pos['sl_dist'] * pos['risk_amt']
                self.balance += pnl
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                logger.info(f"[{trade_type}] TIMEOUT exit! PnL=${pnl:.2f} | Balance=${self.balance:.2f}")
                await self.telegram.send_message(
                    f"⏰ *{trade_type} TIMEOUT* - Position closed\nPnL: ${pnl:.2f}\nBalance: ${self.balance:.2f}"
                )
                if trade_type == 'SC' and pnl > 0:
                    self.sc_stats['wins'] += 1

            self.current_position = None
            self.last_exit_bar = self.bar_count

    async def check_retrain(self):
        if self.model.last_train is None:
            return
        days_since_train = (datetime.now() - self.model.last_train).days
        if days_since_train >= 7:
            logger.info("Weekly retrain triggered...")
            await self.model.train()
            if self.scalper:
                await self.scalper.train(self.model)
                logger.info("Scalper V6 retrained")

    async def run(self):
        self.running = True
        await self.initialize()

        while self.running:
            try:
                current_hour = datetime.now().hour
                if not hasattr(self, '_last_hour'):
                    self._last_hour = current_hour
                if current_hour != self._last_hour:
                    self.bar_count += 1
                    self._last_hour = current_hour
                    logger.info(f"New bar #{self.bar_count} | Balance=${self.balance:.2f} | "
                                f"DD={max(0, (self.peak_balance - self.balance) / self.peak_balance * 100):.1f}%")

                await self.candle_collector.update()
                await self.monitor_position()

                if not self.current_position:
                    current_minute = datetime.now().minute
                    if current_minute <= self.config.SIGNAL_WINDOW_MINUTES:
                        signal = await self.check_for_signal()
                        if signal:
                            await self.open_position(signal)

                await self.check_retrain()
                await asyncio.sleep(self.config.CHECK_INTERVAL)

            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)

        sc_info = ""
        if self.scalper:
            sc_wr = (self.sc_stats['wins'] / self.sc_stats['trades'] * 100) if self.sc_stats['trades'] > 0 else 0
            sc_info = (
                f"\n⚡ SC Stats: {self.sc_stats['trades']}T | "
                f"{sc_wr:.0f}% WR | {self.sc_stats['trailing_locks']} trailing locks"
            )
        await self.telegram.send_message(
            f"🛑 *TradeVersum V3B + Scalper V6 Stopped*\n"
            f"Final Balance: ${self.balance:.2f}{sc_info}"
        )
        logger.info("TradeVersum V3B + Scalper V6 shutdown complete")


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
    ║  🥇 TRADEVERSUM V3B + SCALPER V6 - GOLD TRADING BOT 🥇   ║
    ║                                                           ║
    ║  V3B: 27 Features | ML=0.455 | RR=1:3 | RF+GB            ║
    ║  SC6: 41 Features | TH=0.42  | RR=1:5 | Stacked Ensemble ║
    ║       5 Models | HMM 5-State | Trailing Stop (+1R@+2R)   ║
    ║  Exchange: AsterDex (XAUUSDT Futures)                     ║
    ║                                                           ║
    ║        ⚠️  HIGH RISK - USE AT YOUR OWN RISK  ⚠️          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    asyncio.run(main())
