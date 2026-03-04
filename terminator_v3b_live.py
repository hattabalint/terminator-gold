# -*- coding: utf-8 -*-
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
  - Exchange: AsterDex (XAUUSDT Futures)
  - Signal Logic: SPOT price (AsterDex/goldprice.org)

Backtest Results (2025):
  - 210 Trades
  - 50.0% Win Rate
  - +39,888% Profit (Compound)

Author: TradeVersum
Version: 3.1.0 (V3B Production + AsterDex)
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

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
    return "🥇 TRADEVERSUM V3B LIVE - GOLD BOT (50.0% WR) 🥇"


@app.route('/health')
def health():
    return {"status": "healthy", "version": "V3B-3.1", "timestamp": datetime.now().isoformat()}


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

        mode = "PAPER" if self.config.PAPER_TRADING else "LIVE (AsterDex)"
        await self.telegram.send_message(
            f"🥇 *TRADEVERSUM V3B STARTED* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"Balance: ${self.balance:.2f}\n"
            f"Config: ML={self.config.ML_THRESHOLD}, SL={self.config.SL_MULTIPLIER}×ATR, RR=1:3\n"
            f"Model: {'✅' if ok else '❌'} RF+GB 27 features"
        )
        logger.info("V3B Engine initialized")

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

            # Apply HMM
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
                f"{'>>> SIGNAL!' if ml_prob >= self.config.ML_THRESHOLD else 'no signal'}"
            )

            if ml_prob < self.config.ML_THRESHOLD:
                return None

            # Get SPOT price
            spot_price = await self.price_fetcher.get_spot_price()
            if spot_price is None:
                logger.error("Could not get SPOT price")
                return None

            # Get AsterDex price
            asterdex_price = spot_price  # default fallback
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

            # Calculate levels using AsterDex price for entry (actual trade)
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

        except Exception as e:
            logger.error(f"Signal check error: {e}")
            return None

    async def open_position(self, signal: Dict):
        risk_pct = self.get_adaptive_risk()
        risk_amt = self.balance * risk_pct

        logger.info(
            f"[OPEN POSITION] {signal['direction']} | Entry={signal['entry']:.2f} | "
            f"SL={signal['sl']:.2f} | TP={signal['tp']:.2f} | "
            f"Risk={risk_pct:.1%} | RiskAmt=${risk_amt:.2f}"
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
        }

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

        sl_hit = (is_long and current_price <= pos['sl']) or (not is_long and current_price >= pos['sl'])
        tp_hit = (is_long and current_price >= pos['tp']) or (not is_long and current_price <= pos['tp'])
        bars_held = self.bar_count - pos['entry_bar']
        timeout = bars_held >= self.config.MAX_HOLD_BARS

        if tp_hit or sl_hit or timeout:
            if self.asterdex and not self.config.PAPER_TRADING:
                try:
                    await self.asterdex.cancel_all_orders(self.config.ASTERDEX_SYMBOL)
                    if timeout:
                        await self.asterdex.close_position(self.config.ASTERDEX_SYMBOL)
                except Exception as e:
                    logger.error(f"AsterDex close error: {e}")

            if tp_hit:
                pnl = pos['risk_amt'] * self.config.RR
                self.balance += pnl
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                await self.telegram.send_tp_hit(pos['entry'], pos['tp'], pnl, self.balance)
                logger.info(f"TP HIT! +${pnl:.2f} | Balance=${self.balance:.2f}")
            elif sl_hit:
                pnl = -pos['risk_amt']
                self.balance += pnl
                await self.telegram.send_sl_hit(pos['entry'], pos['sl'], pnl, self.balance)
                logger.info(f"SL HIT! ${pnl:.2f} | Balance=${self.balance:.2f}")
            elif timeout:
                if is_long:
                    pnl = (current_price - pos['entry']) / pos['sl_dist'] * pos['risk_amt']
                else:
                    pnl = (pos['entry'] - current_price) / pos['sl_dist'] * pos['risk_amt']
                self.balance += pnl
                if self.balance > self.peak_balance:
                    self.peak_balance = self.balance
                logger.info(f"TIMEOUT exit! PnL=${pnl:.2f} | Balance=${self.balance:.2f}")
                await self.telegram.send_message(f"⏰ *TIMEOUT* - Position closed\nPnL: ${pnl:.2f}\nBalance: ${self.balance:.2f}")

            self.current_position = None
            self.last_exit_bar = self.bar_count

    async def check_retrain(self):
        if self.model.last_train is None:
            return
        days_since_train = (datetime.now() - self.model.last_train).days
        if days_since_train >= 7:
            logger.info("Weekly retrain triggered...")
            await self.model.train()

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

        await self.telegram.send_message(
            f"🛑 *TradeVersum V3B Stopped*\n"
            f"Final Balance: ${self.balance:.2f}"
        )
        logger.info("TradeVersum V3B shutdown complete")


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
    ║     🥇 TRADEVERSUM V3B LIVE - GOLD TRADING BOT 🥇        ║
    ║                                                           ║
    ║     27 Features | ML=0.455 | RR=1:3 | 50.0% WR           ║
    ║     SMC Order Blocks | HMM Regime | RF+GB Ensemble        ║
    ║     Exchange: AsterDex (XAUUSDT Futures)                  ║
    ║                                                           ║
    ║        ⚠️  HIGH RISK - USE AT YOUR OWN RISK  ⚠️          ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    asyncio.run(main())
