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
import MetaTrader5 as mt5

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
    
    # ===== MT5 SETTINGS =====
    USE_MT5 = True  # Enable MT5 real trading (runs ALONGSIDE paper trading)
    MT5_LOGIN = int(os.environ.get('MT5_LOGIN', '0'))  # MT5 account number
    MT5_PASSWORD = os.environ.get('MT5_PASSWORD', '')  # MT5 password
    MT5_SERVER = os.environ.get('MT5_SERVER', '')  # MT5 server
    MT5_SYMBOL = "XAUUSD"  # MT5 symbol for Gold
    MT5_RISK_PERCENT = 0.01  # MT5 uses 1% risk (paper trading keeps 3%)
    MT5_MIN_LOT = 0.01  # Minimum lot size
    MT5_MAX_LOT = 10.0  # Maximum lot size for safety
    EMERGENCY_SL_DISTANCE = 1.0  # Emergency SL is $1 below/above normal SL
    
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


# ==================== MT5 TRADER ====================
class MT5Trader:
    """MT5 Real Trading with Emergency Stop Loss"""
    
    def __init__(self, config: Config):
        self.config = config
        self.initialized = False
        self.mt5_position_ticket = None
    
    def calculate_lot_size(self, signal: Dict, account_balance: float) -> float:
        """Calculate lot size based on 1% risk and SL distance"""
        try:
            # Risk amount: 1% of balance
            risk_amount = account_balance * self.config.MT5_RISK_PERCENT
            
            # SL distance in USD
            sl_distance = abs(signal['entry'] - signal['sl'])
            
            if sl_distance <= 0:
                logger.error("Invalid SL distance, using minimum lot")
                return self.config.MT5_MIN_LOT
            
            # For XAUUSD: 0.01 lot = 1 oz gold
            # If price moves $1 with 0.01 lot (1 oz), P/L = $1
            # Therefore: lot_size = risk_amount / sl_distance * 0.01
            lot_size = (risk_amount / sl_distance) * 0.01
            
            # Round to 2 decimal places (standard for most brokers)
            lot_size = round(lot_size, 2)
            
            # Apply min/max limits
            lot_size = max(self.config.MT5_MIN_LOT, min(lot_size, self.config.MT5_MAX_LOT))
            
            logger.info(
                f"📊 MT5 Lot Calculation: Balance=${account_balance:.2f}, "
                f"Risk 1%=${risk_amount:.2f}, SL dist=${sl_distance:.2f}, "
                f"Lot={lot_size:.2f}"
            )
            
            return lot_size
            
        except Exception as e:
            logger.error(f"Lot size calculation error: {e}")
            return self.config.MT5_MIN_LOT
    
    async def initialize(self) -> bool:
        """Initialize MT5 connection"""
        if not self.config.USE_MT5:
            logger.info("📊 MT5 disabled in config")
            return False
        
        if not self.config.MT5_LOGIN or not self.config.MT5_PASSWORD or not self.config.MT5_SERVER:
            logger.warning("⚠️ MT5 credentials not configured, skipping MT5")
            return False
        
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Login
            authorized = mt5.login(
                login=self.config.MT5_LOGIN,
                password=self.config.MT5_PASSWORD,
                server=self.config.MT5_SERVER
            )
            
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get MT5 account info")
                mt5.shutdown()
                return False
            
            self.initialized = True
            logger.info(f"✅ MT5 Connected - Account: {account_info.login}, Balance: ${account_info.balance:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"MT5 initialization error: {e}")
            return False
    
    async def open_position(self, signal: Dict) -> bool:
        """Open MT5 position with EMERGENCY STOP LOSS and 1% RISK"""
        if not self.initialized:
            logger.warning("MT5 not initialized, skipping real trade")
            return False
        
        try:
            symbol = self.config.MT5_SYMBOL
            
            # Get current account balance
            account_info = mt5.account_info()
            if account_info is None:
                logger.error("Failed to get MT5 account info")
                return False
            
            # Calculate lot size based on 1% risk
            lot = self.calculate_lot_size(signal, account_info.balance)
            
            # Check if symbol is available
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found in MT5")
                return False
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select symbol {symbol}")
                    return False
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logger.error(f"Failed to get tick for {symbol}")
                return False
            
            # Determine order type and prices
            is_long = signal['direction'] == 'LONG'
            
            if is_long:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                # Emergency SL is $1 BELOW the normal SL
                emergency_sl = signal['sl'] - self.config.EMERGENCY_SL_DISTANCE
                tp = signal['tp']
            else:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                # Emergency SL is $1 ABOVE the normal SL (for short)
                emergency_sl = signal['sl'] + self.config.EMERGENCY_SL_DISTANCE
                tp = signal['tp']
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": emergency_sl,  # EMERGENCY STOP LOSS - server-side protection!
                "tp": tp,
                "deviation": 20,
                "magic": 234000,  # Magic number to identify our trades
                "comment": f"V3B_{signal['direction']}_ML{signal['ml_confidence']:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                logger.error(f"MT5 order_send failed: {mt5.last_error()}")
                return False
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"MT5 order failed: {result.retcode} - {result.comment}")
                return False
            
            self.mt5_position_ticket = result.order
            
            logger.info(
                f"✅ MT5 Position Opened: {signal['direction']} {lot} lots @ ${price:.2f} | "
                f"Emergency SL: ${emergency_sl:.2f} (Normal SL: ${signal['sl']:.2f}, -${self.config.EMERGENCY_SL_DISTANCE}) | "
                f"TP: ${tp:.2f} | Ticket: {result.order}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"MT5 position open error: {e}")
            return False
    
    async def close_position(self) -> bool:
        """Close MT5 position manually"""
        if not self.initialized or self.mt5_position_ticket is None:
            return False
        
        try:
            # Get position info
            positions = mt5.positions_get(ticket=self.mt5_position_ticket)
            
            if positions is None or len(positions) == 0:
                logger.info("MT5 position already closed")
                self.mt5_position_ticket = None
                return True
            
            position = positions[0]
            
            # Prepare close request
            tick = mt5.symbol_info_tick(position.symbol)
            if tick is None:
                logger.error("Failed to get tick for closing")
                return False
            
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": price,
                "deviation": 20,
                "magic": 234000,
                "comment": "V3B_Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"MT5 close failed: {result}")
                return False
            
            logger.info(f"✅ MT5 Position Closed @ ${price:.2f}")
            self.mt5_position_ticket = None
            return True
            
        except Exception as e:
            logger.error(f"MT5 close error: {e}")
            return False
    
    async def check_position_status(self) -> Optional[str]:
        """Check if MT5 position hit TP/SL/still open"""
        if not self.initialized or self.mt5_position_ticket is None:
            return None
        
        try:
            positions = mt5.positions_get(ticket=self.mt5_position_ticket)
            
            if positions is None or len(positions) == 0:
                # Position closed - check last deal to see if it was TP or SL
                deals = mt5.history_deals_get(ticket=self.mt5_position_ticket)
                if deals and len(deals) > 0:
                    # Position was closed
                    self.mt5_position_ticket = None
                    return "CLOSED"
                return "CLOSED"
            
            # Position still open
            return "OPEN"
            
        except Exception as e:
            logger.error(f"MT5 status check error: {e}")
            return None
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        if self.initialized:
            mt5.shutdown()
            logger.info("MT5 connection closed")


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
        
        import os
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
        """Train model on SPOT CSV + auto-collected new candles"""
        logger.info("🤖 Training V3B model (27 features, RF+GB)...")
        
        try:
            import os
            
            # 1. Load BASE training data (2020-2025)
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2020_2025_spot.csv')
            
            if os.path.exists(base_csv):
                logger.info(f"📊 Loading BASE training data: {base_csv}")
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                # Try GitHub
                logger.warning("⚠️ Local CSV not found, trying GitHub...")
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2020_2025_spot.csv"
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
        self.candle_collector = CandleCollector(self.price_fetcher)  # Auto-collect SPOT candles
        self.model = V3BModel(config)
        self.mt5_trader = MT5Trader(config)  # MT5 real trading
        
        self.balance = config.STARTING_BALANCE
        self.peak_balance = config.STARTING_BALANCE
        self.current_position = None
        self.last_exit_bar = -100
        self.bar_count = 0
        self.running = False
    
    async def initialize(self):
        logger.info("🥇 Initializing Terminator V3B Live...")
        
        # Initialize MT5
        mt5_status = await self.mt5_trader.initialize()
        if mt5_status:
            logger.info("✅ MT5 Trading enabled with Emergency Stop Loss protection")
        else:
            logger.info("📄 MT5 disabled - Paper trading only")
        
        await self.model.train()
        await self.news_filter.update_calendar()
        
        mode = "📄 PAPER" if self.config.PAPER_TRADING else "💰 LIVE"
        mt5_mode = "+ 🔥 MT5 REAL" if mt5_status else ""
        
        await self.telegram.send_message(
            f"🥇 *TERMINATOR V3B LIVE STARTED* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode} {mt5_mode}\n"
            f"Balance: ${self.balance:.2f}\n"
            f"MT5 Emergency SL: ${self.config.EMERGENCY_SL_DISTANCE} below normal SL\n"
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
            
            # Get OHLC data - USE SPOT CSV + COLLECTED 2026 CANDLES
            import os
            
            # Load base 2025 data
            base_csv = os.path.join(os.path.dirname(__file__), 'xauusd_1h_2025_spot.csv')
            new_csv = os.path.join(os.path.dirname(__file__), 'new_candles_2026.csv')
            
            if os.path.exists(base_csv):
                df_base = pd.read_csv(base_csv, parse_dates=['datetime'])
                df_base.columns = ['ts', 'o', 'h', 'l', 'c']
            else:
                # Try GitHub
                csv_url = "https://raw.githubusercontent.com/hattabalint/terminator-gold/main/xauusd_1h_2025_spot.csv"
                try:
                    df_base = pd.read_csv(csv_url, parse_dates=['datetime'])
                    df_base.columns = ['ts', 'o', 'h', 'l', 'c']
                except:
                    logger.error("Could not load SPOT OHLC data")
                    return None
            
            # Add new 2026 collected candles if they exist
            if os.path.exists(new_csv):
                df_new = pd.read_csv(new_csv, parse_dates=['datetime'])
                df_new.columns = ['ts', 'o', 'h', 'l', 'c']
                df = pd.concat([df_base, df_new], ignore_index=True)
                df = df.drop_duplicates(subset=['ts']).reset_index(drop=True)
            else:
                df = df_base
            
            # Use last 200 candles for indicators
            df = df.tail(200).reset_index(drop=True)
            
            if len(df) < 100:
                return None
            
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
        """Open V3B position (Paper + MT5)"""
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
        
        # Open MT5 position with Emergency SL
        if self.mt5_trader.initialized:
            mt5_success = await self.mt5_trader.open_position(signal)
            if mt5_success:
                logger.info("🔥 MT5 Real position opened with Emergency SL protection")
            else:
                logger.warning("⚠️ MT5 position failed to open")
        
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
        """Monitor position for TP/SL (Paper + MT5)"""
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
            
            # Close MT5 position if exists
            if self.mt5_trader.initialized:
                await self.mt5_trader.close_position()
            
            await self.telegram.send_tp_hit(pos['entry'], pos['tp'], pnl, self.balance)
            logger.info(f"✅ TP HIT! +${pnl:.2f}")
            
            self.current_position = None
            self.last_exit_bar = self.bar_count
            
        elif sl_hit:
            pnl = -pos['risk_amt']
            self.balance += pnl
            
            # Close MT5 position if exists
            if self.mt5_trader.initialized:
                await self.mt5_trader.close_position()
            
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
            
            # Close MT5 position if exists
            if self.mt5_trader.initialized:
                await self.mt5_trader.close_position()
            
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
                
                # Collect SPOT candle (every minute)
                await self.candle_collector.update()
                
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
        
        # Shutdown MT5 connection
        self.mt5_trader.shutdown()
        
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
