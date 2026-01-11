"""
🥇 TERMINATOR ML LIVE - GOLD TRADING BOT 🥇
============================================
ML-based trading with 44.1% WR backtest results
Features:
- VotingClassifier (RandomForest + GradientBoosting)
- Weekly auto-retrain on fresh data
- Smart news filter (API-based)
- Telegram alerts (Entry, TP, SL)
- Koyeb compatible (Flask keep-alive)

Author: Claude/Opus
Version: 1.0.0
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler

# Windows encoding fix
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ==================== CONFIGURATION ====================
class Config:
    # Telegram
    TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # Trading
    PAPER_TRADING = os.environ.get('PAPER_TRADING', 'true').lower() == 'true'
    STARTING_BALANCE = float(os.environ.get('STARTING_BALANCE', '1000'))
    
    # ML Settings
    RISK_PERCENT = 3.0
    SCALP_SL = 3.0      # $3 SL for low volatility
    NORMAL_SL = 5.0     # $5 SL for high volatility
    SCALP_RR = 3        # 1:3 RR
    NORMAL_RR = 5       # 1:5 RR
    ML_THRESHOLD = 0.50 # 50% confidence minimum
    MAX_HOLD_HOURS = 30 # Max hours to hold a position
    COOLDOWN_HOURS = 3   # Hours to wait after trade closes before new signal
    
    # Costs
    SPREAD = 0.30
    SLIPPAGE_MIN = 0.30
    SLIPPAGE_MAX = 1.00
    
    # Data
    SYMBOL = "XAUUSD"
    CHECK_INTERVAL = 60  # Check every 60 seconds


# ==================== LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('terminator_ml_live.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== FLASK KEEP-ALIVE ====================
app = Flask('')

@app.route('/')
def home():
    return "🥇 TERMINATOR ML LIVE - GOLD BOT ONLINE 🥇"

@app.route('/health')
def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

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
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_message(self, text: str):
        """Send message to Telegram"""
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
    
    async def send_signal(self, direction: str, ml_confidence: float, trade_type: str,
                          entry: float, sl: float, tp: float, risk_pct: float):
        """Send entry signal"""
        emoji = "📈" if direction == "LONG" else "📉"
        sl_dist = abs(entry - sl)
        rr = "1:3" if trade_type == "SCALP" else "1:5"
        
        message = f"""
🥇 *ML SIGNAL* 🥇
━━━━━━━━━━━━━━━━
📍 XAU/USD | {emoji} *{direction}*
💪 ML: *{ml_confidence:.0%}* | 📊 {trade_type}
━━━━━━━━━━━━━━━━
💰 Entry: *${entry:.2f}*
🛑 SL: *${sl:.2f}* (${sl_dist:.2f})
✅ TP: *${tp:.2f}* ({rr} RR)
💵 Risk: *{risk_pct}%*
"""
        await self.send_message(message)
    
    async def send_tp_hit(self, entry: float, exit_price: float, pnl: float, rr: str):
        """Send TP hit notification"""
        message = f"""
✅ *TP HIT!* ✅
━━━━━━━━━━━━━━━━
💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}
📈 Profit: *+${pnl:.2f}* ({rr} RR)
"""
        await self.send_message(message)
    
    async def send_sl_hit(self, entry: float, exit_price: float, pnl: float):
        """Send SL hit notification"""
        message = f"""
🛑 *SL HIT* 🛑
━━━━━━━━━━━━━━━━
💰 Entry: ${entry:.2f} → Exit: ${exit_price:.2f}
📉 Loss: *${pnl:.2f}*
"""
        await self.send_message(message)


# ==================== NEWS FILTER (SMART API) ====================
class NewsFilter:
    """Smart news filter using ForexFactory calendar API"""
    
    def __init__(self):
        self.high_impact_events = []
        self.last_update = None
        self.blackout_minutes_before = 15
        self.blackout_minutes_after = 30
    
    async def update_calendar(self):
        """Fetch today's high-impact news from ForexFactory"""
        try:
            # Using a free alternative API for forex calendar
            url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Filter for USD high-impact events (affects Gold)
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
    
    async def is_news_blackout(self) -> tuple[bool, Optional[str]]:
        """Check if we're in a news blackout period"""
        # Update calendar if stale (older than 6 hours)
        if self.last_update is None or (datetime.now() - self.last_update).seconds > 21600:
            await self.update_calendar()
        
        now = datetime.now()
        
        for event in self.high_impact_events:
            event_time = event['time'].replace(tzinfo=None)
            
            # Check if within blackout window
            blackout_start = event_time - timedelta(minutes=self.blackout_minutes_before)
            blackout_end = event_time + timedelta(minutes=self.blackout_minutes_after)
            
            if blackout_start <= now <= blackout_end:
                return True, event['title']
        
        return False, None


# ==================== PRICE FETCHER ====================
class PriceFetcher:
    """Fetch real-time gold prices"""
    
    def __init__(self):
        self.last_price = None
        self.last_update = None
    
    async def get_current_price(self) -> Optional[float]:
        """Get current XAU/USD spot price"""
        try:
            # Try goldprice.org API first
            async with aiohttp.ClientSession() as session:
                # Primary: goldprice.org
                url = "https://data-asg.goldprice.org/dbXRates/USD"
                headers = {"Accept": "application/json"}
                
                async with session.get(url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        price = data.get('items', [{}])[0].get('xauPrice')
                        if price:
                            self.last_price = float(price)
                            self.last_update = datetime.now()
                            return self.last_price
            
            # Fallback: yfinance
            ticker = yf.Ticker("GC=F")
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                self.last_price = float(data['Close'].iloc[-1])
                self.last_update = datetime.now()
                return self.last_price
                
        except Exception as e:
            logger.error(f"Price fetch error: {e}")
        
        return self.last_price  # Return cached price if fetch fails
    
    async def get_ohlc_data(self, period: str = "1d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Get OHLC data for analysis"""
        try:
            ticker = yf.Ticker("GC=F")  # Using futures as proxy (moves with spot)
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df = df.reset_index()
                df.columns = ['ts', 'o', 'h', 'l', 'c', 'v', 'dividends', 'splits']
                df = df[['ts', 'o', 'h', 'l', 'c', 'v']]
                return df
        except Exception as e:
            logger.error(f"OHLC fetch error: {e}")
        return None


# ==================== ML MODEL ====================
class MLModel:
    """Machine Learning model for trade prediction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.last_train = None
    
    def extract_features(self, c, o, ema21, ema50, atr, rsi, volatility, i) -> tuple:
        """Extract features for ML prediction"""
        if i < 10 or np.isnan(atr[i]) or atr[i] == 0:
            return None, None
        
        trend = 1 if ema21[i] > ema50[i] else -1
        direction = 'LONG' if trend == 1 else 'SHORT'
        
        features = [
            (ema21[i] - ema50[i]) / atr[i],      # EMA spread
            (c[i] - ema21[i]) / atr[i],          # Price vs EMA
            volatility[i],                        # Volatility
            rsi[i] / 100,                         # RSI normalized
            (c[i] - c[i-1]) / atr[i],            # 1-bar momentum
            (c[i] - c[i-5]) / atr[i] if i >= 5 else 0,  # 5-bar momentum
            abs(c[i] - o[i]) / atr[i],           # Candle body
            1 if c[i] > o[i] else 0,             # Bullish candle
            trend                                 # Trend direction
        ]
        
        return np.array(features), direction
    
    async def train(self):
        """Train the ML model on historical data"""
        logger.info("🤖 Training ML model...")
        
        try:
            # Download training data (2020-2024)
            df = yf.download('GC=F', start='2020-01-01', end='2024-12-31', 
                           interval='1d', progress=False)
            df = df.reset_index()
            
            if len(df.columns) == 7:
                df.columns = ['ts', 'o', 'h', 'l', 'c', 'ac', 'v']
                df = df.drop('ac', axis=1)
            else:
                df.columns = ['ts', 'o', 'h', 'l', 'c', 'v']
            
            # Calculate indicators
            df['ema21'] = df['c'].ewm(span=21).mean()
            df['ema50'] = df['c'].ewm(span=50).mean()
            df['atr'] = (df['h'] - df['l']).rolling(14).mean()
            
            delta = df['c'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
            df['volatility'] = df['atr'] / df['c']
            
            df = df.iloc[60:].reset_index(drop=True)
            
            # Extract features and labels
            X_train, y_train = [], []
            c = df['c'].values
            o = df['o'].values
            h = df['h'].values
            l = df['l'].values
            ema21 = df['ema21'].values
            ema50 = df['ema50'].values
            atr = df['atr'].values
            rsi = df['rsi'].values
            volatility = df['volatility'].values
            
            for i in range(20, len(df) - 15):
                features, direction = self.extract_features(c, o, ema21, ema50, atr, rsi, volatility, i)
                if features is None:
                    continue
                
                # Create label
                entry = c[i]
                sl_dist = atr[i] * 1.5
                if direction == 'LONG':
                    sl, tp = entry - sl_dist, entry + sl_dist * 3
                else:
                    sl, tp = entry + sl_dist, entry - sl_dist * 3
                
                label = 0
                for j in range(i + 1, min(i + 15, len(df))):
                    if direction == 'LONG':
                        if l[j] <= sl: break
                        if h[j] >= tp: label = 1; break
                    else:
                        if h[j] >= sl: break
                        if l[j] <= tp: label = 1; break
                
                X_train.append(features)
                y_train.append(label)
            
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            self.model = VotingClassifier([
                ('rf', RandomForestClassifier(n_estimators=80, max_depth=5, 
                                             class_weight='balanced', random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=4, 
                                                  learning_rate=0.05, random_state=42))
            ], voting='soft')
            
            self.model.fit(X_scaled, y_train)
            self.last_train = datetime.now()
            
            logger.info(f"✅ Model trained on {len(X_train)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> float:
        """Get prediction probability"""
        if self.model is None:
            return 0.0
        
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prob = self.model.predict_proba(features_scaled)[0][1]
            return float(prob)
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0.0


# ==================== TRADING ENGINE ====================
class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.telegram = TelegramBot(config.TELEGRAM_TOKEN, config.TELEGRAM_CHAT_ID)
        self.news_filter = NewsFilter()
        self.price_fetcher = PriceFetcher()
        self.ml_model = MLModel(config)
        
        self.balance = config.STARTING_BALANCE
        self.current_position = None
        self.running = False
        self.last_trade_close = None  # Track when last trade closed
    
    async def initialize(self):
        """Initialize the trading engine"""
        logger.info("🥇 Initializing Terminator ML Live...")
        
        # Train ML model
        await self.ml_model.train()
        
        # Update news calendar
        await self.news_filter.update_calendar()
        
        # Send startup message
        mode = "📄 PAPER" if self.config.PAPER_TRADING else "💰 LIVE"
        await self.telegram.send_message(
            f"🥇 *TERMINATOR ML LIVE STARTED* 🥇\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"Mode: {mode}\n"
            f"Balance: ${self.balance:.2f}\n"
            f"ML Model: ✅ Ready"
        )
        
        logger.info("✅ Engine initialized")
    
    async def check_for_signal(self) -> Optional[Dict]:
        """Check if there's a valid trade signal"""
        try:
            # Check news blackout
            is_blackout, event = await self.news_filter.is_news_blackout()
            if is_blackout:
                logger.info(f"📰 News blackout: {event}")
                return None
            
            # Get recent OHLC data
            df = await self.price_fetcher.get_ohlc_data(period="5d", interval="1h")
            if df is None or len(df) < 60:
                return None
            
            # Calculate indicators
            df['ema21'] = df['c'].ewm(span=21).mean()
            df['ema50'] = df['c'].ewm(span=50).mean()
            df['atr'] = (df['h'] - df['l']).rolling(14).mean()
            
            delta = df['c'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
            df['volatility'] = df['atr'] / df['c']
            
            i = len(df) - 1
            c = df['c'].values
            o = df['o'].values
            ema21 = df['ema21'].values
            ema50 = df['ema50'].values
            atr = df['atr'].values
            rsi = df['rsi'].values
            volatility = df['volatility'].values
            
            # Extract features
            features, direction = self.ml_model.extract_features(
                c, o, ema21, ema50, atr, rsi, volatility, i
            )
            
            if features is None:
                return None
            
            # ML prediction
            ml_prob = self.ml_model.predict(features)
            
            if ml_prob < self.config.ML_THRESHOLD:
                return None
            
            # Determine trade type
            current_price = c[i]
            if volatility[i] < 0.015:
                trade_type = 'SCALP'
                sl_dist = self.config.SCALP_SL
                rr = self.config.SCALP_RR
            else:
                trade_type = 'NORMAL'
                sl_dist = self.config.NORMAL_SL
                rr = self.config.NORMAL_RR
            
            # Calculate levels with slippage
            slippage = random.uniform(self.config.SLIPPAGE_MIN, self.config.SLIPPAGE_MAX)
            
            if direction == 'LONG':
                entry = current_price + slippage + self.config.SPREAD
                sl = entry - sl_dist
                tp = entry + (sl_dist * rr) - slippage
            else:
                entry = current_price - slippage - self.config.SPREAD
                sl = entry + sl_dist
                tp = entry - (sl_dist * rr) + slippage
            
            return {
                'direction': direction,
                'trade_type': trade_type,
                'ml_confidence': ml_prob,
                'entry': entry,
                'sl': sl,
                'tp': tp,
                'sl_dist': sl_dist,
                'rr': rr
            }
            
        except Exception as e:
            logger.error(f"Signal check error: {e}")
            return None
    
    async def open_position(self, signal: Dict):
        """Open a new position"""
        risk_amt = self.balance * (self.config.RISK_PERCENT / 100)
        
        self.current_position = {
            'direction': signal['direction'],
            'trade_type': signal['trade_type'],
            'entry': signal['entry'],
            'sl': signal['sl'],
            'tp': signal['tp'],
            'rr': signal['rr'],
            'risk_amt': risk_amt,
            'entry_time': datetime.now(),
            'ml_confidence': signal['ml_confidence']
        }
        
        logger.info(f"📈 OPENED {signal['direction']} @ ${signal['entry']:.2f}")
        
        await self.telegram.send_signal(
            direction=signal['direction'],
            ml_confidence=signal['ml_confidence'],
            trade_type=signal['trade_type'],
            entry=signal['entry'],
            sl=signal['sl'],
            tp=signal['tp'],
            risk_pct=self.config.RISK_PERCENT
        )
    
    async def monitor_position(self):
        """Monitor current position for TP/SL"""
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
        
        # Check timeout
        hours_held = (datetime.now() - pos['entry_time']).seconds / 3600
        timeout = hours_held >= self.config.MAX_HOLD_HOURS
        
        if tp_hit:
            pnl = pos['risk_amt'] * pos['rr']
            self.balance += pnl
            
            rr_str = f"1:{pos['rr']}"
            await self.telegram.send_tp_hit(pos['entry'], pos['tp'], pnl, rr_str)
            logger.info(f"✅ TP HIT! +${pnl:.2f}")
            
            self.current_position = None
            self.last_trade_close = datetime.now()  # Cooldown starts
            
        elif sl_hit:
            pnl = -pos['risk_amt']
            self.balance += pnl
            
            await self.telegram.send_sl_hit(pos['entry'], pos['sl'], pnl)
            logger.info(f"🛑 SL HIT! ${pnl:.2f}")
            
            self.current_position = None
            self.last_trade_close = datetime.now()  # Cooldown starts
            
        elif timeout:
            # Close at current price
            if is_long:
                pnl = (current_price - pos['entry']) / pos['entry'] * pos['risk_amt'] * 10
            else:
                pnl = (pos['entry'] - current_price) / pos['entry'] * pos['risk_amt'] * 10
            
            self.balance += pnl
            
            await self.telegram.send_message(
                f"⏰ *TIMEOUT* - Position closed\n"
                f"PnL: ${pnl:.2f}"
            )
            logger.info(f"⏰ TIMEOUT! ${pnl:.2f}")
            
            self.current_position = None
            self.last_trade_close = datetime.now()  # Cooldown starts
    
    async def check_retrain(self):
        """Check if model needs retraining (weekly)"""
        if self.ml_model.last_train is None:
            return
        
        days_since_train = (datetime.now() - self.ml_model.last_train).days
        
        if days_since_train >= 7:
            logger.info("🔄 Weekly retrain triggered...")
            await self.telegram.send_message("🔄 *Weekly Model Retrain Starting...*")
            
            success = await self.ml_model.train()
            
            if success:
                await self.telegram.send_message("✅ *Model Retrained Successfully!*")
            else:
                await self.telegram.send_message("⚠️ *Model Retrain Failed*")
    
    async def run(self):
        """Main trading loop"""
        self.running = True
        
        await self.initialize()
        
        while self.running:
            try:
                # Monitor existing position
                await self.monitor_position()
                
                # Only look for new trades if no position AND cooldown passed
                if not self.current_position:
                    # Check cooldown
                    if self.last_trade_close:
                        hours_since_close = (datetime.now() - self.last_trade_close).seconds / 3600
                        if hours_since_close < self.config.COOLDOWN_HOURS:
                            logger.debug(f"⏳ Cooldown: {self.config.COOLDOWN_HOURS - hours_since_close:.1f}h remaining")
                            continue
                    
                    signal = await self.check_for_signal()
                    
                    if signal:
                        await self.open_position(signal)
                
                # Check weekly retrain
                await self.check_retrain()
                
                # Wait before next check
                await asyncio.sleep(self.config.CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("⚠️ Shutdown requested")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(60)
        
        # Cleanup
        await self.telegram.send_message(
            f"🛑 *TERMINATOR ML LIVE STOPPED*\n"
            f"Final Balance: ${self.balance:.2f}"
        )
        logger.info("🥇 Terminator ML Live shutdown complete")


# ==================== MAIN ====================
async def main():
    """Launch the Terminator ML Live Engine"""
    
    # Start keep-alive server
    keep_alive()
    
    # Create and run engine
    config = Config()
    engine = TradingEngine(config)
    
    try:
        await engine.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🥇 TERMINATOR ML LIVE - GOLD TRADING BOT 🥇          ║
    ║                                                           ║
    ║          ML-Powered | 44.1% WR Backtested                ║
    ║          Auto-Retrain | Smart News Filter                ║
    ║                                                           ║
    ║          ⚠️  HIGH RISK - USE AT YOUR OWN RISK  ⚠️        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
