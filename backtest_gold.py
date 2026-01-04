"""
TERMINATOR GOLD BACKTESTER
Automated backtesting system for SMC Strategy

Uses yfinance for FREE 1-year historical Gold data
Tests the SMC strategy and generates complete statistics

Author: Terminator Genesis Engine
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


# ==================== TRADE RESULT ====================
@dataclass
class Trade:
    """Single trade record"""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl_dollars: float
    pnl_percent: float
    result: str  # 'WIN', 'LOSS', 'BREAKEVEN'
    rr_achieved: float
    exit_reason: str  # 'TP1', 'TP2', 'TP3', 'SL', 'TIMEOUT'
    trade_type: str  # 'SCALP', 'STANDARD', 'SWING'


# ==================== SMC BACKTESTER ENGINE ====================
class SMCBacktester:
    """
    Backtesting engine for Gold SMC Strategy
    
    Features:
    - 1 year free historical data from yfinance
    - Full SMC strategy simulation (ZigZag, CHoCH, BOS)
    - Realistic trade execution
    - Complete statistics
    """
    
    def __init__(self, 
                 starting_balance: float = 1000.0,
                 risk_percent: float = 1.0):
        """
        Initialize backtester
        
        Args:
            starting_balance: Starting account balance in USD
            risk_percent: Risk per trade as percentage
        """
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.risk_percent = risk_percent
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.current_position: Optional[Dict] = None
        
        # Statistics
        self.max_balance = starting_balance
        self.min_balance = starting_balance
        self.peak_balance = starting_balance
        self.max_drawdown = 0.0
        
        # SMC Config (matching main bot)
        self.zigzag_depth = 5
        self.zigzag_deviation = 0.001
        self.sl_limits = {"SCALP": 6, "STANDARD": 10, "SWING": 25}
        self.min_score = 60
        
    def fetch_historical_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Fetch 1 year of Gold futures data from yfinance
        
        Returns DataFrame with OHLCV data
        """
        print("📊 Fetching 1 year Gold historical data from yfinance...")
        
        # GC=F is Gold Futures - most liquid and available
        df = yf.download("GC=F", period=period, interval="1h", progress=True)
        
        if df.empty:
            raise ValueError("Failed to fetch Gold data from yfinance")
        
        # Reset index and rename columns
        df = df.reset_index()
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        print(f"✅ Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators needed for SMC strategy"""
        # Returns
        df['returns'] = df['close'].pct_change()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # ADX
        df['plus_dm'] = df['high'].diff()
        df['minus_dm'] = -df['low'].diff()
        df['plus_dm'] = df['plus_dm'].where((df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), 0)
        df['minus_dm'] = df['minus_dm'].where((df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), 0)
        
        atr_14 = df['atr']
        df['plus_di'] = 100 * (df['plus_dm'].rolling(14).mean() / atr_14)
        df['minus_di'] = 100 * (df['minus_dm'].rolling(14).mean() / atr_14)
        df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']))
        df['adx'] = df['dx'].rolling(14).mean()
        
        return df.dropna()
    
    def find_zigzag_pivots(self, df: pd.DataFrame, start_idx: int, lookback: int = 100) -> List[Dict]:
        """Find ZigZag swing highs and lows"""
        if start_idx < lookback:
            lookback = start_idx
        
        subset = df.iloc[start_idx - lookback:start_idx + 1].copy()
        if len(subset) < self.zigzag_depth * 2:
            return []
        
        pivots = []
        depth = self.zigzag_depth
        
        highs = subset['high'].values
        lows = subset['low'].values
        
        # Find swing highs
        for i in range(depth, len(subset) - depth):
            window_highs = highs[i - depth:i + depth + 1]
            if highs[i] == max(window_highs):
                left_low = min(lows[i - depth:i])
                if (highs[i] - left_low) / left_low >= self.zigzag_deviation:
                    pivots.append({
                        'type': 'HIGH',
                        'price': float(highs[i]),
                        'index': start_idx - lookback + i
                    })
        
        # Find swing lows
        for i in range(depth, len(subset) - depth):
            window_lows = lows[i - depth:i + depth + 1]
            if lows[i] == min(window_lows):
                left_high = max(highs[i - depth:i])
                if (left_high - lows[i]) / lows[i] >= self.zigzag_deviation:
                    pivots.append({
                        'type': 'LOW',
                        'price': float(lows[i]),
                        'index': start_idx - lookback + i
                    })
        
        pivots.sort(key=lambda x: x['index'])
        
        # Filter consecutive same types
        filtered = []
        last_type = None
        for p in pivots:
            if p['type'] != last_type:
                filtered.append(p)
                last_type = p['type']
            elif filtered:
                if p['type'] == 'HIGH' and p['price'] > filtered[-1]['price']:
                    filtered[-1] = p
                elif p['type'] == 'LOW' and p['price'] < filtered[-1]['price']:
                    filtered[-1] = p
        
        return filtered
    
    def detect_smc_structure(self, df: pd.DataFrame, idx: int) -> Dict:
        """Detect SMC structure at given index"""
        pivots = self.find_zigzag_pivots(df, idx)
        
        result = {
            'trend': 'RANGING',
            'choch_bullish': False,
            'choch_bearish': False,
            'bos_bullish': False,
            'bos_bearish': False,
            'last_swing_high': None,
            'last_swing_low': None,
            'bias': 'NEUTRAL'
        }
        
        if len(pivots) < 4:
            return result
        
        swing_highs = [p for p in pivots if p['type'] == 'HIGH'][-3:]
        swing_lows = [p for p in pivots if p['type'] == 'LOW'][-3:]
        
        if not swing_highs or not swing_lows:
            return result
        
        result['last_swing_high'] = swing_highs[-1]['price']
        result['last_swing_low'] = swing_lows[-1]['price']
        
        current_price = float(df['close'].iloc[idx])
        
        # Determine trend
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]['price'] > swing_highs[-2]['price']
            hl = swing_lows[-1]['price'] > swing_lows[-2]['price']
            lh = swing_highs[-1]['price'] < swing_highs[-2]['price']
            ll = swing_lows[-1]['price'] < swing_lows[-2]['price']
            
            if hh and hl:
                result['trend'] = 'BULLISH'
            elif lh and ll:
                result['trend'] = 'BEARISH'
        
        # Detect CHoCH and BOS
        if result['trend'] == 'BEARISH' and current_price > result['last_swing_high']:
            result['choch_bullish'] = True
            result['bias'] = 'LONG'
        elif result['trend'] == 'BULLISH' and current_price < result['last_swing_low']:
            result['choch_bearish'] = True
            result['bias'] = 'SHORT'
        elif result['trend'] == 'BULLISH' and current_price > result['last_swing_high']:
            result['bos_bullish'] = True
            result['bias'] = 'LONG'
        elif result['trend'] == 'BEARISH' and current_price < result['last_swing_low']:
            result['bos_bearish'] = True
            result['bias'] = 'SHORT'
        elif result['trend'] == 'BULLISH':
            result['bias'] = 'LONG'
        elif result['trend'] == 'BEARISH':
            result['bias'] = 'SHORT'
        
        return result
    
    def calculate_score(self, df: pd.DataFrame, idx: int, structure: Dict) -> int:
        """Calculate signal score similar to main bot"""
        score = 0
        row = df.iloc[idx]
        
        # RSI score
        rsi = row['rsi']
        if 30 < rsi < 70:
            score += 10
        
        # ADX score (trend strength)
        adx = row['adx']
        if adx > 25:
            score += 15
        elif adx > 20:
            score += 10
        
        # Moving average alignment
        price = row['close']
        if price > row['sma_20'] > row['sma_50']:
            score += 15  # Bullish alignment
        elif price < row['sma_20'] < row['sma_50']:
            score += 15  # Bearish alignment
        
        # SMC structure bonus
        if structure['choch_bullish'] or structure['choch_bearish']:
            score += 25  # CHoCH is a strong signal
        elif structure['bos_bullish'] or structure['bos_bearish']:
            score += 20  # BOS is also good
        elif structure['bias'] != 'NEUTRAL':
            score += 10
        
        # Trend alignment
        if structure['trend'] != 'RANGING':
            score += 10
        
        return score
    
    def determine_trade_type(self, score: int) -> str:
        """Determine trade type based on score"""
        if score >= 80:
            return "SWING"
        elif score >= 70:
            return "STANDARD"
        else:
            return "SCALP"
    
    def calculate_sl_tp(self, df: pd.DataFrame, idx: int, direction: str, 
                        structure: Dict, trade_type: str) -> Tuple[float, float, float, float]:
        """Calculate SL and TP levels"""
        price = float(df['close'].iloc[idx])
        atr = float(df['atr'].iloc[idx])
        
        max_sl = self.sl_limits[trade_type]
        
        if direction == 'LONG':
            # SL below swing low
            swing_low = structure.get('last_swing_low')
            if swing_low and price - swing_low < max_sl:
                sl = swing_low - 1  # $1 buffer
            else:
                sl = price - max_sl
            
            sl_distance = price - sl
            tp1 = price + sl_distance * 3  # 1:3 RR
            tp2 = price + sl_distance * 5  # 1:5 RR
            tp3 = price + sl_distance * 8  # 1:8 RR
        else:
            # SL above swing high
            swing_high = structure.get('last_swing_high')
            if swing_high and swing_high - price < max_sl:
                sl = swing_high + 1  # $1 buffer
            else:
                sl = price + max_sl
            
            sl_distance = sl - price
            tp1 = price - sl_distance * 3
            tp2 = price - sl_distance * 5
            tp3 = price - sl_distance * 8
        
        return sl, tp1, tp2, tp3
    
    def simulate_trade_exit(self, df: pd.DataFrame, entry_idx: int, 
                           direction: str, entry_price: float,
                           sl: float, tp1: float, tp2: float, tp3: float) -> Tuple[int, float, str]:
        """
        Simulate trade execution and find exit point
        
        Returns: (exit_idx, exit_price, exit_reason)
        """
        max_candles = 100  # Max 100 candles (~4 days at 1H)
        
        for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
            high = df['high'].iloc[i]
            low = df['low'].iloc[i]
            
            if direction == 'LONG':
                # Check SL hit first (conservative)
                if low <= sl:
                    return i, sl, 'SL'
                # Check TP hits
                if high >= tp3:
                    return i, tp3, 'TP3'
                if high >= tp2:
                    return i, tp2, 'TP2'
                if high >= tp1:
                    return i, tp1, 'TP1'
            else:  # SHORT
                # Check SL hit first
                if high >= sl:
                    return i, sl, 'SL'
                # Check TP hits
                if low <= tp3:
                    return i, tp3, 'TP3'
                if low <= tp2:
                    return i, tp2, 'TP2'
                if low <= tp1:
                    return i, tp1, 'TP1'
        
        # Timeout - exit at last available price
        exit_idx = min(entry_idx + max_candles - 1, len(df) - 1)
        return exit_idx, df['close'].iloc[exit_idx], 'TIMEOUT'
    
    def run_backtest(self) -> Dict:
        """
        Run the full backtest
        
        Returns dictionary with all results
        """
        print("\n" + "="*60)
        print("🥇 TERMINATOR GOLD BACKTESTER - SMC STRATEGY 🥇")
        print("="*60 + "\n")
        
        # Fetch data
        df = self.fetch_historical_data("1y")
        df = self.calculate_indicators(df)
        
        print(f"\n🔄 Running backtest on {len(df)} candles...")
        print(f"💰 Starting balance: ${self.starting_balance:.2f}")
        print(f"📊 Risk per trade: {self.risk_percent}%\n")
        
        # Track balance history for drawdown
        balance_history = [self.starting_balance]
        
        # Main backtest loop
        i = 200  # Start after enough data for indicators
        while i < len(df) - 100:  # Leave room for trade simulation
            
            # Skip if already in position
            if self.current_position:
                i += 1
                continue
            
            # Detect SMC structure
            structure = self.detect_smc_structure(df, i)
            
            # Skip if no clear bias
            if structure['bias'] == 'NEUTRAL':
                i += 1
                continue
            
            # Calculate score
            score = self.calculate_score(df, i, structure)
            
            # Skip if score too low
            if score < self.min_score:
                i += 1
                continue
            
            # Determine direction and trade type
            direction = structure['bias']
            trade_type = self.determine_trade_type(score)
            
            # Calculate SL/TP
            entry_price = float(df['close'].iloc[i])
            sl, tp1, tp2, tp3 = self.calculate_sl_tp(df, i, direction, structure, trade_type)
            
            # Simulate trade
            exit_idx, exit_price, exit_reason = self.simulate_trade_exit(
                df, i, direction, entry_price, sl, tp1, tp2, tp3
            )
            
            # Calculate P/L
            if direction == 'LONG':
                pnl_percent = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_percent = (entry_price - exit_price) / entry_price * 100
            
            # Calculate dollar P/L based on risk
            risk_amount = self.current_balance * (self.risk_percent / 100)
            sl_distance = abs(entry_price - sl)
            profit_distance = abs(exit_price - entry_price)  # Moved here
            rr_achieved = profit_distance / sl_distance if sl_distance > 0 else 0
            
            if exit_reason == 'SL':
                pnl_dollars = -risk_amount
            else:
                # Winning trade - use RR achieved
                pnl_dollars = risk_amount * rr_achieved
            
            # Determine result
            if pnl_dollars > 0:
                result = 'WIN'
            elif pnl_dollars < 0:
                result = 'LOSS'
            else:
                result = 'BREAKEVEN'
            
            # Update balance
            self.current_balance += pnl_dollars
            balance_history.append(self.current_balance)
            
            # Track max/min balance
            self.max_balance = max(self.max_balance, self.current_balance)
            self.min_balance = min(self.min_balance, self.current_balance)
            
            # Calculate drawdown
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            drawdown = (self.peak_balance - self.current_balance) / self.peak_balance * 100
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            # Record trade
            trade = Trade(
                entry_time=df['timestamp'].iloc[i],
                exit_time=df['timestamp'].iloc[exit_idx],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_loss=sl,
                take_profit=tp1,  # Primary TP
                pnl_dollars=pnl_dollars,
                pnl_percent=pnl_percent,
                result=result,
                rr_achieved=rr_achieved,
                exit_reason=exit_reason,
                trade_type=trade_type
            )
            self.trades.append(trade)
            
            # Move to after exit
            i = exit_idx + 1
        
        # Calculate final statistics
        return self.calculate_statistics()
    
    def calculate_statistics(self) -> Dict:
        """Calculate comprehensive statistics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        wins = [t for t in self.trades if t.result == 'WIN']
        losses = [t for t in self.trades if t.result == 'LOSS']
        
        total_trades = len(self.trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t.pnl_dollars for t in self.trades)
        profit_percent = (self.current_balance - self.starting_balance) / self.starting_balance * 100
        
        biggest_win = max((t.pnl_dollars for t in wins), default=0)
        biggest_loss = min((t.pnl_dollars for t in losses), default=0)
        
        avg_win = np.mean([t.pnl_dollars for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl_dollars for t in losses]) if losses else 0
        
        avg_rr = np.mean([t.rr_achieved for t in wins]) if wins else 0
        
        # Trade type breakdown
        scalp_trades = [t for t in self.trades if t.trade_type == 'SCALP']
        standard_trades = [t for t in self.trades if t.trade_type == 'STANDARD']
        swing_trades = [t for t in self.trades if t.trade_type == 'SWING']
        
        # Direction breakdown
        long_trades = [t for t in self.trades if t.direction == 'LONG']
        short_trades = [t for t in self.trades if t.direction == 'SHORT']
        
        long_wins = len([t for t in long_trades if t.result == 'WIN'])
        short_wins = len([t for t in short_trades if t.result == 'WIN'])
        
        stats = {
            "summary": {
                "starting_balance": self.starting_balance,
                "ending_balance": round(self.current_balance, 2),
                "total_profit_loss": round(total_profit, 2),
                "profit_percent": round(profit_percent, 2),
                "max_drawdown_percent": round(self.max_drawdown, 2),
            },
            "trade_stats": {
                "total_trades": total_trades,
                "wins": win_count,
                "losses": loss_count,
                "win_rate_percent": round(win_rate, 2),
                "biggest_win": round(biggest_win, 2),
                "biggest_loss": round(biggest_loss, 2),
                "average_win": round(avg_win, 2),
                "average_loss": round(avg_loss, 2),
                "average_rr_on_wins": round(avg_rr, 2),
            },
            "trade_types": {
                "scalp_trades": len(scalp_trades),
                "standard_trades": len(standard_trades),
                "swing_trades": len(swing_trades),
            },
            "direction_stats": {
                "long_trades": len(long_trades),
                "long_wins": long_wins,
                "long_win_rate": round(long_wins / len(long_trades) * 100, 2) if long_trades else 0,
                "short_trades": len(short_trades),
                "short_wins": short_wins,
                "short_win_rate": round(short_wins / len(short_trades) * 100, 2) if short_trades else 0,
            },
            "exit_reasons": {
                "TP1": len([t for t in self.trades if t.exit_reason == 'TP1']),
                "TP2": len([t for t in self.trades if t.exit_reason == 'TP2']),
                "TP3": len([t for t in self.trades if t.exit_reason == 'TP3']),
                "SL": len([t for t in self.trades if t.exit_reason == 'SL']),
                "TIMEOUT": len([t for t in self.trades if t.exit_reason == 'TIMEOUT']),
            }
        }
        
        return stats
    
    def print_report(self, stats: Dict):
        """Print formatted backtest report"""
        print("\n" + "="*60)
        print("📊 BACKTEST RESULTS - GOLD SMC STRATEGY")
        print("="*60)
        
        s = stats["summary"]
        print(f"""
💰 ACCOUNT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Starting Balance:     ${s['starting_balance']:.2f}
• Ending Balance:       ${s['ending_balance']:.2f}
• Total P/L:            ${s['total_profit_loss']:.2f} ({s['profit_percent']:+.2f}%)
• Max Drawdown:         {s['max_drawdown_percent']:.2f}%
""")
        
        t = stats["trade_stats"]
        print(f"""📈 TRADE STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Trades:         {t['total_trades']}
• Wins:                 {t['wins']}
• Losses:               {t['losses']}
• Win Rate:             {t['win_rate_percent']:.2f}%
• Biggest Win:          ${t['biggest_win']:.2f}
• Biggest Loss:         ${t['biggest_loss']:.2f}
• Average Win:          ${t['average_win']:.2f}
• Average Loss:         ${t['average_loss']:.2f}
• Avg RR on Wins:       {t['average_rr_on_wins']:.2f}
""")
        
        tt = stats["trade_types"]
        print(f"""🏷️ TRADE TYPES
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• SCALP trades:         {tt['scalp_trades']}
• STANDARD trades:      {tt['standard_trades']}
• SWING trades:         {tt['swing_trades']}
""")
        
        d = stats["direction_stats"]
        print(f"""📍 DIRECTION BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• LONG trades:          {d['long_trades']} ({d['long_win_rate']:.1f}% win rate)
• SHORT trades:         {d['short_trades']} ({d['short_win_rate']:.1f}% win rate)
""")
        
        e = stats["exit_reasons"]
        print(f"""🎯 EXIT REASONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• TP1 (1:3 RR):         {e['TP1']}
• TP2 (1:5 RR):         {e['TP2']}
• TP3 (1:8 RR):         {e['TP3']}
• Stop Loss:            {e['SL']}
• Timeout:              {e['TIMEOUT']}
""")
        
        print("="*60)
        print("🥇 TERMINATOR GOLD BACKTESTER COMPLETE 🥇")
        print("="*60 + "\n")
    
    def save_report(self, stats: Dict, filename: str = "backtest_results.json"):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        print(f"📄 Results saved to {filename}")


# ==================== MAIN ====================
def main():
    """Run the backtester"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     🥇 TERMINATOR GOLD BACKTESTER 🥇                     ║
    ║                                                           ║
    ║     SMC Strategy Automated Backtest                       ║
    ║     1 Year Historical Data | Full Statistics              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Initialize backtester
    backtester = SMCBacktester(
        starting_balance=1000.0,  # $1000 starting
        risk_percent=1.0          # 1% risk per trade
    )
    
    # Run backtest
    stats = backtester.run_backtest()
    
    # Print report
    backtester.print_report(stats)
    
    # Save to file
    backtester.save_report(stats)
    
    return stats


if __name__ == "__main__":
    main()
