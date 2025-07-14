#!/usr/bin/env python3
"""
OPTIMIZED GOLD TRADING STRATEGY
Based on SMA Reversion Analysis for Real Trading Costs
$0.14 per $34,000 contract = 0.0004% trading cost

Author: AI Assistant
Strategy: Cost-Optimized SMA Reversion (Gold 5M)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptimizedGoldStrategy:
    def __init__(self, data_path):
        """Initialize Optimized Gold Strategy"""
        self.data_path = data_path
        self.df = None
        self.trades = []
        
        # Trading costs
        self.trading_cost = 0.14 / 34000  # $0.14 per $34,000 contract
        self.round_trip_cost = self.trading_cost * 2  # 0.0008%
        
        # OPTIMIZED PARAMETERS (based on analysis)
        self.min_distance = 0.002  # 0.2% minimum distance from SMA
        self.take_profit = 0.003   # 0.3% take profit
        self.stop_loss = 0.002     # 0.2% stop loss
        self.max_hold_candles = 10 # 10 candles max hold
        
    def load_and_prepare_data(self):
        """Load data and calculate technical features"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # Calculate SMAs
        self.df['SMA5'] = self.df['close'].rolling(window=5).mean()
        self.df['SMA10'] = self.df['close'].rolling(window=10).mean()
        
        # Calculate distance from SMA5
        self.df['distance_from_sma5'] = (self.df['close'] - self.df['SMA5']) / self.df['close']
        self.df['abs_distance_sma5'] = abs(self.df['distance_from_sma5'])
        
        # Calculate momentum
        self.df['price_momentum'] = self.df['close'].pct_change()
        self.df['sma_momentum'] = self.df['SMA5'].pct_change()
        
        # Calculate volatility
        self.df['volatility'] = self.df['close'].rolling(10).std() / self.df['close']
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records")
        return self.df
    
    def check_entry_conditions(self, index):
        """Check optimized entry conditions"""
        row = self.df.iloc[index]
        
        # Must be at least 0.2% away from SMA5
        if row['abs_distance_sma5'] < self.min_distance:
            return None, None, None
        
        # Determine direction
        if row['close'] < row['SMA5']:  # Price below SMA5
            direction = 'LONG'
            sma_value = row['SMA5']
        else:  # Price above SMA5
            direction = 'SHORT'
            sma_value = row['SMA5']
        
        return direction, 'SMA5', sma_value
    
    def check_exit_conditions(self, index, position):
        """Check optimized exit conditions"""
        current_price = self.df.iloc[index]['close']
        entry_price = position['entry_price']
        
        # Take Profit: 0.3%
        if position['direction'] == 'LONG' and current_price >= entry_price * (1 + self.take_profit):
            return 'Take Profit'
        elif position['direction'] == 'SHORT' and current_price <= entry_price * (1 - self.take_profit):
            return 'Take Profit'
        
        # Stop Loss: 0.2%
        if position['direction'] == 'LONG' and current_price <= entry_price * (1 - self.stop_loss):
            return 'Stop Loss'
        elif position['direction'] == 'SHORT' and current_price >= entry_price * (1 + self.stop_loss):
            return 'Stop Loss'
        
        # Timeout: 10 candles
        if index - position['entry_index'] >= self.max_hold_candles:
            return 'Timeout'
        
        return None
    
    def backtest_strategy(self):
        """Run optimized strategy backtest"""
        print("Running optimized strategy backtest...")
        
        self.trades = []
        position = None
        
        for i in range(len(self.df)):
            current_time = self.df.iloc[i]['time']
            current_price = self.df.iloc[i]['close']
            
            # Check for exit if in position
            if position is not None:
                exit_reason = self.check_exit_conditions(i, position)
                if exit_reason:
                    # Close position
                    exit_price = current_price
                    pnl = self.calculate_pnl(position, exit_price)
                    
                    # Apply trading costs
                    pnl_after_costs = pnl - self.round_trip_cost
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl_before_costs': pnl,
                        'pnl_after_costs': pnl_after_costs,
                        'exit_reason': exit_reason,
                        'hold_candles': i - position['entry_index'],
                        'distance_from_sma': position['distance_from_sma']
                    }
                    self.trades.append(trade)
                    position = None
            
            # Check for new entry if not in position
            if position is None:
                direction, sma_col, sma_value = self.check_entry_conditions(i)
                
                if direction is not None:
                    # Enter position
                    position = {
                        'entry_time': current_time,
                        'entry_index': i,
                        'direction': direction,
                        'entry_price': current_price,
                        'sma_col': sma_col,
                        'sma_value': sma_value,
                        'distance_from_sma': self.df.iloc[i]['abs_distance_sma5']
                    }
        
        # Close any remaining position at end
        if position is not None:
            exit_price = self.df.iloc[-1]['close']
            pnl = self.calculate_pnl(position, exit_price)
            pnl_after_costs = pnl - self.round_trip_cost
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': self.df.iloc[-1]['time'],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl_before_costs': pnl,
                'pnl_after_costs': pnl_after_costs,
                'exit_reason': 'End of data',
                'hold_candles': len(self.df) - 1 - position['entry_index'],
                'distance_from_sma': position['distance_from_sma']
            }
            self.trades.append(trade)
        
        return self.trades
    
    def calculate_pnl(self, position, exit_price):
        """Calculate P&L for a trade"""
        entry_price = position['entry_price']
        
        if position['direction'] == 'LONG':
            return (exit_price - entry_price) / entry_price
        else:  # SHORT
            return (entry_price - exit_price) / entry_price
    
    def analyze_results(self):
        """Analyze optimized strategy results"""
        if not self.trades:
            print("No trades executed!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        print("\n" + "="*70)
        print("OPTIMIZED GOLD STRATEGY RESULTS")
        print("="*70)
        
        # Before costs
        total_trades = len(trades_df)
        winning_trades_before = len(trades_df[trades_df['pnl_before_costs'] > 0])
        losing_trades_before = len(trades_df[trades_df['pnl_before_costs'] < 0])
        
        win_rate_before = winning_trades_before / total_trades * 100
        total_pnl_before = trades_df['pnl_before_costs'].sum()
        avg_pnl_before = trades_df['pnl_before_costs'].mean()
        
        # After costs
        winning_trades_after = len(trades_df[trades_df['pnl_after_costs'] > 0])
        losing_trades_after = len(trades_df[trades_df['pnl_after_costs'] < 0])
        
        win_rate_after = winning_trades_after / total_trades * 100
        total_pnl_after = trades_df['pnl_after_costs'].sum()
        avg_pnl_after = trades_df['pnl_after_costs'].mean()
        
        print(f"Total Trades: {total_trades}")
        print(f"\nBEFORE TRADING COSTS:")
        print(f"  Win Rate: {win_rate_before:.1f}%")
        print(f"  Total P&L: {total_pnl_before:.4f} ({total_pnl_before*100:.2f}%)")
        print(f"  Average P&L: {avg_pnl_before:.4f} ({avg_pnl_before*100:.2f}%)")
        
        print(f"\nAFTER TRADING COSTS:")
        print(f"  Win Rate: {win_rate_after:.1f}%")
        print(f"  Total P&L: {total_pnl_after:.4f} ({total_pnl_after*100:.2f}%)")
        print(f"  Average P&L: {avg_pnl_after:.4f} ({avg_pnl_after*100:.2f}%)")
        print(f"  Trading Costs: ${self.round_trip_cost*34000:.2f} per trade")
        
        # Exit reason analysis
        print(f"\nExit Reasons:")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Direction analysis
        print(f"\nDirection Performance (After Costs):")
        direction_perf = trades_df.groupby('direction').agg({
            'pnl_after_costs': ['count', 'mean', 'sum'],
            'distance_from_sma': 'mean'
        }).round(4)
        print(direction_perf)
        
        # Distance analysis
        print(f"\nDistance Analysis:")
        print(f"Average distance from SMA: {trades_df['distance_from_sma'].mean()*100:.3f}%")
        print(f"Min distance: {trades_df['distance_from_sma'].min()*100:.3f}%")
        print(f"Max distance: {trades_df['distance_from_sma'].max()*100:.3f}%")
        
        # Save results
        trades_df.to_csv('optimized_gold_strategy_results.csv', index=False)
        print(f"\nResults saved to: optimized_gold_strategy_results.csv")
        
        # Create visualizations
        self.create_strategy_visualizations(trades_df)
    
    def create_strategy_visualizations(self, trades_df):
        """Create strategy performance visualizations"""
        print("Creating strategy visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cumulative P&L (before and after costs)
        cumulative_before = trades_df['pnl_before_costs'].cumsum()
        cumulative_after = trades_df['pnl_after_costs'].cumsum()
        
        axes[0,0].plot(cumulative_before.index, cumulative_before.values, 
                      linewidth=2, color='blue', label='Before Costs')
        axes[0,0].plot(cumulative_after.index, cumulative_after.values, 
                      linewidth=2, color='red', label='After Costs')
        axes[0,0].set_title('Cumulative P&L')
        axes[0,0].set_ylabel('Cumulative P&L')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 2. P&L Distribution (after costs)
        axes[0,1].hist(trades_df['pnl_after_costs'], bins=20, alpha=0.7, color='green')
        axes[0,1].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[0,1].set_title('P&L Distribution (After Costs)')
        axes[0,1].set_xlabel('P&L')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # 3. Exit Reasons
        exit_reasons = trades_df['exit_reason'].value_counts()
        exit_reasons.plot(kind='bar', ax=axes[1,0], color='orange')
        axes[1,0].set_title('Exit Reasons')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Distance vs P&L
        axes[1,1].scatter(trades_df['distance_from_sma'], trades_df['pnl_after_costs'], alpha=0.6)
        axes[1,1].axhline(0, color='red', linestyle='--', label='Break-even')
        axes[1,1].set_title('Distance from SMA vs P&L (After Costs)')
        axes[1,1].set_xlabel('Distance from SMA')
        axes[1,1].set_ylabel('P&L (After Costs)')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('optimized_gold_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Strategy visualizations saved to: optimized_gold_strategy_performance.png")

def main():
    """Main execution function"""
    print("="*70)
    print("OPTIMIZED GOLD TRADING STRATEGY")
    print("="*70)
    print("Based on SMA Reversion Analysis for Real Trading Costs")
    print("="*70)
    
    # Strategy parameters
    print(f"\nSTRATEGY PARAMETERS:")
    print(f"Trading Cost: ${0.14:.2f} per ${34000} contract")
    print(f"Round Trip Cost: {0.14*2/34000*100:.4f}%")
    print(f"Minimum Distance: {0.002*100:.1f}% from SMA5")
    print(f"Take Profit: {0.003*100:.1f}%")
    print(f"Stop Loss: {0.002*100:.1f}%")
    print(f"Max Hold Time: 10 candles (50 minutes)")
    
    print(f"\nENTRY RULES:")
    print(f"1. Price must be at least 0.2% away from SMA5")
    print(f"2. LONG: Price below SMA5")
    print(f"3. SHORT: Price above SMA5")
    
    print(f"\nEXIT RULES:")
    print(f"1. Take Profit: 0.3% (3x trading cost)")
    print(f"2. Stop Loss: 0.2% (2.5x trading cost)")
    print(f"3. Timeout: 10 candles (50 minutes)")
    
    # Initialize strategy
    strategy = OptimizedGoldStrategy('stock_data/XAUUSDm_M5_1year_data.csv')
    
    # Load and prepare data
    strategy.load_and_prepare_data()
    
    # Run backtest
    trades = strategy.backtest_strategy()
    
    # Analyze results
    strategy.analyze_results()
    
    print(f"\n" + "="*70)
    print("OPTIMIZED STRATEGY BACKTEST COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main() 