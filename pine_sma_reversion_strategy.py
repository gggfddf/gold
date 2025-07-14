#!/usr/bin/env python3
"""
PINE SCRIPT SMA REVERSION STRATEGY - PYTHON VERSION
Converted from Pine Script to Python for backtesting
Pure SMA reversion strategy without ML components

Original Pine Script:
- SMA Length: 50
- Hold Time: 12 candles
- Entry Distance: 0.32% from SMA
- LONG only strategy
- Initial Capital: $200
- Quantity: 1 contract

Author: AI Assistant
Strategy: SMA5 Reversion Strategy (Inputs Enabled)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class PineSMAReversionStrategy:
    def __init__(self, data_path, sma_length=50, hold_candles=12, entry_distance=0.0032, 
                 initial_capital=200, contract_quantity=1):
        """
        Initialize Pine SMA Reversion Strategy
        
        Args:
            data_path: Path to CSV data
            sma_length: SMA period (default: 50)
            hold_candles: Hold time in candles (default: 12)
            entry_distance: Entry distance as % of SMA (default: 0.32%)
            initial_capital: Initial capital in dollars (default: $200)
            contract_quantity: Number of contracts to trade (default: 1)
        """
        self.data_path = data_path
        self.sma_length = sma_length
        self.hold_candles = hold_candles
        self.entry_distance = entry_distance
        self.initial_capital = initial_capital
        self.contract_quantity = contract_quantity
        self.df = None
        self.trades = []
        
        # Trading costs for gold
        self.trading_cost = 0.14 / 34000  # $0.14 per $34,000 contract
        self.round_trip_cost = self.trading_cost * 2
        
    def load_and_prepare_data(self):
        """Load data and calculate SMA"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # Calculate SMA (equivalent to ta.sma(close, sma_length))
        self.df['sma'] = self.df['close'].rolling(window=self.sma_length).mean()
        
        # Calculate distance from SMA (equivalent to (sma - close) / sma)
        self.df['distance_from_sma'] = (self.df['sma'] - self.df['close']) / self.df['sma']
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        print(f"SMA Length: {self.sma_length}")
        print(f"Hold Candles: {self.hold_candles}")
        print(f"Entry Distance: {self.entry_distance*100:.2f}%")
        print(f"Initial Capital: ${self.initial_capital}")
        print(f"Contract Quantity: {self.contract_quantity}")
        
        return self.df
    
    def backtest_strategy(self):
        """Run Pine Script strategy backtest"""
        print("Running Pine Script strategy backtest...")
        
        self.trades = []
        position = None
        current_balance = self.initial_capital
        
        for i in range(len(self.df)):
            current_time = self.df.iloc[i]['time']
            current_price = self.df.iloc[i]['close']
            current_sma = self.df.iloc[i]['sma']
            current_distance = self.df.iloc[i]['distance_from_sma']
            
            # Check for exit if in position (equivalent to exit_condition)
            if position is not None:
                # Exit condition: bar_index - entry_bar_index >= hold_candles
                if i - position['entry_index'] >= self.hold_candles:
                    # Close position
                    exit_price = current_price
                    pnl_percent = self.calculate_pnl_percent(position, exit_price)
                    pnl_dollars = self.calculate_pnl_dollars(position, exit_price)
                    pnl_after_costs = pnl_dollars - (self.round_trip_cost * 34000 * self.contract_quantity)
                    
                    # Update balance
                    current_balance += pnl_after_costs
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'entry_sma': position['entry_sma'],
                        'exit_sma': current_sma,
                        'entry_distance': position['entry_distance'],
                        'pnl_percent': pnl_percent,
                        'pnl_dollars': pnl_dollars,
                        'pnl_after_costs': pnl_after_costs,
                        'balance_after_trade': current_balance,
                        'hold_candles': i - position['entry_index'],
                        'exit_reason': 'Timeout'
                    }
                    self.trades.append(trade)
                    position = None
            
            # Check for new entry if not in position (equivalent to entry_condition)
            if position is None:
                # Entry condition: distance_from_sma >= entry_distance
                if current_distance >= self.entry_distance:
                    # Enter LONG position
                    position = {
                        'entry_time': current_time,
                        'entry_index': i,
                        'entry_price': current_price,
                        'entry_sma': current_sma,
                        'entry_distance': current_distance
                    }
        
        # Close any remaining position at end
        if position is not None:
            exit_price = self.df.iloc[-1]['close']
            pnl_percent = self.calculate_pnl_percent(position, exit_price)
            pnl_dollars = self.calculate_pnl_dollars(position, exit_price)
            pnl_after_costs = pnl_dollars - (self.round_trip_cost * 34000 * self.contract_quantity)
            
            # Update balance
            current_balance += pnl_after_costs
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': self.df.iloc[-1]['time'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'entry_sma': position['entry_sma'],
                'exit_sma': self.df.iloc[-1]['sma'],
                'entry_distance': position['entry_distance'],
                'pnl_percent': pnl_percent,
                'pnl_dollars': pnl_dollars,
                'pnl_after_costs': pnl_after_costs,
                'balance_after_trade': current_balance,
                'hold_candles': len(self.df) - 1 - position['entry_index'],
                'exit_reason': 'End of data'
            }
            self.trades.append(trade)
        
        return self.trades
    
    def calculate_pnl_percent(self, position, exit_price):
        """Calculate P&L as percentage for a LONG trade"""
        entry_price = position['entry_price']
        return (exit_price - entry_price) / entry_price
    
    def calculate_pnl_dollars(self, position, exit_price):
        """Calculate P&L in dollars for a LONG trade"""
        entry_price = position['entry_price']
        pnl_per_contract = (exit_price - entry_price) * 100  # Gold contract = 100 oz
        return pnl_per_contract * self.contract_quantity
    
    def analyze_results(self):
        """Analyze Pine Script strategy results"""
        if not self.trades:
            print("No trades executed!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        print("\n" + "="*70)
        print("PINE SCRIPT SMA REVERSION STRATEGY RESULTS")
        print("="*70)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades_before = len(trades_df[trades_df['pnl_dollars'] > 0])
        losing_trades_before = len(trades_df[trades_df['pnl_dollars'] < 0])
        
        win_rate_before = winning_trades_before / total_trades * 100
        total_pnl_before = trades_df['pnl_dollars'].sum()
        avg_pnl_before = trades_df['pnl_dollars'].mean()
        
        # After costs
        winning_trades_after = len(trades_df[trades_df['pnl_after_costs'] > 0])
        losing_trades_after = len(trades_df[trades_df['pnl_after_costs'] < 0])
        
        win_rate_after = winning_trades_after / total_trades * 100
        total_pnl_after = trades_df['pnl_after_costs'].sum()
        avg_pnl_after = trades_df['pnl_after_costs'].mean()
        
        # Final balance
        final_balance = trades_df['balance_after_trade'].iloc[-1]
        total_return = (final_balance - self.initial_capital) / self.initial_capital * 100
        
        print(f"Strategy Parameters:")
        print(f"  SMA Length: {self.sma_length}")
        print(f"  Hold Candles: {self.hold_candles}")
        print(f"  Entry Distance: {self.entry_distance*100:.2f}%")
        print(f"  Initial Capital: ${self.initial_capital}")
        print(f"  Contract Quantity: {self.contract_quantity}")
        
        print(f"\nTrading Costs:")
        print(f"  Per Trade: ${self.round_trip_cost*34000:.2f}")
        print(f"  As Percentage: {self.round_trip_cost*100:.4f}%")
        
        print(f"\nPerformance Summary:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Initial Balance: ${self.initial_capital}")
        print(f"  Final Balance: ${final_balance:.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        
        print(f"\nBEFORE TRADING COSTS:")
        print(f"  Win Rate: {win_rate_before:.1f}%")
        print(f"  Total P&L: ${total_pnl_before:.2f}")
        print(f"  Average P&L: ${avg_pnl_before:.2f}")
        print(f"  Max Profit: ${trades_df['pnl_dollars'].max():.2f}")
        print(f"  Max Loss: ${trades_df['pnl_dollars'].min():.2f}")
        
        print(f"\nAFTER TRADING COSTS:")
        print(f"  Win Rate: {win_rate_after:.1f}%")
        print(f"  Total P&L: ${total_pnl_after:.2f}")
        print(f"  Average P&L: ${avg_pnl_after:.2f}")
        print(f"  Max Profit: ${trades_df['pnl_after_costs'].max():.2f}")
        print(f"  Max Loss: ${trades_df['pnl_after_costs'].min():.2f}")
        
        # Exit reason analysis
        print(f"\nExit Reasons:")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Distance analysis
        print(f"\nDistance Analysis:")
        print(f"  Average entry distance: {trades_df['entry_distance'].mean()*100:.3f}%")
        print(f"  Min entry distance: {trades_df['entry_distance'].min()*100:.3f}%")
        print(f"  Max entry distance: {trades_df['entry_distance'].max()*100:.3f}%")
        
        # Hold time analysis
        print(f"\nHold Time Analysis:")
        print(f"  Average hold time: {trades_df['hold_candles'].mean():.1f} candles")
        print(f"  Min hold time: {trades_df['hold_candles'].min()} candles")
        print(f"  Max hold time: {trades_df['hold_candles'].max()} candles")
        
        # Risk analysis
        print(f"\nRisk Analysis:")
        print(f"  Largest drawdown: ${min(trades_df['pnl_after_costs']):.2f}")
        print(f"  Best trade: ${max(trades_df['pnl_after_costs']):.2f}")
        print(f"  Profit factor: {abs(trades_df[trades_df['pnl_after_costs'] > 0]['pnl_after_costs'].sum() / trades_df[trades_df['pnl_after_costs'] < 0]['pnl_after_costs'].sum()):.2f}")
        
        # Save results
        trades_df.to_csv('pine_sma_strategy_results.csv', index=False)
        print(f"\nResults saved to: pine_sma_strategy_results.csv")
        
        # Create visualizations
        self.create_strategy_visualizations(trades_df)
    
    def create_strategy_visualizations(self, trades_df):
        """Create strategy performance visualizations"""
        print("Creating strategy visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Account Balance Over Time
        balance_curve = trades_df['balance_after_trade']
        exit_times = pd.to_datetime(trades_df['exit_time'])
        axes[0,0].plot(exit_times, balance_curve.values, linewidth=2, color='blue')
        axes[0,0].axhline(self.initial_capital, color='red', linestyle='--', label=f'Initial: ${self.initial_capital}')
        axes[0,0].set_title('Account Balance Over Time')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Balance ($)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].legend()
        
        # 2. P&L Distribution (after costs)
        axes[0,1].hist(trades_df['pnl_after_costs'], bins=20, alpha=0.7, color='green')
        axes[0,1].axvline(0, color='red', linestyle='--', label='Break-even')
        axes[0,1].set_title('P&L Distribution (After Costs)')
        axes[0,1].set_xlabel('P&L ($)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # 3. Entry Distance vs P&L
        axes[1,0].scatter(trades_df['entry_distance'], trades_df['pnl_after_costs'], alpha=0.6)
        axes[1,0].axhline(0, color='red', linestyle='--', label='Break-even')
        axes[1,0].set_title('Entry Distance vs P&L (After Costs)')
        axes[1,0].set_xlabel('Entry Distance from SMA')
        axes[1,0].set_ylabel('P&L ($)')
        axes[1,0].legend()
        
        # 4. Cumulative P&L over Time
        cumulative_pnl = trades_df['pnl_after_costs'].cumsum()
        exit_times = pd.to_datetime(trades_df['exit_time'])
        axes[1,1].plot(exit_times, cumulative_pnl.values, linewidth=2, color='green')
        axes[1,1].axhline(0, color='red', linestyle='--', label='Break-even')
        axes[1,1].set_title('Cumulative P&L (After Costs)')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Cumulative P&L ($)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('pine_sma_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Strategy visualizations saved to: pine_sma_strategy_performance.png")

def main():
    """Main execution function"""
    print("="*70)
    print("PINE SCRIPT SMA REVERSION STRATEGY - PYTHON VERSION")
    print("="*70)
    print("Converted from Pine Script to Python")
    print("Pure SMA reversion strategy without ML components")
    print("="*70)
    
    # Strategy parameters (from Pine Script)
    sma_length = 50
    hold_candles = 12
    entry_distance = 0.0032  # 0.32%
    initial_capital = 4000  # $4000 initial balance
    contract_quantity = 1  # 1 gold contract
    
    print(f"\nPINE SCRIPT PARAMETERS:")
    print(f"SMA Length: {sma_length}")
    print(f"Hold Candles: {hold_candles}")
    print(f"Entry Distance: {entry_distance*100:.2f}%")
    print(f"Initial Capital: ${initial_capital}")
    print(f"Contract Quantity: {contract_quantity}")
    
    print(f"\nSTRATEGY LOGIC:")
    print(f"1. Calculate SMA{50} of close prices")
    print(f"2. Enter LONG when price is 0.32% below SMA")
    print(f"3. Hold for exactly {hold_candles} candles")
    print(f"4. Exit at market price after hold time")
    print(f"5. Trade {contract_quantity} gold contract per trade")
    
    # Initialize strategy
    strategy = PineSMAReversionStrategy(
        data_path='stock_data/XAUUSDm_M5_1year_data.csv',
        sma_length=sma_length,
        hold_candles=hold_candles,
        entry_distance=entry_distance,
        initial_capital=initial_capital,
        contract_quantity=contract_quantity
    )
    
    # Load and prepare data
    strategy.load_and_prepare_data()
    
    # Run backtest
    trades = strategy.backtest_strategy()
    
    # Analyze results
    strategy.analyze_results()
    
    print(f"\n" + "="*70)
    print("PINE SCRIPT STRATEGY BACKTEST COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main() 