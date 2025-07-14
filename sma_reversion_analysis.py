#!/usr/bin/env python3
"""
SMA REVERSION ANALYSIS FOR GOLD TRADING
Analyze maximum distance from SMA and reversion behavior
Optimize strategy for real trading costs ($0.14 per $34,000 contract)

Author: AI Assistant
Purpose: Quant analysis for profitable gold trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class SMAReversionAnalysis:
    def __init__(self, data_path):
        """Initialize SMA Reversion Analysis"""
        self.data_path = data_path
        self.df = None
        self.trading_cost = 0.14 / 34000  # $0.14 per $34,000 contract
        
    def load_and_prepare_data(self):
        """Load data and calculate SMA distances"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # Calculate SMAs
        self.df['SMA5'] = self.df['close'].rolling(window=5).mean()
        self.df['SMA10'] = self.df['close'].rolling(window=10).mean()
        self.df['SMA20'] = self.df['close'].rolling(window=20).mean()
        
        # Calculate distance from SMAs
        self.df['distance_from_sma5'] = (self.df['close'] - self.df['SMA5']) / self.df['close']
        self.df['distance_from_sma10'] = (self.df['close'] - self.df['SMA10']) / self.df['close']
        self.df['distance_from_sma20'] = (self.df['close'] - self.df['SMA20']) / self.df['close']
        
        # Absolute distances
        self.df['abs_distance_sma5'] = abs(self.df['distance_from_sma5'])
        self.df['abs_distance_sma10'] = abs(self.df['distance_from_sma10'])
        self.df['abs_distance_sma20'] = abs(self.df['distance_from_sma20'])
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        return self.df
    
    def analyze_sma_distances(self):
        """Analyze maximum distances from SMA"""
        print("\n" + "="*60)
        print("SMA DISTANCE ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print(f"\nDistance from SMA5 Statistics:")
        print(f"Mean: {self.df['abs_distance_sma5'].mean():.6f} ({self.df['abs_distance_sma5'].mean()*100:.4f}%)")
        print(f"Median: {self.df['abs_distance_sma5'].median():.6f} ({self.df['abs_distance_sma5'].median()*100:.4f}%)")
        print(f"Std: {self.df['abs_distance_sma5'].std():.6f} ({self.df['abs_distance_sma5'].std()*100:.4f}%)")
        print(f"Max: {self.df['abs_distance_sma5'].max():.6f} ({self.df['abs_distance_sma5'].max()*100:.4f}%)")
        print(f"95th percentile: {self.df['abs_distance_sma5'].quantile(0.95):.6f} ({self.df['abs_distance_sma5'].quantile(0.95)*100:.4f}%)")
        print(f"99th percentile: {self.df['abs_distance_sma5'].quantile(0.99):.6f} ({self.df['abs_distance_sma5'].quantile(0.99)*100:.4f}%)")
        
        # Distance buckets
        distance_buckets = [0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05]
        bucket_labels = ['0-0.1%', '0.1-0.2%', '0.2-0.3%', '0.3-0.5%', '0.5-1%', '1-2%', '2-5%']
        
        print(f"\nDistance Distribution (SMA5):")
        for i in range(len(distance_buckets)-1):
            count = len(self.df[(self.df['abs_distance_sma5'] >= distance_buckets[i]) & 
                               (self.df['abs_distance_sma5'] < distance_buckets[i+1])])
            pct = count / len(self.df) * 100
            print(f"  {bucket_labels[i]}: {count} candles ({pct:.1f}%)")
        
        # Trading cost analysis
        print(f"\nTrading Cost Analysis:")
        print(f"Trading cost: ${self.trading_cost*34000:.2f} per ${34000} contract")
        print(f"Trading cost as %: {self.trading_cost*100:.4f}%")
        
        # Minimum profitable distance
        min_profitable_distance = self.trading_cost * 2  # Round trip
        print(f"Minimum profitable distance (round trip): {min_profitable_distance*100:.4f}%")
        
        # Count candles above minimum profitable distance
        profitable_candles = len(self.df[self.df['abs_distance_sma5'] >= min_profitable_distance])
        print(f"Candles above minimum profitable distance: {profitable_candles} ({profitable_candles/len(self.df)*100:.1f}%)")
        
        return min_profitable_distance
    
    def analyze_reversion_behavior(self, lookback_periods=[5, 10, 15, 20]):
        """Analyze reversion behavior after reaching certain distances"""
        print(f"\n" + "="*60)
        print("REVERSION BEHAVIOR ANALYSIS")
        print("="*60)
        
        reversion_data = []
        
        for lookback in lookback_periods:
            print(f"\nAnalyzing {lookback}-candle reversion behavior...")
            
            # Find candles with significant distance from SMA5
            significant_distances = [0.001, 0.002, 0.003, 0.005, 0.01]
            
            for distance in significant_distances:
                # Find candles where price is significantly away from SMA5
                long_candidates = self.df[(self.df['close'] < self.df['SMA5']) & 
                                        (self.df['abs_distance_sma5'] >= distance)]
                short_candidates = self.df[(self.df['close'] > self.df['SMA5']) & 
                                         (self.df['abs_distance_sma5'] >= distance)]
                
                # Analyze reversion for LONG candidates
                long_reversions = 0
                long_total = 0
                long_pnl = []
                
                for idx in long_candidates.index:
                    if idx + lookback < len(self.df):
                        long_total += 1
                        future_price = self.df.iloc[idx + lookback]['close']
                        entry_price = self.df.iloc[idx]['close']
                        pnl = (future_price - entry_price) / entry_price
                        long_pnl.append(pnl)
                        
                        if future_price >= self.df.iloc[idx + lookback]['SMA5']:
                            long_reversions += 1
                
                # Analyze reversion for SHORT candidates
                short_reversions = 0
                short_total = 0
                short_pnl = []
                
                for idx in short_candidates.index:
                    if idx + lookback < len(self.df):
                        short_total += 1
                        future_price = self.df.iloc[idx + lookback]['close']
                        entry_price = self.df.iloc[idx]['close']
                        pnl = (entry_price - future_price) / entry_price
                        short_pnl.append(pnl)
                        
                        if future_price <= self.df.iloc[idx + lookback]['SMA5']:
                            short_reversions += 1
                
                # Calculate statistics
                if long_total > 0:
                    long_reversion_rate = long_reversions / long_total
                    long_avg_pnl = np.mean(long_pnl)
                    long_win_rate = len([p for p in long_pnl if p > self.trading_cost]) / len(long_pnl)
                else:
                    long_reversion_rate = 0
                    long_avg_pnl = 0
                    long_win_rate = 0
                
                if short_total > 0:
                    short_reversion_rate = short_reversions / short_total
                    short_avg_pnl = np.mean(short_pnl)
                    short_win_rate = len([p for p in short_pnl if p > self.trading_cost]) / len(short_pnl)
                else:
                    short_reversion_rate = 0
                    short_avg_pnl = 0
                    short_win_rate = 0
                
                # Store results
                reversion_data.append({
                    'lookback': lookback,
                    'distance': distance,
                    'distance_pct': distance * 100,
                    'long_candidates': long_total,
                    'long_reversion_rate': long_reversion_rate,
                    'long_avg_pnl': long_avg_pnl,
                    'long_win_rate': long_win_rate,
                    'short_candidates': short_total,
                    'short_reversion_rate': short_reversion_rate,
                    'short_avg_pnl': short_avg_pnl,
                    'short_win_rate': short_win_rate
                })
                
                print(f"  Distance {distance*100:.1f}%:")
                print(f"    LONG: {long_total} candidates, {long_reversion_rate*100:.1f}% reversion, {long_avg_pnl*100:.3f}% avg P&L, {long_win_rate*100:.1f}% profitable")
                print(f"    SHORT: {short_total} candidates, {short_reversion_rate*100:.1f}% reversion, {short_avg_pnl*100:.3f}% avg P&L, {short_win_rate*100:.1f}% profitable")
        
        return pd.DataFrame(reversion_data)
    
    def find_optimal_parameters(self, reversion_df):
        """Find optimal parameters for profitable trading"""
        print(f"\n" + "="*60)
        print("OPTIMAL PARAMETER ANALYSIS")
        print("="*60)
        
        # Filter for profitable setups
        profitable_setups = []
        
        for _, row in reversion_df.iterrows():
            # LONG setups
            if row['long_win_rate'] > 0.6 and row['long_avg_pnl'] > self.trading_cost * 2:
                profitable_setups.append({
                    'direction': 'LONG',
                    'lookback': row['lookback'],
                    'distance': row['distance'],
                    'distance_pct': row['distance_pct'],
                    'candidates': row['long_candidates'],
                    'reversion_rate': row['long_reversion_rate'],
                    'avg_pnl': row['long_avg_pnl'],
                    'win_rate': row['long_win_rate'],
                    'profit_after_costs': row['long_avg_pnl'] - self.trading_cost * 2
                })
            
            # SHORT setups
            if row['short_win_rate'] > 0.6 and row['short_avg_pnl'] > self.trading_cost * 2:
                profitable_setups.append({
                    'direction': 'SHORT',
                    'lookback': row['lookback'],
                    'distance': row['distance'],
                    'distance_pct': row['distance_pct'],
                    'candidates': row['short_candidates'],
                    'reversion_rate': row['short_reversion_rate'],
                    'avg_pnl': row['short_avg_pnl'],
                    'win_rate': row['short_win_rate'],
                    'profit_after_costs': row['short_avg_pnl'] - self.trading_cost * 2
                })
        
        if profitable_setups:
            profitable_df = pd.DataFrame(profitable_setups)
            profitable_df = profitable_df.sort_values('profit_after_costs', ascending=False)
            
            print(f"\nTop 10 Most Profitable Setups (After Trading Costs):")
            print(profitable_df.head(10).to_string(index=False))
            
            # Recommend best parameters
            best_setup = profitable_df.iloc[0]
            print(f"\nRECOMMENDED PARAMETERS:")
            print(f"Direction: {best_setup['direction']}")
            print(f"Minimum Distance: {best_setup['distance_pct']:.2f}%")
            print(f"Hold Time: {best_setup['lookback']} candles")
            print(f"Expected P&L after costs: {best_setup['profit_after_costs']*100:.3f}%")
            print(f"Win Rate: {best_setup['win_rate']*100:.1f}%")
            print(f"Reversion Rate: {best_setup['reversion_rate']*100:.1f}%")
            
            return best_setup
        else:
            print("No profitable setups found with current parameters.")
            return None
    
    def create_visualizations(self):
        """Create visualizations for SMA distance analysis"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distance from SMA5 distribution
        axes[0,0].hist(self.df['abs_distance_sma5'], bins=50, alpha=0.7, color='blue')
        axes[0,0].axvline(self.trading_cost * 2, color='red', linestyle='--', 
                         label=f'Min Profitable ({self.trading_cost*2*100:.3f}%)')
        axes[0,0].set_title('Distance from SMA5 Distribution')
        axes[0,0].set_xlabel('Distance from SMA5')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # 2. Distance over time
        sample_data = self.df.iloc[::100]  # Sample every 100th point for clarity
        axes[0,1].plot(sample_data.index, sample_data['abs_distance_sma5'], alpha=0.7)
        axes[0,1].axhline(self.trading_cost * 2, color='red', linestyle='--', 
                         label=f'Min Profitable ({self.trading_cost*2*100:.3f}%)')
        axes[0,1].set_title('Distance from SMA5 Over Time')
        axes[0,1].set_xlabel('Time Index')
        axes[0,1].set_ylabel('Distance from SMA5')
        axes[0,1].legend()
        
        # 3. Price vs SMA5
        sample_data = self.df.iloc[::50]  # Sample every 50th point
        axes[1,0].plot(sample_data.index, sample_data['close'], label='Price', alpha=0.7)
        axes[1,0].plot(sample_data.index, sample_data['SMA5'], label='SMA5', alpha=0.7)
        axes[1,0].set_title('Price vs SMA5')
        axes[1,0].set_xlabel('Time Index')
        axes[1,0].set_ylabel('Price')
        axes[1,0].legend()
        
        # 4. Distance vs Reversion Rate (placeholder for now)
        axes[1,1].text(0.5, 0.5, 'Reversion Analysis\n(see console output)', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Reversion Rate vs Distance')
        
        plt.tight_layout()
        plt.savefig('sma_reversion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to: sma_reversion_analysis.png")

def main():
    """Main execution function"""
    print("="*70)
    print("SMA REVERSION ANALYSIS FOR GOLD TRADING")
    print("="*70)
    print("Analyzing maximum distance from SMA and reversion behavior")
    print("Optimizing for real trading costs ($0.14 per $34,000 contract)")
    print("="*70)
    
    # Initialize analysis
    analysis = SMAReversionAnalysis('stock_data/XAUUSDm_M5_1year_data.csv')
    
    # Load and prepare data
    analysis.load_and_prepare_data()
    
    # Analyze SMA distances
    min_profitable_distance = analysis.analyze_sma_distances()
    
    # Analyze reversion behavior
    reversion_df = analysis.analyze_reversion_behavior()
    
    # Find optimal parameters
    best_setup = analysis.find_optimal_parameters(reversion_df)
    
    # Create visualizations
    analysis.create_visualizations()
    
    # Save detailed results
    reversion_df.to_csv('sma_reversion_analysis.csv', index=False)
    
    print(f"\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("Files generated:")
    print("  - sma_reversion_analysis.csv (detailed reversion data)")
    print("  - sma_reversion_analysis.png (visualizations)")

if __name__ == "__main__":
    main() 