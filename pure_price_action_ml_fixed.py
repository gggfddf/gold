#!/usr/bin/env python3
"""
PURE PRICE ACTION SMA MEAN REVERSION MACHINE LEARNING PIPELINE - FIXED VERSION
Analyzes candlestick structure and SMA crossovers for mean reversion prediction
NO INDICATORS - ONLY PRICE ACTION AND CANDLESTICK BEHAVIOR

Author: AI Assistant
Dataset: XAUUSDm 5-minute data from MT5
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

class PurePriceActionAnalyzer:
    def __init__(self, data_path, lookahead_candles=30):
        """
        Initialize the Pure Price Action Analyzer
        
        Args:
            data_path: Path to the CSV file with OHLC data
            lookahead_candles: Number of candles to look ahead for reversion
        """
        self.data_path = data_path
        self.lookahead_candles = lookahead_candles
        self.df = None
        self.events_df = None
        self.sma_periods = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Reduced for faster processing
        
    def load_and_prepare_data(self):
        """Load data and calculate multiple SMAs"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # Calculate multiple SMAs
        for sma_period in self.sma_periods:
            self.df[f'SMA{sma_period}'] = self.df['close'].rolling(window=sma_period).mean()
        
        # Calculate candlestick structure features
        self.calculate_candlestick_features()
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        print(f"Calculated SMAs: {self.sma_periods}")
        return self.df
    
    def calculate_candlestick_features(self):
        """Calculate pure candlestick structure features"""
        print("Calculating candlestick structure features...")
        
        # Basic candlestick features
        self.df['candle_body'] = abs(self.df['close'] - self.df['open'])
        self.df['candle_range'] = self.df['high'] - self.df['low']
        self.df['candle_color'] = (self.df['close'] > self.df['open']).astype(int)  # 1=bullish, 0=bearish
        
        # Wick calculations
        self.df['upper_wick'] = np.where(
            self.df['candle_color'] == 1,  # Bullish candle
            self.df['high'] - self.df['close'],
            self.df['high'] - self.df['open']
        )
        
        self.df['lower_wick'] = np.where(
            self.df['candle_color'] == 1,  # Bullish candle
            self.df['open'] - self.df['low'],
            self.df['close'] - self.df['low']
        )
        
        # Wick-to-body ratios
        self.df['upper_wick_ratio'] = np.where(
            self.df['candle_body'] > 0,
            self.df['upper_wick'] / self.df['candle_body'],
            0
        )
        
        self.df['lower_wick_ratio'] = np.where(
            self.df['candle_body'] > 0,
            self.df['lower_wick'] / self.df['candle_body'],
            0
        )
        
        # Body size relative to full range
        self.df['body_to_range_ratio'] = np.where(
            self.df['candle_range'] > 0,
            self.df['candle_body'] / self.df['candle_range'],
            0
        )
        
        # Close position within candle
        self.df['close_position'] = np.where(
            self.df['candle_range'] > 0,
            (self.df['close'] - self.df['low']) / self.df['candle_range'],
            0.5
        )
        
        # Gap from prior close
        self.df['gap_from_prior'] = self.df['open'] - self.df['close'].shift(1)
        self.df['gap_pct'] = np.where(
            self.df['close'].shift(1) > 0,
            self.df['gap_from_prior'] / self.df['close'].shift(1) * 100,
            0
        )
        
        # Calculate candlestick patterns
        self.calculate_candlestick_patterns()
        
        # Calculate recent price behavior (last 5 candles)
        self.calculate_recent_price_behavior()
    
    def calculate_candlestick_patterns(self):
        """Identify candlestick patterns"""
        print("Identifying candlestick patterns...")
        
        # Doji pattern (very small body)
        self.df['is_doji'] = (self.df['body_to_range_ratio'] < 0.1).astype(int)
        
        # Hammer pattern (long lower wick, small body)
        self.df['is_hammer'] = (
            (self.df['lower_wick_ratio'] > 2) & 
            (self.df['body_to_range_ratio'] < 0.3) &
            (self.df['candle_color'] == 1)
        ).astype(int)
        
        # Shooting star pattern (long upper wick, small body)
        self.df['is_shooting_star'] = (
            (self.df['upper_wick_ratio'] > 2) & 
            (self.df['body_to_range_ratio'] < 0.3) &
            (self.df['candle_color'] == 0)
        ).astype(int)
        
        # Indecision candles (high wicks, small body)
        self.df['is_indecision'] = (
            (self.df['body_to_range_ratio'] < 0.2) &
            ((self.df['upper_wick_ratio'] > 1) | (self.df['lower_wick_ratio'] > 1))
        ).astype(int)
    
    def calculate_recent_price_behavior(self):
        """Calculate recent price behavior patterns"""
        print("Calculating recent price behavior...")
        
        # Consecutive bullish/bearish candles
        self.df['consecutive_bullish'] = 0
        self.df['consecutive_bearish'] = 0
        
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['candle_color'] == 1:
                self.df.iloc[i, self.df.columns.get_loc('consecutive_bullish')] = \
                    self.df.iloc[i-1]['consecutive_bullish'] + 1
            else:
                self.df.iloc[i, self.df.columns.get_loc('consecutive_bearish')] = \
                    self.df.iloc[i-1]['consecutive_bearish'] + 1
        
        # Recent volatility (5-period)
        self.df['recent_volatility'] = self.df['close'].rolling(5).std()
        
        # Recent trend strength (5-period)
        self.df['recent_trend'] = self.df['close'].rolling(5).mean() - self.df['close'].rolling(10).mean()
    
    def detect_sma_crossover_events(self):
        """Detect crossover events for all SMAs"""
        print("Detecting SMA crossover events...")
        
        all_events = []
        
        for sma_period in self.sma_periods:
            sma_col = f'SMA{sma_period}'
            print(f"Processing {sma_col}...")
            
            events = self.detect_crossovers_for_sma(sma_period, sma_col)
            all_events.extend(events)
        
        self.events_df = pd.DataFrame(all_events)
        print(f"Total crossover events detected: {len(self.events_df)}")
        return self.events_df
    
    def detect_crossovers_for_sma(self, sma_period, sma_col):
        """Detect crossovers for a specific SMA"""
        events = []
        
        for i in range(1, len(self.df)):
            current_price = self.df.iloc[i]['close']
            current_sma = self.df.iloc[i][sma_col]
            prev_price = self.df.iloc[i-1]['close']
            prev_sma = self.df.iloc[i-1][sma_col]
            
            # Skip if SMA is NaN
            if pd.isna(current_sma) or pd.isna(prev_sma):
                continue
            
            # Detect bullish crossover
            if prev_price <= prev_sma and current_price > current_sma:
                event_type = 'bullish'
                events.append(self.create_event_record(i, sma_period, event_type, current_price, current_sma))
            
            # Detect bearish crossover
            elif prev_price >= prev_sma and current_price < current_sma:
                event_type = 'bearish'
                events.append(self.create_event_record(i, sma_period, event_type, current_price, current_sma))
        
        return events
    
    def create_event_record(self, index, sma_period, event_type, price, sma):
        """Create a comprehensive event record with all candlestick features"""
        row = self.df.iloc[index]
        
        # Distance from SMA (normalized)
        distance_from_sma = (price - sma) / sma * 100
        
        return {
            'index': index,
            'time': row['time'],
            'sma_period': sma_period,
            'event_type': event_type,
            'price_at_crossover': price,
            'sma_at_crossover': sma,
            'distance_from_sma_pct': distance_from_sma,
            
            # Candlestick structure
            'candle_body': row['candle_body'],
            'candle_range': row['candle_range'],
            'candle_color': row['candle_color'],
            'upper_wick_ratio': row['upper_wick_ratio'],
            'lower_wick_ratio': row['lower_wick_ratio'],
            'body_to_range_ratio': row['body_to_range_ratio'],
            'close_position': row['close_position'],
            'gap_pct': row['gap_pct'],
            
            # Candlestick patterns
            'is_doji': row['is_doji'],
            'is_hammer': row['is_hammer'],
            'is_shooting_star': row['is_shooting_star'],
            'is_indecision': row['is_indecision'],
            
            # Recent behavior
            'consecutive_bullish': row['consecutive_bullish'],
            'consecutive_bearish': row['consecutive_bearish'],
            'recent_volatility': row['recent_volatility'],
            'recent_trend': row['recent_trend']
        }
    
    def analyze_reversion_events(self):
        """Analyze each crossover event for reversion"""
        print("Analyzing reversion events...")
        
        reversion_data = []
        
        for _, event in self.events_df.iterrows():
            event_index = event['index']
            event_type = event['event_type']
            sma_period = event['sma_period']
            
            # Look ahead for reversion
            reverted = False
            candles_to_reversion = None
            max_adverse_move = 0
            max_favorable_move = 0
            
            # Check next N candles for reversion
            for lookahead in range(1, min(self.lookahead_candles + 1, len(self.df) - event_index)):
                future_index = event_index + lookahead
                if future_index >= len(self.df):
                    break
                
                future_price = self.df.iloc[future_index]['close']
                future_sma = self.df.iloc[future_index][f'SMA{sma_period}']
                
                # Skip if SMA is NaN
                if pd.isna(future_sma):
                    continue
                
                # Calculate movement from crossover point
                price_change = future_price - event['price_at_crossover']
                price_change_pct = (price_change / event['price_at_crossover']) * 100
                
                # Track max movements
                if event_type == 'bullish':
                    if price_change > max_favorable_move:
                        max_favorable_move = price_change
                    if price_change < max_adverse_move:
                        max_adverse_move = price_change
                    
                    # Check for bearish reversion (price crosses back below SMA)
                    if future_price <= future_sma:
                        reverted = True
                        candles_to_reversion = lookahead
                        break
                        
                elif event_type == 'bearish':
                    if price_change < max_favorable_move:
                        max_favorable_move = price_change
                    if price_change > max_adverse_move:
                        max_adverse_move = price_change
                    
                    # Check for bullish reversion (price crosses back above SMA)
                    if future_price >= future_sma:
                        reverted = True
                        candles_to_reversion = lookahead
                        break
            
            # Create reversion record
            reversion_record = {
                'index': event_index,
                'time': event['time'],
                'sma_period': sma_period,
                'event_type': event_type,
                'price_at_crossover': event['price_at_crossover'],
                'sma_at_crossover': event['sma_at_crossover'],
                'distance_from_sma_pct': event['distance_from_sma_pct'],
                'candle_body': event['candle_body'],
                'candle_range': event['candle_range'],
                'candle_color': event['candle_color'],
                'upper_wick_ratio': event['upper_wick_ratio'],
                'lower_wick_ratio': event['lower_wick_ratio'],
                'body_to_range_ratio': event['body_to_range_ratio'],
                'close_position': event['close_position'],
                'gap_pct': event['gap_pct'],
                'is_doji': event['is_doji'],
                'is_hammer': event['is_hammer'],
                'is_shooting_star': event['is_shooting_star'],
                'is_indecision': event['is_indecision'],
                'consecutive_bullish': event['consecutive_bullish'],
                'consecutive_bearish': event['consecutive_bearish'],
                'recent_volatility': event['recent_volatility'],
                'recent_trend': event['recent_trend'],
                'reverted': reverted,
                'candles_to_reversion': candles_to_reversion,
                'max_adverse_move': max_adverse_move,
                'max_favorable_move': max_favorable_move,
                'max_adverse_move_pct': (max_adverse_move / event['price_at_crossover']) * 100,
                'max_favorable_move_pct': (max_favorable_move / event['price_at_crossover']) * 100
            }
            
            reversion_data.append(reversion_record)
        
        self.reversion_df = pd.DataFrame(reversion_data)
        
        # Calculate summary statistics
        self.print_reversion_summary()
        
        return self.reversion_df
    
    def print_reversion_summary(self):
        """Print comprehensive reversion summary"""
        print("\n" + "="*60)
        print("PURE PRICE ACTION REVERSION ANALYSIS SUMMARY")
        print("="*60)
        
        total_events = len(self.reversion_df)
        reversion_rate = self.reversion_df['reverted'].mean() * 100
        avg_candles_to_reversion = self.reversion_df[self.reversion_df['reverted']]['candles_to_reversion'].mean()
        
        print(f"Total Events: {total_events:,}")
        print(f"Overall Reversion Rate: {reversion_rate:.1f}%")
        print(f"Average Candles to Reversion: {avg_candles_to_reversion:.1f}")
        
        # By SMA period
        print("\nReversion Rate by SMA Period:")
        sma_summary = self.reversion_df.groupby('sma_period').agg({
            'reverted': ['count', 'mean'],
            'candles_to_reversion': 'mean'
        }).round(3)
        
        sma_summary.columns = ['Event_Count', 'Reversion_Rate', 'Avg_Candles']
        sma_summary['Reversion_Rate'] = sma_summary['Reversion_Rate'] * 100
        print(sma_summary)
        
        # By event type
        print("\nReversion Rate by Event Type:")
        type_summary = self.reversion_df.groupby('event_type').agg({
            'reverted': ['count', 'mean'],
            'candles_to_reversion': 'mean'
        }).round(3)
        
        type_summary.columns = ['Event_Count', 'Reversion_Rate', 'Avg_Candles']
        type_summary['Reversion_Rate'] = type_summary['Reversion_Rate'] * 100
        print(type_summary)
    
    def save_results(self):
        """Save analysis results"""
        print("\nSaving results...")
        
        # Save reversion analysis
        self.reversion_df.to_csv('pure_price_action_reversion_analysis.csv', index=False)
        
        # Save summary statistics
        summary_stats = {
            'total_events': len(self.reversion_df),
            'reversion_rate': self.reversion_df['reverted'].mean() * 100,
            'avg_candles_to_reversion': self.reversion_df[self.reversion_df['reverted']]['candles_to_reversion'].mean(),
            'avg_adverse_move': self.reversion_df['max_adverse_move_pct'].mean(),
            'avg_favorable_move': self.reversion_df['max_favorable_move_pct'].mean()
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv('pure_price_action_summary.csv', index=False)
        
        print("Results saved to:")
        print("  - pure_price_action_reversion_analysis.csv")
        print("  - pure_price_action_summary.csv")

def main():
    """Main execution function"""
    print("="*70)
    print("PURE PRICE ACTION SMA MEAN REVERSION ANALYSIS - FIXED VERSION")
    print("="*70)
    print("NO INDICATORS - ONLY CANDLESTICK STRUCTURE AND PRICE BEHAVIOR")
    print("="*70)
    
    # Initialize analyzer
    analyzer = PurePriceActionAnalyzer('stock_data/XAUUSDm_M5_1year_data.csv', lookahead_candles=30)
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Detect crossover events
    analyzer.detect_sma_crossover_events()
    
    # Analyze reversion events
    analyzer.analyze_reversion_events()
    
    # Save results
    analyzer.save_results()
    
    print("\n" + "="*70)
    print("PURE PRICE ACTION ANALYSIS COMPLETE!")
    print("="*70)
    print("Check the generated files for detailed results:")
    print("  - pure_price_action_reversion_analysis.csv")
    print("  - pure_price_action_summary.csv")

if __name__ == "__main__":
    main() 