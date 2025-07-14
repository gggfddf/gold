#!/usr/bin/env python3
"""
ML-DRIVEN MEAN REVERSION SCALPER STRATEGY
Complete backtesting engine for XAUUSD 5-minute mean reversion trading
Pure price action with machine learning probability filtering

Author: AI Assistant
Strategy: ML-Driven Mean Reversion Scalper (Gold 5M)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

class MLMeanReversionStrategy:
    def __init__(self, data_path, prob_cutoff=0.95, sma_distance=0.0025, max_hold_candles=7):
        """
        Initialize the ML Mean Reversion Strategy
        
        Args:
            data_path: Path to the CSV file with OHLC data
            prob_cutoff: ML probability threshold (0.90-0.99)
            sma_distance: Minimum distance from SMA (0.25%-0.50%)
            max_hold_candles: Maximum candles to hold position
        """
        self.data_path = data_path
        self.prob_cutoff = prob_cutoff
        self.sma_distance = sma_distance
        self.max_hold_candles = max_hold_candles
        self.df = None
        self.model = None
        self.scaler = None
        self.trades = []
        
        # Strategy parameters
        self.bullish_patterns = ["hanging_man", "inverted_hammer", "bullish_engulfing", "morning_star"]
        self.bearish_patterns = ["shooting_star", "bearish_engulfing", "evening_star"]
        self.stop_loss_pct = 0.0035  # 0.35%
        
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
        
        # Calculate candlestick features
        self.calculate_candlestick_features()
        
        # Detect patterns
        self.detect_patterns()
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        return self.df
    
    def calculate_candlestick_features(self):
        """Calculate candlestick structure features"""
        print("Calculating candlestick features...")
        
        # Basic features
        self.df['candle_body'] = abs(self.df['close'] - self.df['open'])
        self.df['candle_range'] = self.df['high'] - self.df['low']
        self.df['candle_color'] = (self.df['close'] > self.df['open']).astype(int)
        
        # Wick calculations
        self.df['upper_wick'] = np.where(
            self.df['candle_color'] == 1,
            self.df['high'] - self.df['close'],
            self.df['high'] - self.df['open']
        )
        
        self.df['lower_wick'] = np.where(
            self.df['candle_color'] == 1,
            self.df['open'] - self.df['low'],
            self.df['close'] - self.df['low']
        )
        
        # Ratios
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
        
        self.df['body_ratio'] = np.where(
            self.df['candle_range'] > 0,
            self.df['candle_body'] / self.df['candle_range'],
            0
        )
        
        # Distance from SMAs
        self.df['distance_from_sma5'] = abs(self.df['close'] - self.df['SMA5']) / self.df['close']
        self.df['distance_from_sma10'] = abs(self.df['close'] - self.df['SMA10']) / self.df['close']
        
        # Previous candle features
        self.df['prev_candle_color'] = self.df['candle_color'].shift(1)
        self.df['prev_candle_body'] = self.df['candle_body'].shift(1)
        
        # Recent behavior
        self.df['recent_volatility'] = self.df['close'].rolling(5).std()
        self.df['recent_trend'] = self.df['close'].rolling(5).mean() - self.df['close'].rolling(10).mean()
        
        # Consecutive candles
        self.df['consecutive_bullish'] = 0
        self.df['consecutive_bearish'] = 0
        
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['candle_color'] == 1:
                self.df.iloc[i, self.df.columns.get_loc('consecutive_bullish')] = \
                    self.df.iloc[i-1]['consecutive_bullish'] + 1
            else:
                self.df.iloc[i, self.df.columns.get_loc('consecutive_bearish')] = \
                    self.df.iloc[i-1]['consecutive_bearish'] + 1
    
    def detect_patterns(self):
        """Detect candlestick patterns"""
        print("Detecting candlestick patterns...")
        
        # Initialize pattern columns
        pattern_columns = [
            'hanging_man', 'inverted_hammer', 'bullish_engulfing', 'morning_star',
            'shooting_star', 'bearish_engulfing', 'evening_star'
        ]
        
        for pattern in pattern_columns:
            self.df[pattern] = 0
        
        # Hanging Man (long lower wick, small body, bearish)
        self.df['hanging_man'] = (
            (self.df['lower_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 0)
        ).astype(int)
        
        # Inverted Hammer (long upper wick, small body, bullish)
        self.df['inverted_hammer'] = (
            (self.df['upper_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 1)
        ).astype(int)
        
        # Bullish Engulfing
        self.df['bullish_engulfing'] = (
            (self.df['candle_color'] == 1) &
            (self.df['prev_candle_color'] == 0) &
            (self.df['open'] < self.df['close'].shift(1)) &
            (self.df['close'] > self.df['open'].shift(1))
        ).astype(int)
        
        # Bearish Engulfing
        self.df['bearish_engulfing'] = (
            (self.df['candle_color'] == 0) &
            (self.df['prev_candle_color'] == 1) &
            (self.df['open'] > self.df['close'].shift(1)) &
            (self.df['close'] < self.df['open'].shift(1))
        ).astype(int)
        
        # Shooting Star (long upper wick, small body, bearish)
        self.df['shooting_star'] = (
            (self.df['upper_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 0)
        ).astype(int)
        
        # Morning Star (bearish, doji, bullish)
        self.df['morning_star'] = (
            (self.df['candle_color'] == 1) &
            (self.df['prev_candle_color'] == 0) &
            (self.df['body_ratio'].shift(1) < 0.1) &
            (self.df['candle_color'].shift(2) == 0)
        ).astype(int)
        
        # Evening Star (bullish, doji, bearish)
        self.df['evening_star'] = (
            (self.df['candle_color'] == 0) &
            (self.df['prev_candle_color'] == 1) &
            (self.df['body_ratio'].shift(1) < 0.1) &
            (self.df['candle_color'].shift(2) == 1)
        ).astype(int)
        
        # Create pattern name column
        self.df['pattern_name'] = 'none'
        
        for pattern in pattern_columns:
            mask = self.df[pattern] == 1
            self.df.loc[mask, 'pattern_name'] = pattern
        
        pattern_counts = self.df['pattern_name'].value_counts()
        print(f"Pattern frequencies:\n{pattern_counts}")
    
    def prepare_ml_training_data(self):
        """Prepare data for ML model training"""
        print("Preparing ML training data...")
        
        # Create training dataset from historical crossovers
        training_data = []
        
        for i in range(1, len(self.df)):
            # Check for SMA crossovers
            current_price = self.df.iloc[i]['close']
            current_sma5 = self.df.iloc[i]['SMA5']
            current_sma10 = self.df.iloc[i]['SMA10']
            prev_price = self.df.iloc[i-1]['close']
            prev_sma5 = self.df.iloc[i-1]['SMA5']
            prev_sma10 = self.df.iloc[i-1]['SMA10']
            
            # Skip if SMAs are NaN
            if pd.isna(current_sma5) or pd.isna(prev_sma5):
                continue
            
            # Detect crossovers
            if prev_price <= prev_sma5 and current_price > current_sma5:
                # Bullish crossover
                reverted = self.check_reversion(i, 'SMA5', 'bullish')
                training_data.append(self.create_ml_sample(i, 'bullish', 'SMA5', reverted))
            elif prev_price >= prev_sma5 and current_price < current_sma5:
                # Bearish crossover
                reverted = self.check_reversion(i, 'SMA5', 'bearish')
                training_data.append(self.create_ml_sample(i, 'bearish', 'SMA5', reverted))
            
            if pd.isna(current_sma10) or pd.isna(prev_sma10):
                continue
                
            if prev_price <= prev_sma10 and current_price > current_sma10:
                reverted = self.check_reversion(i, 'SMA10', 'bullish')
                training_data.append(self.create_ml_sample(i, 'bullish', 'SMA10', reverted))
            elif prev_price >= prev_sma10 and current_price < current_sma10:
                reverted = self.check_reversion(i, 'SMA10', 'bearish')
                training_data.append(self.create_ml_sample(i, 'bearish', 'SMA10', reverted))
        
        return pd.DataFrame(training_data)
    
    def check_reversion(self, index, sma_col, direction):
        """Check if price reverted to SMA within lookahead period"""
        lookahead = 10  # Check next 10 candles
        
        for i in range(1, min(lookahead + 1, len(self.df) - index)):
            future_index = index + i
            if future_index >= len(self.df):
                break
            
            future_price = self.df.iloc[future_index]['close']
            future_sma = self.df.iloc[future_index][sma_col]
            
            if pd.isna(future_sma):
                continue
            
            if direction == 'bullish' and future_price <= future_sma:
                return True
            elif direction == 'bearish' and future_price >= future_sma:
                return True
        
        return False
    
    def create_ml_sample(self, index, direction, sma_col, reverted):
        """Create ML training sample"""
        row = self.df.iloc[index]
        
        return {
            'direction': direction,
            'sma_col': sma_col,
            'candle_body': row['candle_body'],
            'candle_range': row['candle_range'],
            'candle_color': row['candle_color'],
            'upper_wick_ratio': row['upper_wick_ratio'],
            'lower_wick_ratio': row['lower_wick_ratio'],
            'body_ratio': row['body_ratio'],
            'distance_from_sma5': row['distance_from_sma5'],
            'distance_from_sma10': row['distance_from_sma10'],
            'prev_candle_color': row['prev_candle_color'],
            'prev_candle_body': row['prev_candle_body'],
            'consecutive_bullish': row['consecutive_bullish'],
            'consecutive_bearish': row['consecutive_bearish'],
            'recent_volatility': row['recent_volatility'],
            'recent_trend': row['recent_trend'],
            'hanging_man': row['hanging_man'],
            'inverted_hammer': row['inverted_hammer'],
            'bullish_engulfing': row['bullish_engulfing'],
            'morning_star': row['morning_star'],
            'shooting_star': row['shooting_star'],
            'bearish_engulfing': row['bearish_engulfing'],
            'evening_star': row['evening_star'],
            'reverted': reverted
        }
    
    def train_ml_model(self):
        """Train the ML model for reversion prediction"""
        print("Training ML model...")
        
        # Prepare training data
        training_df = self.prepare_ml_training_data()
        
        if len(training_df) == 0:
            print("No training data available!")
            return False
        
        # Prepare features
        feature_columns = [
            'candle_body', 'candle_range', 'candle_color',
            'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            'distance_from_sma5', 'distance_from_sma10',
            'prev_candle_color', 'prev_candle_body',
            'consecutive_bullish', 'consecutive_bearish',
            'recent_volatility', 'recent_trend',
            'hanging_man', 'inverted_hammer', 'bullish_engulfing', 'morning_star',
            'shooting_star', 'bearish_engulfing', 'evening_star'
        ]
        
        # Create direction dummy
        training_df['direction_bullish'] = (training_df['direction'] == 'bullish').astype(int)
        feature_columns.append('direction_bullish')
        
        X = training_df[feature_columns].copy()
        y = training_df['reverted'].astype(int)
        
        # Remove NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) == 0:
            print("No valid training samples after cleaning!")
            return False
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = XGBClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print(f"\nML Model Performance:")
        print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return True
    
    def check_entry_conditions(self, index):
        """Check if entry conditions are met"""
        row = self.df.iloc[index]
        
        # Check SMA distance
        distance_sma5 = abs(row['close'] - row['SMA5']) / row['close']
        distance_sma10 = abs(row['close'] - row['SMA10']) / row['close']
        
        # Determine which SMA to use
        if distance_sma5 >= self.sma_distance:
            sma_col = 'SMA5'
            sma_value = row['SMA5']
            distance = distance_sma5
        elif distance_sma10 >= self.sma_distance:
            sma_col = 'SMA10'
            sma_value = row['SMA10']
            distance = distance_sma10
        else:
            return None, None, None
        
        # Check pattern conditions
        pattern_name = row['pattern_name']
        
        # Check for bullish setup
        if row['close'] < sma_value:  # Price below SMA
            if pattern_name in self.bullish_patterns:
                # Check wick and body conditions
                if (row['lower_wick_ratio'] >= 0.6 and row['body_ratio'] <= 0.35):
                    direction = 'LONG'
                    return direction, sma_col, sma_value
        
        # Check for bearish setup
        elif row['close'] > sma_value:  # Price above SMA
            if pattern_name in self.bearish_patterns:
                # Check wick and body conditions
                if (row['upper_wick_ratio'] >= 0.6 and row['body_ratio'] <= 0.35):
                    direction = 'SHORT'
                    return direction, sma_col, sma_value
        
        return None, None, None
    
    def get_ml_probability(self, index):
        """Get ML probability for reversion"""
        if self.model is None:
            return 0.0
        
        row = self.df.iloc[index]
        
        # Create feature vector
        features = {
            'candle_body': row['candle_body'],
            'candle_range': row['candle_range'],
            'candle_color': row['candle_color'],
            'upper_wick_ratio': row['upper_wick_ratio'],
            'lower_wick_ratio': row['lower_wick_ratio'],
            'body_ratio': row['body_ratio'],
            'distance_from_sma5': row['distance_from_sma5'],
            'distance_from_sma10': row['distance_from_sma10'],
            'prev_candle_color': row['prev_candle_color'],
            'prev_candle_body': row['prev_candle_body'],
            'consecutive_bullish': row['consecutive_bullish'],
            'consecutive_bearish': row['consecutive_bearish'],
            'recent_volatility': row['recent_volatility'],
            'recent_trend': row['recent_trend'],
            'hanging_man': row['hanging_man'],
            'inverted_hammer': row['inverted_hammer'],
            'bullish_engulfing': row['bullish_engulfing'],
            'morning_star': row['morning_star'],
            'shooting_star': row['shooting_star'],
            'bearish_engulfing': row['bearish_engulfing'],
            'evening_star': row['evening_star'],
            'direction_bullish': 1  # Assume bullish for probability
        }
        
        feature_columns = list(features.keys())
        X = pd.DataFrame([features])[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probability
        prob = self.model.predict_proba(X_scaled)[0, 1]
        return prob
    
    def backtest_strategy(self):
        """Run the complete backtest"""
        print("Running strategy backtest...")
        
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
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'direction': position['direction'],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'hold_candles': i - position['entry_index'],
                        'pattern': position.get('pattern', 'unknown')
                    }
                    self.trades.append(trade)
                    position = None
            
            # Check for new entry if not in position
            if position is None:
                direction, sma_col, sma_value = self.check_entry_conditions(i)
                
                if direction is not None:
                    # Get ML probability
                    ml_prob = self.get_ml_probability(i)
                    
                    # Check ML probability threshold
                    if ml_prob >= self.prob_cutoff:
                        # Enter position
                        position = {
                            'entry_time': current_time,
                            'entry_index': i,
                            'direction': direction,
                            'entry_price': current_price,
                            'sma_col': sma_col,
                            'sma_value': sma_value,
                            'ml_probability': ml_prob,
                            'pattern': self.df.iloc[i]['pattern_name']
                        }
        
        # Close any remaining position at end
        if position is not None:
            exit_price = self.df.iloc[-1]['close']
            pnl = self.calculate_pnl(position, exit_price)
            
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': self.df.iloc[-1]['time'],
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'exit_reason': 'End of data',
                'hold_candles': len(self.df) - 1 - position['entry_index'],
                'pattern': position.get('pattern', 'unknown')
            }
            self.trades.append(trade)
        
        return self.trades
    
    def check_exit_conditions(self, index, position):
        """Check if exit conditions are met"""
        current_price = self.df.iloc[index]['close']
        entry_price = position['entry_price']
        sma_value = self.df.iloc[index][position['sma_col']]
        
        # Take Profit: Price returns to SMA
        if position['direction'] == 'LONG' and current_price >= sma_value:
            return 'Take Profit'
        elif position['direction'] == 'SHORT' and current_price <= sma_value:
            return 'Take Profit'
        
        # Stop Loss: Price moves against by 0.35%
        if position['direction'] == 'LONG' and current_price <= entry_price * (1 - self.stop_loss_pct):
            return 'Stop Loss'
        elif position['direction'] == 'SHORT' and current_price >= entry_price * (1 + self.stop_loss_pct):
            return 'Stop Loss'
        
        # Timeout: Held for more than max_hold_candles
        if index - position['entry_index'] >= self.max_hold_candles:
            return 'Timeout'
        
        return None
    
    def calculate_pnl(self, position, exit_price):
        """Calculate P&L for a trade"""
        entry_price = position['entry_price']
        
        if position['direction'] == 'LONG':
            return (exit_price - entry_price) / entry_price
        else:  # SHORT
            return (entry_price - exit_price) / entry_price
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.trades:
            print("No trades executed!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        print("\n" + "="*60)
        print("STRATEGY BACKTEST RESULTS")
        print("="*60)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades * 100
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        max_profit = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total P&L: {total_pnl:.4f} ({total_pnl*100:.2f}%)")
        print(f"Average P&L: {avg_pnl:.4f} ({avg_pnl*100:.2f}%)")
        print(f"Max Profit: {max_profit:.4f} ({max_profit*100:.2f}%)")
        print(f"Max Loss: {max_loss:.4f} ({max_loss*100:.2f}%)")
        
        # Exit reason analysis
        print(f"\nExit Reasons:")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Direction analysis
        print(f"\nDirection Performance:")
        direction_perf = trades_df.groupby('direction').agg({
            'pnl': ['count', 'mean', 'sum']
        }).round(4)
        print(direction_perf)
        
        # Pattern analysis
        print(f"\nPattern Performance:")
        if 'pattern' in trades_df.columns:
            pattern_perf = trades_df.groupby('pattern').agg({
                'pnl': ['count', 'mean', 'sum']
            }).round(4)
            print(pattern_perf)
        else:
            print("Pattern data not available in trades")
        
        # Save results
        trades_df.to_csv('strategy_backtest_results.csv', index=False)
        print(f"\nResults saved to: strategy_backtest_results.csv")
        
        # Create visualizations
        self.create_strategy_visualizations(trades_df)
    
    def create_strategy_visualizations(self, trades_df):
        """Create strategy performance visualizations"""
        print("Creating strategy visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cumulative P&L
        cumulative_pnl = trades_df['pnl'].cumsum()
        axes[0,0].plot(cumulative_pnl.index, cumulative_pnl.values, linewidth=2, color='blue')
        axes[0,0].set_title('Cumulative P&L')
        axes[0,0].set_ylabel('Cumulative P&L')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. P&L Distribution
        axes[0,1].hist(trades_df['pnl'], bins=20, alpha=0.7, color='green')
        axes[0,1].set_title('P&L Distribution')
        axes[0,1].set_xlabel('P&L')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Exit Reasons
        exit_reasons = trades_df['exit_reason'].value_counts()
        exit_reasons.plot(kind='bar', ax=axes[1,0], color='orange')
        axes[1,0].set_title('Exit Reasons')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Direction Performance
        direction_pnl = trades_df.groupby('direction')['pnl'].mean()
        direction_pnl.plot(kind='bar', ax=axes[1,1], color=['red', 'green'])
        axes[1,1].set_title('Average P&L by Direction')
        axes[1,1].set_ylabel('Average P&L')
        
        plt.tight_layout()
        plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Strategy visualizations saved to: strategy_performance.png")

def main():
    """Main execution function"""
    print("="*70)
    print("ML-DRIVEN MEAN REVERSION SCALPER STRATEGY")
    print("="*70)
    print("Pure Price Action with Machine Learning Filtering")
    print("="*70)
    
    # Initialize strategy
    strategy = MLMeanReversionStrategy(
        data_path='stock_data/XAUUSDm_M5_1year_data.csv',
        prob_cutoff=0.95,  # 95% ML probability threshold
        sma_distance=0.0025,  # 0.25% minimum distance
        max_hold_candles=7  # 7 candles max hold
    )
    
    # Load and prepare data
    strategy.load_and_prepare_data()
    
    # Train ML model
    if not strategy.train_ml_model():
        print("Failed to train ML model. Exiting.")
        return
    
    # Run backtest
    trades = strategy.backtest_strategy()
    
    # Analyze results
    strategy.analyze_results()
    
    print("\n" + "="*70)
    print("STRATEGY BACKTEST COMPLETE!")
    print("="*70)
    print("Check the generated files for detailed results:")
    print("  - strategy_backtest_results.csv")
    print("  - strategy_performance.png")

if __name__ == "__main__":
    main() 