#!/usr/bin/env python3
"""
QUANT-STYLE ML REVERSION STRATEGY
Loosened entry/exit rules for maximum ML training data
Pure SMA5 crossover + reversion detection
Quant-grade machine learning approach

Author: AI Assistant
Strategy: Quant ML Reversion (Gold 5M)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap

class QuantMLReversionStrategy:
    def __init__(self, data_path, prob_cutoff=0.70, sma_distance=0.001, max_hold_candles=15):
        """
        Initialize the Quant ML Reversion Strategy
        
        Args:
            data_path: Path to the CSV file with OHLC data
            prob_cutoff: ML probability threshold (loosened to 70%)
            sma_distance: Minimum distance from SMA (loosened to 0.1%)
            max_hold_candles: Maximum candles to hold position (increased to 15)
        """
        self.data_path = data_path
        self.prob_cutoff = prob_cutoff
        self.sma_distance = sma_distance
        self.max_hold_candles = max_hold_candles
        self.df = None
        self.model = None
        self.scaler = None
        self.trades = []
        
        # Loosened strategy parameters
        self.stop_loss_pct = 0.005  # 0.5% stop loss (loosened)
        self.take_profit_pct = 0.003  # 0.3% take profit (loosened)
        
    def load_and_prepare_data(self):
        """Load data and calculate technical features"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # Calculate SMAs (focus on SMA5)
        self.df['SMA5'] = self.df['close'].rolling(window=5).mean()
        self.df['SMA10'] = self.df['close'].rolling(window=10).mean()
        self.df['SMA20'] = self.df['close'].rolling(window=20).mean()
        
        # Calculate candlestick features
        self.calculate_candlestick_features()
        
        # Calculate momentum and volatility features
        self.calculate_momentum_features()
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        return self.df
    
    def calculate_candlestick_features(self):
        """Calculate comprehensive candlestick structure features"""
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
        self.df['distance_from_sma20'] = abs(self.df['close'] - self.df['SMA20']) / self.df['close']
        
        # Previous candle features
        self.df['prev_candle_color'] = self.df['candle_color'].shift(1)
        self.df['prev_candle_body'] = self.df['candle_body'].shift(1)
        self.df['prev_candle_range'] = self.df['candle_range'].shift(1)
        
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
        
        # Price momentum
        self.df['price_momentum'] = self.df['close'].pct_change()
        self.df['price_momentum_3'] = self.df['close'].pct_change(3)
        self.df['price_momentum_5'] = self.df['close'].pct_change(5)
        
        # Volume proxy (using range as volume proxy)
        self.df['volume_proxy'] = self.df['candle_range'] / self.df['close']
        self.df['volume_ma'] = self.df['volume_proxy'].rolling(10).mean()
        self.df['volume_ratio'] = self.df['volume_proxy'] / self.df['volume_ma']
    
    def calculate_momentum_features(self):
        """Calculate momentum and technical indicators"""
        print("Calculating momentum features...")
        
        # RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = self.df['close'].ewm(span=12).mean()
        exp2 = self.df['close'].ewm(span=26).mean()
        self.df['macd'] = exp1 - exp2
        self.df['macd_signal'] = self.df['macd'].ewm(span=9).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        # Bollinger Bands
        self.df['bb_middle'] = self.df['close'].rolling(20).mean()
        bb_std = self.df['close'].rolling(20).std()
        self.df['bb_upper'] = self.df['bb_middle'] + (bb_std * 2)
        self.df['bb_lower'] = self.df['bb_middle'] - (bb_std * 2)
        self.df['bb_position'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # Stochastic
        low_min = self.df['low'].rolling(14).min()
        high_max = self.df['high'].rolling(14).max()
        self.df['stoch_k'] = 100 * (self.df['close'] - low_min) / (high_max - low_min)
        self.df['stoch_d'] = self.df['stoch_k'].rolling(3).mean()
        
        # ATR (Average True Range)
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        self.df['atr'] = true_range.rolling(14).mean()
        
        # Support/Resistance levels
        self.df['support_level'] = self.df['low'].rolling(20).min()
        self.df['resistance_level'] = self.df['high'].rolling(20).max()
        self.df['support_distance'] = (self.df['close'] - self.df['support_level']) / self.df['close']
        self.df['resistance_distance'] = (self.df['resistance_level'] - self.df['close']) / self.df['close']
    
    def prepare_ml_training_data(self):
        """Prepare comprehensive ML training data from SMA5 crossovers"""
        print("Preparing ML training data...")
        
        # Create training dataset from SMA5 crossovers
        training_data = []
        
        for i in range(1, len(self.df)):
            # Check for SMA5 crossovers
            current_price = self.df.iloc[i]['close']
            current_sma5 = self.df.iloc[i]['SMA5']
            prev_price = self.df.iloc[i-1]['close']
            prev_sma5 = self.df.iloc[i-1]['SMA5']
            
            # Skip if SMA5 is NaN
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
        
        return pd.DataFrame(training_data)
    
    def check_reversion(self, index, sma_col, direction):
        """Check if price reverted to SMA within lookahead period"""
        lookahead = 15  # Increased lookahead for more data
        
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
        """Create comprehensive ML training sample"""
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
            'distance_from_sma20': row['distance_from_sma20'],
            'prev_candle_color': row['prev_candle_color'],
            'prev_candle_body': row['prev_candle_body'],
            'prev_candle_range': row['prev_candle_range'],
            'consecutive_bullish': row['consecutive_bullish'],
            'consecutive_bearish': row['consecutive_bearish'],
            'recent_volatility': row['recent_volatility'],
            'recent_trend': row['recent_trend'],
            'price_momentum': row['price_momentum'],
            'price_momentum_3': row['price_momentum_3'],
            'price_momentum_5': row['price_momentum_5'],
            'volume_ratio': row['volume_ratio'],
            'rsi': row['rsi'],
            'macd': row['macd'],
            'macd_signal': row['macd_signal'],
            'macd_histogram': row['macd_histogram'],
            'bb_position': row['bb_position'],
            'stoch_k': row['stoch_k'],
            'stoch_d': row['stoch_d'],
            'atr': row['atr'],
            'support_distance': row['support_distance'],
            'resistance_distance': row['resistance_distance'],
            'reverted': reverted
        }
    
    def train_ml_model(self):
        """Train advanced ML model for reversion prediction"""
        print("Training advanced ML model...")
        
        # Prepare training data
        training_df = self.prepare_ml_training_data()
        
        if len(training_df) == 0:
            print("No training data available!")
            return False
        
        # Prepare features
        feature_columns = [
            'candle_body', 'candle_range', 'candle_color',
            'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio',
            'distance_from_sma5', 'distance_from_sma10', 'distance_from_sma20',
            'prev_candle_color', 'prev_candle_body', 'prev_candle_range',
            'consecutive_bullish', 'consecutive_bearish',
            'recent_volatility', 'recent_trend',
            'price_momentum', 'price_momentum_3', 'price_momentum_5',
            'volume_ratio', 'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'stoch_k', 'stoch_d', 'atr',
            'support_distance', 'resistance_distance'
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
        
        print(f"Training data shape: {X.shape}")
        print(f"Reversion rate: {y.mean():.3f} ({y.sum()}/{len(y)})")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        models = {
            'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, random_state=42),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            test_score = model.score(X_test_scaled, y_test)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            auc_score = roc_auc_score(y_test, y_proba)
            
            print(f"{name} Results:")
            print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            print(f"  Test Score: {test_score:.3f}")
            print(f"  AUC Score: {auc_score:.3f}")
            
            if auc_score > best_score:
                best_score = auc_score
                best_model = model
        
        self.model = best_model
        
        # Feature importance analysis
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
        
        return True
    
    def check_entry_conditions(self, index):
        """Check if entry conditions are met (loosened)"""
        row = self.df.iloc[index]
        
        # Check SMA5 distance (loosened)
        distance_sma5 = abs(row['close'] - row['SMA5']) / row['close']
        
        if distance_sma5 >= self.sma_distance:
            # Determine direction based on price vs SMA5
            if row['close'] < row['SMA5']:
                direction = 'LONG'
                sma_value = row['SMA5']
            else:
                direction = 'SHORT'
                sma_value = row['SMA5']
            
            return direction, 'SMA5', sma_value
        
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
            'distance_from_sma20': row['distance_from_sma20'],
            'prev_candle_color': row['prev_candle_color'],
            'prev_candle_body': row['prev_candle_body'],
            'prev_candle_range': row['prev_candle_range'],
            'consecutive_bullish': row['consecutive_bullish'],
            'consecutive_bearish': row['consecutive_bearish'],
            'recent_volatility': row['recent_volatility'],
            'recent_trend': row['recent_trend'],
            'price_momentum': row['price_momentum'],
            'price_momentum_3': row['price_momentum_3'],
            'price_momentum_5': row['price_momentum_5'],
            'volume_ratio': row['volume_ratio'],
            'rsi': row['rsi'],
            'macd': row['macd'],
            'macd_signal': row['macd_signal'],
            'macd_histogram': row['macd_histogram'],
            'bb_position': row['bb_position'],
            'stoch_k': row['stoch_k'],
            'stoch_d': row['stoch_d'],
            'atr': row['atr'],
            'support_distance': row['support_distance'],
            'resistance_distance': row['resistance_distance'],
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
        """Run the complete backtest with loosened rules"""
        print("Running quant strategy backtest...")
        
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
                        'ml_probability': position['ml_probability']
                    }
                    self.trades.append(trade)
                    position = None
            
            # Check for new entry if not in position
            if position is None:
                direction, sma_col, sma_value = self.check_entry_conditions(i)
                
                if direction is not None:
                    # Get ML probability
                    ml_prob = self.get_ml_probability(i)
                    
                    # Check ML probability threshold (loosened)
                    if ml_prob >= self.prob_cutoff:
                        # Enter position
                        position = {
                            'entry_time': current_time,
                            'entry_index': i,
                            'direction': direction,
                            'entry_price': current_price,
                            'sma_col': sma_col,
                            'sma_value': sma_value,
                            'ml_probability': ml_prob
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
                'ml_probability': position['ml_probability']
            }
            self.trades.append(trade)
        
        return self.trades
    
    def check_exit_conditions(self, index, position):
        """Check if exit conditions are met (loosened)"""
        current_price = self.df.iloc[index]['close']
        entry_price = position['entry_price']
        
        # Take Profit: Percentage-based (loosened)
        if position['direction'] == 'LONG' and current_price >= entry_price * (1 + self.take_profit_pct):
            return 'Take Profit'
        elif position['direction'] == 'SHORT' and current_price <= entry_price * (1 - self.take_profit_pct):
            return 'Take Profit'
        
        # Stop Loss: Percentage-based (loosened)
        if position['direction'] == 'LONG' and current_price <= entry_price * (1 - self.stop_loss_pct):
            return 'Stop Loss'
        elif position['direction'] == 'SHORT' and current_price >= entry_price * (1 + self.stop_loss_pct):
            return 'Stop Loss'
        
        # Timeout: Held for more than max_hold_candles (increased)
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
        
        print("\n" + "="*70)
        print("QUANT ML REVERSION STRATEGY RESULTS")
        print("="*70)
        
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
        
        # ML Probability analysis
        print(f"\nML Probability Analysis:")
        print(f"Average ML Probability: {trades_df['ml_probability'].mean():.3f}")
        print(f"ML Probability Range: {trades_df['ml_probability'].min():.3f} - {trades_df['ml_probability'].max():.3f}")
        
        # Exit reason analysis
        print(f"\nExit Reasons:")
        exit_reasons = trades_df['exit_reason'].value_counts()
        for reason, count in exit_reasons.items():
            pct = count / total_trades * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
        
        # Direction analysis
        print(f"\nDirection Performance:")
        direction_perf = trades_df.groupby('direction').agg({
            'pnl': ['count', 'mean', 'sum'],
            'ml_probability': 'mean'
        }).round(4)
        print(direction_perf)
        
        # ML Probability buckets
        print(f"\nML Probability Buckets:")
        # Use more granular buckets for maximum insight
        bins = [0, 0.85, 0.90, 0.95, 0.98, 0.995, 1.0]
        labels = ['0-0.85', '0.85-0.90', '0.90-0.95', '0.95-0.98', '0.98-0.995', '0.995-1.0']
        trades_df['prob_bucket'] = pd.cut(trades_df['ml_probability'], bins=bins, labels=labels, include_lowest=True)
        prob_perf = trades_df.groupby('prob_bucket').agg({
            'pnl': ['count', 'mean', 'sum'],
            'ml_probability': 'mean',
            'direction': lambda x: (x == 'LONG').sum() / len(x)
        }).round(4)
        print(prob_perf)
        # Win/loss by bucket
        print("\nWin/Loss by ML Probability Bucket:")
        for bucket in labels:
            bucket_trades = trades_df[trades_df['prob_bucket'] == bucket]
            if len(bucket_trades) == 0:
                continue
            win = (bucket_trades['pnl'] > 0).sum()
            loss = (bucket_trades['pnl'] < 0).sum()
            win_rate = win / len(bucket_trades) * 100
            print(f"  {bucket}: {len(bucket_trades)} trades | Win: {win} | Loss: {loss} | Win Rate: {win_rate:.1f}% | Avg P&L: {bucket_trades['pnl'].mean():.4f}")
        
        # Save results
        trades_df.to_csv('quant_strategy_backtest_results.csv', index=False)
        print(f"\nResults saved to: quant_strategy_backtest_results.csv")
        
        # Create visualizations
        self.create_strategy_visualizations(trades_df)
    
    def create_strategy_visualizations(self, trades_df):
        """Create strategy performance visualizations"""
        print("Creating strategy visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
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
        
        # 3. ML Probability vs P&L
        axes[0,2].scatter(trades_df['ml_probability'], trades_df['pnl'], alpha=0.6)
        axes[0,2].set_title('ML Probability vs P&L')
        axes[0,2].set_xlabel('ML Probability')
        axes[0,2].set_ylabel('P&L')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Exit Reasons
        exit_reasons = trades_df['exit_reason'].value_counts()
        exit_reasons.plot(kind='bar', ax=axes[1,0], color='orange')
        axes[1,0].set_title('Exit Reasons')
        axes[1,0].set_ylabel('Count')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 5. Direction Performance
        direction_pnl = trades_df.groupby('direction')['pnl'].mean()
        direction_pnl.plot(kind='bar', ax=axes[1,1], color=['red', 'green'])
        axes[1,1].set_title('Average P&L by Direction')
        axes[1,1].set_ylabel('Average P&L')
        
        # 6. ML Probability Distribution
        axes[1,2].hist(trades_df['ml_probability'], bins=15, alpha=0.7, color='purple')
        axes[1,2].set_title('ML Probability Distribution')
        axes[1,2].set_xlabel('ML Probability')
        axes[1,2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('quant_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Strategy visualizations saved to: quant_strategy_performance.png")

def main():
    """Main execution function"""
    print("="*70)
    print("QUANT-STYLE ML REVERSION STRATEGY")
    print("="*70)
    print("Loosened Entry/Exit Rules for Maximum ML Training Data")
    print("="*70)
    
    # Initialize strategy with loosened parameters
    strategy = QuantMLReversionStrategy(
        data_path='stock_data/XAUUSDm_M5_1year_data.csv',
        prob_cutoff=0.70,  # Loosened to 70% ML probability threshold
        sma_distance=0.001,  # Loosened to 0.1% minimum distance
        max_hold_candles=15  # Increased to 15 candles max hold
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
    print("QUANT STRATEGY BACKTEST COMPLETE!")
    print("="*70)
    print("Check the generated files for detailed results:")
    print("  - quant_strategy_backtest_results.csv")
    print("  - quant_strategy_performance.png")

if __name__ == "__main__":
    main() 