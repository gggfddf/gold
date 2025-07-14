#!/usr/bin/env python3
"""
PURE PRICE ACTION SMA MEAN REVERSION MACHINE LEARNING PIPELINE
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans

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
        self.sma_periods = list(range(5, 101, 5))  # SMA5 to SMA100 every 5 steps
        
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
        
        # Engulfing patterns (simplified)
        self.df['is_bullish_engulfing'] = (
            (self.df['candle_color'] == 1) &
            (self.df['open'] < self.df['close'].shift(1)) &
            (self.df['close'] > self.df['open'].shift(1))
        ).astype(int)
        
        self.df['is_bearish_engulfing'] = (
            (self.df['candle_color'] == 0) &
            (self.df['open'] > self.df['close'].shift(1)) &
            (self.df['close'] < self.df['open'].shift(1))
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
            'gap_from_prior': row['gap_from_prior'],
            'gap_pct': row['gap_pct'],
            
            # Candlestick patterns
            'is_doji': row['is_doji'],
            'is_hammer': row['is_hammer'],
            'is_shooting_star': row['is_shooting_star'],
            'is_bullish_engulfing': row['is_bullish_engulfing'],
            'is_bearish_engulfing': row['is_bearish_engulfing'],
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
            
            # Store results
            reversion_record = event.copy()
            reversion_record['reverted'] = reverted
            reversion_record['candles_to_reversion'] = candles_to_reversion
            reversion_record['max_adverse_move'] = max_adverse_move
            reversion_record['max_favorable_move'] = max_favorable_move
            reversion_record['max_adverse_move_pct'] = (max_adverse_move / event['price_at_crossover']) * 100
            reversion_record['max_favorable_move_pct'] = (max_favorable_move / event['price_at_crossover']) * 100
            
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
    
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        print("Preparing ML features...")
        
        # Create dummy variables for categorical features
        self.reversion_df['event_type_bullish'] = (self.reversion_df['event_type'] == 'bullish').astype(int)
        
        # Select features (pure price action only)
        features = [
            # SMA and price features
            'sma_period', 'distance_from_sma_pct',
            
            # Candlestick structure
            'candle_body', 'candle_range', 'candle_color',
            'upper_wick_ratio', 'lower_wick_ratio', 'body_to_range_ratio',
            'close_position', 'gap_pct',
            
            # Candlestick patterns
            'is_doji', 'is_hammer', 'is_shooting_star',
            'is_bullish_engulfing', 'is_bearish_engulfing', 'is_indecision',
            
            # Recent behavior
            'consecutive_bullish', 'consecutive_bearish',
            'recent_volatility', 'recent_trend',
            
            # Event type
            'event_type_bullish'
        ]
        
        # Prepare X and y
        X = self.reversion_df[features].copy()
        
        # Target variables
        y_reversion = self.reversion_df['reverted'].astype(int)
        y_candles = self.reversion_df['candles_to_reversion'].fillna(self.lookahead_candles)
        y_adverse_move = self.reversion_df['max_adverse_move_pct']
        
        # Remove rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y_candles.isna())
        X = X[valid_indices]
        y_reversion = y_reversion[valid_indices]
        y_candles = y_candles[valid_indices]
        y_adverse_move = y_adverse_move[valid_indices]
        
        print(f"ML features prepared: {len(X)} samples, {len(features)} features")
        
        return X, y_reversion, y_candles, y_adverse_move, features
    
    def train_models(self, X, y_reversion, y_candles, y_adverse_move):
        """Train machine learning models"""
        print("Training machine learning models...")
        
        # Split data
        X_train, X_test, y_reversion_train, y_reversion_test = train_test_split(
            X, y_reversion, test_size=0.2, random_state=42, stratify=y_reversion
        )
        
        X_train, X_test, y_candles_train, y_candles_test = train_test_split(
            X, y_candles, test_size=0.2, random_state=42
        )
        
        X_train, X_test, y_adverse_train, y_adverse_test = train_test_split(
            X, y_adverse_move, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        print("Training Reversion Classification Model...")
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_reversion_train)
        
        print("Training Candles to Reversion Model...")
        xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)
        xgb_regressor.fit(X_train_scaled, y_candles_train)
        
        print("Training Max Adverse Move Model...")
        xgb_adverse = XGBRegressor(n_estimators=100, random_state=42)
        xgb_adverse.fit(X_train_scaled, y_adverse_train)
        
        # Evaluate models
        self.evaluate_models(
            rf_classifier, xgb_regressor, xgb_adverse,
            X_test_scaled, y_reversion_test, y_candles_test, y_adverse_test,
            scaler, X.columns
        )
        
        return rf_classifier, xgb_regressor, xgb_adverse, scaler
    
    def evaluate_models(self, rf_classifier, xgb_regressor, xgb_adverse, 
                       X_test, y_reversion_test, y_candles_test, y_adverse_test, scaler, feature_names):
        """Evaluate model performance"""
        print("\n" + "="*60)
        print("PURE PRICE ACTION ML MODEL EVALUATION")
        print("="*60)
        
        # 1. Reversion Classification
        y_reversion_pred = rf_classifier.predict(X_test)
        y_reversion_proba = rf_classifier.predict_proba(X_test)[:, 1]
        
        print("\n1. REVERSION CLASSIFICATION (Random Forest)")
        print("-" * 50)
        print(f"Accuracy: {(y_reversion_pred == y_reversion_test).mean():.3f}")
        print("\nClassification Report:")
        print(classification_report(y_reversion_test, y_reversion_pred))
        
        # 2. Candles to Reversion Regression
        y_candles_pred = xgb_regressor.predict(X_test)
        candles_mse = mean_squared_error(y_candles_test, y_candles_pred)
        candles_r2 = r2_score(y_candles_test, y_candles_pred)
        
        print("\n2. CANDLES TO REVERSION REGRESSION (XGBoost)")
        print("-" * 50)
        print(f"Mean Squared Error: {candles_mse:.3f}")
        print(f"R² Score: {candles_r2:.3f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_candles_test - y_candles_pred)):.2f} candles")
        
        # 3. Max Adverse Move Regression
        y_adverse_pred = xgb_adverse.predict(X_test)
        adverse_mse = mean_squared_error(y_adverse_test, y_adverse_pred)
        adverse_r2 = r2_score(y_adverse_test, y_adverse_pred)
        
        print("\n3. MAX ADVERSE MOVE REGRESSION (XGBoost)")
        print("-" * 50)
        print(f"Mean Squared Error: {adverse_mse:.3f}")
        print(f"R² Score: {adverse_r2:.3f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_adverse_test - y_adverse_pred)):.2f}%")
        
        # Feature importance
        print("\n4. FEATURE IMPORTANCE (Top 15)")
        print("-" * 50)
        
        rf_importance = rf_classifier.feature_importances_
        xgb_importance = xgb_regressor.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'RF_Reversion': rf_importance,
            'XGB_Candles': xgb_importance
        }).sort_values('RF_Reversion', ascending=False)
        
        print(importance_df.head(15))
        
        # Save results
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
    
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
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(3, 2, figsize=(18, 15))
        
        # 1. Reversion Rate by SMA Period
        sma_reversion = self.reversion_df.groupby('sma_period')['reverted'].mean() * 100
        sma_reversion.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Reversion Rate by SMA Period')
        axes[0,0].set_ylabel('Reversion Rate (%)')
        axes[0,0].set_xlabel('SMA Period')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Average Candles to Reversion by SMA
        sma_candles = self.reversion_df[self.reversion_df['reverted']].groupby('sma_period')['candles_to_reversion'].mean()
        sma_candles.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Average Candles to Reversion by SMA Period')
        axes[0,1].set_ylabel('Candles to Reversion')
        axes[0,1].set_xlabel('SMA Period')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Candlestick Pattern vs Reversion Rate
        pattern_features = ['is_doji', 'is_hammer', 'is_shooting_star', 'is_bullish_engulfing', 'is_bearish_engulfing', 'is_indecision']
        pattern_reversion = []
        pattern_names = ['Doji', 'Hammer', 'Shooting Star', 'Bullish Engulfing', 'Bearish Engulfing', 'Indecision']
        
        for feature in pattern_features:
            rate = self.reversion_df[self.reversion_df[feature] == 1]['reverted'].mean() * 100
            pattern_reversion.append(rate)
        
        axes[1,0].bar(pattern_names, pattern_reversion, color='orange')
        axes[1,0].set_title('Reversion Rate by Candlestick Pattern')
        axes[1,0].set_ylabel('Reversion Rate (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Distribution of Max Adverse Move
        axes[1,1].hist(self.reversion_df['max_adverse_move_pct'], bins=30, alpha=0.7, color='red')
        axes[1,1].set_title('Distribution of Max Adverse Move')
        axes[1,1].set_xlabel('Max Adverse Move (%)')
        axes[1,1].set_ylabel('Frequency')
        
        # 5. Wick Ratios vs Reversion
        axes[2,0].scatter(self.reversion_df['upper_wick_ratio'], 
                         self.reversion_df['candles_to_reversion'], 
                         alpha=0.6, c=self.reversion_df['reverted'], cmap='viridis')
        axes[2,0].set_title('Upper Wick Ratio vs Candles to Reversion')
        axes[2,0].set_xlabel('Upper Wick Ratio')
        axes[2,0].set_ylabel('Candles to Reversion')
        
        # 6. Body to Range Ratio vs Reversion Rate
        body_range_bins = pd.cut(self.reversion_df['body_to_range_ratio'], bins=10)
        body_range_reversion = self.reversion_df.groupby(body_range_bins)['reverted'].mean() * 100
        body_range_reversion.plot(kind='bar', ax=axes[2,1], color='purple')
        axes[2,1].set_title('Reversion Rate by Body-to-Range Ratio')
        axes[2,1].set_ylabel('Reversion Rate (%)')
        axes[2,1].set_xlabel('Body-to-Range Ratio')
        axes[2,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('pure_price_action_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to: pure_price_action_analysis.png")

def main():
    """Main execution function"""
    print("="*70)
    print("PURE PRICE ACTION SMA MEAN REVERSION MACHINE LEARNING ANALYSIS")
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
    
    # Prepare ML features
    X, y_reversion, y_candles, y_adverse_move, features = analyzer.prepare_ml_features()
    
    # Train models
    rf_classifier, xgb_regressor, xgb_adverse, scaler = analyzer.train_models(
        X, y_reversion, y_candles, y_adverse_move
    )
    
    print("\n" + "="*70)
    print("PURE PRICE ACTION ANALYSIS COMPLETE!")
    print("="*70)
    print("Check the generated files for detailed results:")
    print("  - pure_price_action_reversion_analysis.csv")
    print("  - pure_price_action_summary.csv")
    print("  - pure_price_action_analysis.png")

if __name__ == "__main__":
    main() 