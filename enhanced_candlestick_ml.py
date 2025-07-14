#!/usr/bin/env python3
"""
ENHANCED CANDLESTICK PATTERN RECOGNITION & MEAN REVERSION ML PIPELINE
Comprehensive candlestick pattern analysis for SMA mean reversion prediction
NO INDICATORS - ONLY PURE PRICE ACTION AND CANDLESTICK STRUCTURE

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import shap

class EnhancedCandlestickAnalyzer:
    def __init__(self, data_path, lookahead_candles=30):
        """
        Initialize the Enhanced Candlestick Analyzer
        
        Args:
            data_path: Path to the CSV file with OHLC data
            lookahead_candles: Number of candles to look ahead for reversion
        """
        self.data_path = data_path
        self.lookahead_candles = lookahead_candles
        self.df = None
        self.events_df = None
        self.sma_periods = [5, 10, 15, 20, 25, 30]  # Focus on most effective SMAs
        
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
        
        # Calculate enhanced candlestick features
        self.calculate_enhanced_candlestick_features()
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        print(f"Calculated SMAs: {self.sma_periods}")
        return self.df
    
    def calculate_enhanced_candlestick_features(self):
        """Calculate comprehensive candlestick structure features"""
        print("Calculating enhanced candlestick features...")
        
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
        
        # Enhanced ratios
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
        
        # Body position within candle
        self.df['body_position'] = np.where(
            self.df['candle_range'] > 0,
            (self.df['close'] - self.df['low']) / self.df['candle_range'],
            0.5
        )
        
        # Previous candle features
        self.df['prev_candle_color'] = self.df['candle_color'].shift(1)
        self.df['prev_candle_body'] = self.df['candle_body'].shift(1)
        self.df['prev_candle_range'] = self.df['candle_range'].shift(1)
        
        # Gap analysis
        self.df['gap_from_prior'] = self.df['open'] - self.df['close'].shift(1)
        self.df['gap_pct'] = np.where(
            self.df['close'].shift(1) > 0,
            self.df['gap_from_prior'] / self.df['close'].shift(1) * 100,
            0
        )
        
        # Recent price behavior
        self.df['recent_volatility'] = self.df['close'].rolling(5).std()
        self.df['recent_trend'] = self.df['close'].rolling(5).mean() - self.df['close'].rolling(10).mean()
        
        # Consecutive candles
        self.calculate_consecutive_candles()
        
        # Detect candlestick patterns
        self.detect_candlestick_patterns()
    
    def calculate_consecutive_candles(self):
        """Calculate consecutive bullish/bearish candles"""
        self.df['consecutive_bullish'] = 0
        self.df['consecutive_bearish'] = 0
        
        for i in range(1, len(self.df)):
            if self.df.iloc[i]['candle_color'] == 1:
                self.df.iloc[i, self.df.columns.get_loc('consecutive_bullish')] = \
                    self.df.iloc[i-1]['consecutive_bullish'] + 1
            else:
                self.df.iloc[i, self.df.columns.get_loc('consecutive_bearish')] = \
                    self.df.iloc[i-1]['consecutive_bearish'] + 1
    
    def detect_candlestick_patterns(self):
        """Detect comprehensive candlestick patterns"""
        print("Detecting candlestick patterns...")
        
        # Initialize pattern columns
        pattern_columns = [
            'doji', 'long_legged_doji', 'hammer', 'inverted_hammer',
            'shooting_star', 'hanging_man', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing',
            'morning_star', 'evening_star'
        ]
        
        for pattern in pattern_columns:
            self.df[pattern] = 0
        
        # Single candle patterns
        self.detect_single_candle_patterns()
        
        # Multi-candle patterns
        self.detect_multi_candle_patterns()
        
        # Create pattern name column
        self.create_pattern_name_column()
    
    def detect_single_candle_patterns(self):
        """Detect single candle patterns"""
        
        # Doji patterns
        self.df['doji'] = (self.df['body_ratio'] < 0.1).astype(int)
        
        # Long-legged doji (doji with long wicks)
        self.df['long_legged_doji'] = (
            (self.df['body_ratio'] < 0.1) &
            ((self.df['upper_wick_ratio'] > 2) | (self.df['lower_wick_ratio'] > 2))
        ).astype(int)
        
        # Hammer (long lower wick, small body, bullish)
        self.df['hammer'] = (
            (self.df['lower_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 1)
        ).astype(int)
        
        # Inverted Hammer (long upper wick, small body, bullish)
        self.df['inverted_hammer'] = (
            (self.df['upper_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 1)
        ).astype(int)
        
        # Shooting Star (long upper wick, small body, bearish)
        self.df['shooting_star'] = (
            (self.df['upper_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 0)
        ).astype(int)
        
        # Hanging Man (long lower wick, small body, bearish)
        self.df['hanging_man'] = (
            (self.df['lower_wick_ratio'] > 2) &
            (self.df['body_ratio'] < 0.3) &
            (self.df['candle_color'] == 0)
        ).astype(int)
        
        # Spinning Top (small body, equal wicks)
        self.df['spinning_top'] = (
            (self.df['body_ratio'] < 0.2) &
            (self.df['upper_wick_ratio'] > 1) &
            (self.df['lower_wick_ratio'] > 1) &
            (abs(self.df['upper_wick_ratio'] - self.df['lower_wick_ratio']) < 0.5)
        ).astype(int)
    
    def detect_multi_candle_patterns(self):
        """Detect multi-candle patterns"""
        
        # Engulfing patterns
        self.df['bullish_engulfing'] = (
            (self.df['candle_color'] == 1) &
            (self.df['prev_candle_color'] == 0) &
            (self.df['open'] < self.df['close'].shift(1)) &
            (self.df['close'] > self.df['open'].shift(1))
        ).astype(int)
        
        self.df['bearish_engulfing'] = (
            (self.df['candle_color'] == 0) &
            (self.df['prev_candle_color'] == 1) &
            (self.df['open'] > self.df['close'].shift(1)) &
            (self.df['close'] < self.df['open'].shift(1))
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
    
    def create_pattern_name_column(self):
        """Create a pattern name column for each candle"""
        pattern_columns = [
            'doji', 'long_legged_doji', 'hammer', 'inverted_hammer',
            'shooting_star', 'hanging_man', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing',
            'morning_star', 'evening_star'
        ]
        
        self.df['pattern_name'] = 'none'
        
        for pattern in pattern_columns:
            mask = self.df[pattern] == 1
            self.df.loc[mask, 'pattern_name'] = pattern
        
        # Count pattern frequencies
        pattern_counts = self.df['pattern_name'].value_counts()
        print(f"\nPattern frequencies:")
        print(pattern_counts)
    
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
                events.append(self.create_enhanced_event_record(i, sma_period, event_type, current_price, current_sma))
            
            # Detect bearish crossover
            elif prev_price >= prev_sma and current_price < current_sma:
                event_type = 'bearish'
                events.append(self.create_enhanced_event_record(i, sma_period, event_type, current_price, current_sma))
        
        return events
    
    def create_enhanced_event_record(self, index, sma_period, event_type, price, sma):
        """Create a comprehensive event record with all enhanced features"""
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
            
            # Enhanced candlestick structure
            'candle_body': row['candle_body'],
            'candle_range': row['candle_range'],
            'candle_color': row['candle_color'],
            'upper_wick_ratio': row['upper_wick_ratio'],
            'lower_wick_ratio': row['lower_wick_ratio'],
            'body_ratio': row['body_ratio'],
            'body_position': row['body_position'],
            'gap_pct': row['gap_pct'],
            
            # Previous candle features
            'prev_candle_color': row['prev_candle_color'],
            'prev_candle_body': row['prev_candle_body'],
            'prev_candle_range': row['prev_candle_range'],
            
            # Recent behavior
            'consecutive_bullish': row['consecutive_bullish'],
            'consecutive_bearish': row['consecutive_bearish'],
            'recent_volatility': row['recent_volatility'],
            'recent_trend': row['recent_trend'],
            
            # Candlestick patterns
            'pattern_name': row['pattern_name'],
            'doji': row['doji'],
            'long_legged_doji': row['long_legged_doji'],
            'hammer': row['hammer'],
            'inverted_hammer': row['inverted_hammer'],
            'shooting_star': row['shooting_star'],
            'hanging_man': row['hanging_man'],
            'spinning_top': row['spinning_top'],
            'bullish_engulfing': row['bullish_engulfing'],
            'bearish_engulfing': row['bearish_engulfing'],
            'morning_star': row['morning_star'],
            'evening_star': row['evening_star']
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
            reversion_record = dict(event)
            reversion_record['reverted'] = reverted
            reversion_record['candles_to_reversion'] = candles_to_reversion
            reversion_record['max_adverse_move'] = max_adverse_move
            reversion_record['max_favorable_move'] = max_favorable_move
            reversion_record['max_adverse_move_pct'] = (max_adverse_move / event['price_at_crossover']) * 100
            reversion_record['max_favorable_move_pct'] = (max_favorable_move / event['price_at_crossover']) * 100
            reversion_data.append(reversion_record)
        
        self.reversion_df = pd.DataFrame(reversion_data)
        
        # Calculate summary statistics
        self.print_enhanced_summary()
        
        return self.reversion_df
    
    def print_enhanced_summary(self):
        """Print comprehensive reversion summary with pattern analysis"""
        print("\n" + "="*70)
        print("ENHANCED CANDLESTICK PATTERN REVERSION ANALYSIS")
        print("="*70)
        
        total_events = len(self.reversion_df)
        reversion_rate = self.reversion_df['reverted'].mean() * 100
        avg_candles_to_reversion = self.reversion_df[self.reversion_df['reverted']]['candles_to_reversion'].mean()
        
        print(f"Total Events: {total_events:,}")
        print(f"Overall Reversion Rate: {reversion_rate:.1f}%")
        print(f"Average Candles to Reversion: {avg_candles_to_reversion:.1f}")
        
        # Pattern analysis
        print("\n" + "="*50)
        print("PATTERN PERFORMANCE ANALYSIS")
        print("="*50)
        
        pattern_performance = self.reversion_df.groupby('pattern_name').agg({
            'reverted': ['count', 'mean'],
            'candles_to_reversion': 'mean',
            'max_adverse_move_pct': 'mean'
        }).round(3)
        
        pattern_performance.columns = ['Event_Count', 'Reversion_Rate', 'Avg_Candles', 'Avg_Adverse_Move']
        pattern_performance['Reversion_Rate'] = pattern_performance['Reversion_Rate'] * 100
        
        # Filter patterns with sufficient events
        significant_patterns = pattern_performance[pattern_performance['Event_Count'] >= 10]
        print(significant_patterns.sort_values('Reversion_Rate', ascending=False))
        
        # SMA analysis
        print("\n" + "="*50)
        print("SMA PERFORMANCE ANALYSIS")
        print("="*50)
        
        sma_summary = self.reversion_df.groupby('sma_period').agg({
            'reverted': ['count', 'mean'],
            'candles_to_reversion': 'mean'
        }).round(3)
        
        sma_summary.columns = ['Event_Count', 'Reversion_Rate', 'Avg_Candles']
        sma_summary['Reversion_Rate'] = sma_summary['Reversion_Rate'] * 100
        print(sma_summary)
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print("Creating comprehensive visualizations...")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Pattern Frequency
        pattern_counts = self.reversion_df['pattern_name'].value_counts()
        pattern_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Pattern Frequency')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Reversion Rate by Pattern
        pattern_reversion = self.reversion_df.groupby('pattern_name')['reverted'].mean() * 100
        pattern_reversion.plot(kind='bar', ax=axes[0,1], color='lightgreen')
        axes[0,1].set_title('Reversion Rate by Pattern')
        axes[0,1].set_ylabel('Reversion Rate (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Candles to Reversion by Pattern
        pattern_candles = self.reversion_df[self.reversion_df['reverted']].groupby('pattern_name')['candles_to_reversion'].mean()
        pattern_candles.plot(kind='bar', ax=axes[0,2], color='orange')
        axes[0,2].set_title('Avg Candles to Reversion by Pattern')
        axes[0,2].set_ylabel('Candles')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. SMA Performance
        sma_reversion = self.reversion_df.groupby('sma_period')['reverted'].mean() * 100
        sma_reversion.plot(kind='bar', ax=axes[1,0], color='purple')
        axes[1,0].set_title('Reversion Rate by SMA Period')
        axes[1,0].set_ylabel('Reversion Rate (%)')
        axes[1,0].set_xlabel('SMA Period')
        
        # 5. Body Ratio vs Reversion
        axes[1,1].scatter(self.reversion_df['body_ratio'], 
                         self.reversion_df['candles_to_reversion'], 
                         alpha=0.6, c=self.reversion_df['reverted'], cmap='viridis')
        axes[1,1].set_title('Body Ratio vs Candles to Reversion')
        axes[1,1].set_xlabel('Body Ratio')
        axes[1,1].set_ylabel('Candles to Reversion')
        
        # 6. Wick Ratios vs Reversion
        axes[1,2].scatter(self.reversion_df['upper_wick_ratio'], 
                         self.reversion_df['lower_wick_ratio'], 
                         alpha=0.6, c=self.reversion_df['reverted'], cmap='viridis')
        axes[1,2].set_title('Upper vs Lower Wick Ratios')
        axes[1,2].set_xlabel('Upper Wick Ratio')
        axes[1,2].set_ylabel('Lower Wick Ratio')
        
        # 7. Distribution of Max Adverse Move
        axes[2,0].hist(self.reversion_df['max_adverse_move_pct'], bins=30, alpha=0.7, color='red')
        axes[2,0].set_title('Distribution of Max Adverse Move')
        axes[2,0].set_xlabel('Max Adverse Move (%)')
        axes[2,0].set_ylabel('Frequency')
        
        # 8. Event Type Performance
        type_reversion = self.reversion_df.groupby('event_type')['reverted'].mean() * 100
        type_reversion.plot(kind='bar', ax=axes[2,1], color=['red', 'green'])
        axes[2,1].set_title('Reversion Rate by Event Type')
        axes[2,1].set_ylabel('Reversion Rate (%)')
        
        # 9. Pattern Success Heatmap
        pattern_success = self.reversion_df.groupby(['pattern_name', 'event_type'])['reverted'].mean().unstack(fill_value=0)
        sns.heatmap(pattern_success, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[2,2])
        axes[2,2].set_title('Pattern Success Rate by Event Type')
        
        plt.tight_layout()
        plt.savefig('enhanced_candlestick_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to: enhanced_candlestick_analysis.png")
    
    def train_enhanced_ml_model(self):
        """Train enhanced ML model with pattern features"""
        print("Training enhanced ML model...")
        
        # Prepare features
        feature_columns = [
            'sma_period', 'distance_from_sma_pct',
            'candle_body', 'candle_range', 'candle_color',
            'upper_wick_ratio', 'lower_wick_ratio', 'body_ratio', 'body_position',
            'gap_pct', 'prev_candle_color', 'prev_candle_body',
            'consecutive_bullish', 'consecutive_bearish',
            'recent_volatility', 'recent_trend',
            'doji', 'long_legged_doji', 'hammer', 'inverted_hammer',
            'shooting_star', 'hanging_man', 'spinning_top',
            'bullish_engulfing', 'bearish_engulfing',
            'morning_star', 'evening_star'
        ]
        
        # Create event type dummy
        self.reversion_df['event_type_bullish'] = (self.reversion_df['event_type'] == 'bullish').astype(int)
        feature_columns.append('event_type_bullish')
        
        X = self.reversion_df[feature_columns].copy()
        y = self.reversion_df['reverted'].astype(int)
        
        # Remove NaN values
        valid_indices = ~X.isna().any(axis=1)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost model
        print("Training XGBoost Classifier...")
        xgb_model = XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = xgb_model.predict(X_test_scaled)
        y_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        print("\n" + "="*50)
        print("ENHANCED ML MODEL RESULTS")
        print("="*50)
        print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15))
        
        # Create SHAP plot
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test_scaled[:100])  # Sample for speed
            
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_scaled[:100], feature_names=feature_columns, show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("SHAP plot saved to: shap_importance.png")
        except Exception as e:
            print(f"SHAP plot creation failed: {e}")
        
        return xgb_model, scaler, feature_columns
    
    def save_enhanced_results(self):
        """Save enhanced analysis results"""
        print("\nSaving enhanced results...")
        
        # Save reversion analysis
        self.reversion_df.to_csv('enhanced_candlestick_reversion_analysis.csv', index=False)
        
        # Save pattern performance summary
        pattern_performance = self.reversion_df.groupby('pattern_name').agg({
            'reverted': ['count', 'mean'],
            'candles_to_reversion': 'mean',
            'max_adverse_move_pct': 'mean'
        }).round(3)
        
        pattern_performance.columns = ['Event_Count', 'Reversion_Rate', 'Avg_Candles', 'Avg_Adverse_Move']
        pattern_performance['Reversion_Rate'] = pattern_performance['Reversion_Rate'] * 100
        pattern_performance.to_csv('pattern_performance_summary.csv')
        
        # Save summary statistics
        summary_stats = {
            'total_events': len(self.reversion_df),
            'reversion_rate': self.reversion_df['reverted'].mean() * 100,
            'avg_candles_to_reversion': self.reversion_df[self.reversion_df['reverted']]['candles_to_reversion'].mean(),
            'avg_adverse_move': self.reversion_df['max_adverse_move_pct'].mean(),
            'avg_favorable_move': self.reversion_df['max_favorable_move_pct'].mean()
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv('enhanced_summary.csv', index=False)
        
        print("Enhanced results saved to:")
        print("  - enhanced_candlestick_reversion_analysis.csv")
        print("  - pattern_performance_summary.csv")
        print("  - enhanced_summary.csv")

def main():
    """Main execution function"""
    print("="*70)
    print("ENHANCED CANDLESTICK PATTERN RECOGNITION & MEAN REVERSION ML")
    print("="*70)
    print("COMPREHENSIVE PATTERN ANALYSIS WITH VISUALIZATION")
    print("="*70)
    
    # Initialize analyzer
    analyzer = EnhancedCandlestickAnalyzer('stock_data/XAUUSDm_M5_1year_data.csv', lookahead_candles=30)
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Detect crossover events
    analyzer.detect_sma_crossover_events()
    
    # Analyze reversion events
    analyzer.analyze_reversion_events()
    
    # Create visualizations
    analyzer.create_comprehensive_visualizations()
    
    # Train enhanced ML model
    analyzer.train_enhanced_ml_model()
    
    # Save results
    analyzer.save_enhanced_results()
    
    print("\n" + "="*70)
    print("ENHANCED CANDLESTICK ANALYSIS COMPLETE!")
    print("="*70)
    print("Check the generated files for detailed results:")
    print("  - enhanced_candlestick_reversion_analysis.csv")
    print("  - pattern_performance_summary.csv")
    print("  - enhanced_summary.csv")
    print("  - enhanced_candlestick_analysis.png")
    print("  - shap_importance.png")

if __name__ == "__main__":
    main() 