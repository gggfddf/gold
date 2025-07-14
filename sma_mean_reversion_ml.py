#!/usr/bin/env python3
"""
20 SMA Mean Reversion Machine Learning Pipeline
Analyzes 5-minute candlestick data to predict mean reversion after SMA20 crossovers

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

class SMAMeanReversionAnalyzer:
    def __init__(self, data_path, lookahead_candles=30):
        """
        Initialize the SMA Mean Reversion Analyzer
        
        Args:
            data_path: Path to the CSV file with OHLC data
            lookahead_candles: Number of candles to look ahead for reversion
        """
        self.data_path = data_path
        self.lookahead_candles = lookahead_candles
        self.df = None
        self.events_df = None
        
    def load_and_prepare_data(self):
        """Load data and calculate technical indicators"""
        print("Loading and preparing data...")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        
        # Calculate 20-period SMA
        self.df['SMA20'] = self.df['close'].rolling(window=20).mean()
        
        # Calculate price position relative to SMA20
        self.df['price_vs_sma20'] = self.df['close'] - self.df['SMA20']
        self.df['price_vs_sma20_pct'] = (self.df['close'] - self.df['SMA20']) / self.df['SMA20'] * 100
        
        # Calculate SMA20 slope (angle)
        self.df['sma20_slope'] = self.df['SMA20'].diff(5)  # 5-period slope
        
        # Calculate candle body size
        self.df['candle_body'] = abs(self.df['close'] - self.df['open'])
        self.df['candle_body_pct'] = self.df['candle_body'] / self.df['close'] * 100
        
        # Calculate recent volatility (5-period)
        self.df['volatility_5'] = self.df['close'].rolling(5).std()
        
        # Remove NaN values
        self.df = self.df.dropna().reset_index(drop=True)
        
        print(f"Data loaded: {len(self.df)} records from {self.df['time'].min()} to {self.df['time'].max()}")
        return self.df
    
    def detect_crossover_events(self):
        """Detect price crossover events with SMA20"""
        print("Detecting crossover events...")
        
        events = []
        
        for i in range(1, len(self.df)):
            current_price = self.df.iloc[i]['close']
            current_sma = self.df.iloc[i]['SMA20']
            prev_price = self.df.iloc[i-1]['close']
            prev_sma = self.df.iloc[i-1]['SMA20']
            
            # Detect bullish crossover (price crosses above SMA20)
            if prev_price <= prev_sma and current_price > current_sma:
                event_type = 'bullish'
                events.append({
                    'index': i,
                    'time': self.df.iloc[i]['time'],
                    'event_type': event_type,
                    'price_at_crossover': current_price,
                    'sma_at_crossover': current_sma,
                    'price_vs_sma_pct': self.df.iloc[i]['price_vs_sma20_pct'],
                    'sma_slope': self.df.iloc[i]['sma20_slope'],
                    'candle_body': self.df.iloc[i]['candle_body'],
                    'volatility': self.df.iloc[i]['volatility_5']
                })
            
            # Detect bearish crossover (price crosses below SMA20)
            elif prev_price >= prev_sma and current_price < current_sma:
                event_type = 'bearish'
                events.append({
                    'index': i,
                    'time': self.df.iloc[i]['time'],
                    'event_type': event_type,
                    'price_at_crossover': current_price,
                    'sma_at_crossover': current_sma,
                    'price_vs_sma_pct': self.df.iloc[i]['price_vs_sma20_pct'],
                    'sma_slope': self.df.iloc[i]['sma20_slope'],
                    'candle_body': self.df.iloc[i]['candle_body'],
                    'volatility': self.df.iloc[i]['volatility_5']
                })
        
        self.events_df = pd.DataFrame(events)
        print(f"Detected {len(self.events_df)} crossover events")
        return self.events_df
    
    def analyze_reversion_events(self):
        """Analyze each crossover event for reversion"""
        print("Analyzing reversion events...")
        
        reversion_data = []
        
        for _, event in self.events_df.iterrows():
            event_index = event['index']
            event_type = event['event_type']
            
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
                future_sma = self.df.iloc[future_index]['SMA20']
                
                # Calculate movement from crossover point
                price_change = future_price - event['price_at_crossover']
                price_change_pct = (price_change / event['price_at_crossover']) * 100
                
                # Track max movements
                if event_type == 'bullish':
                    if price_change > max_favorable_move:
                        max_favorable_move = price_change
                    if price_change < max_adverse_move:
                        max_adverse_move = price_change
                    
                    # Check for bearish reversion (price crosses back below SMA20)
                    if future_price <= future_sma:
                        reverted = True
                        candles_to_reversion = lookahead
                        break
                        
                elif event_type == 'bearish':
                    if price_change < max_favorable_move:
                        max_favorable_move = price_change
                    if price_change > max_adverse_move:
                        max_adverse_move = price_change
                    
                    # Check for bullish reversion (price crosses back above SMA20)
                    if future_price >= future_sma:
                        reverted = True
                        candles_to_reversion = lookahead
                        break
            
            # Store results
            reversion_data.append({
                'event_index': event_index,
                'time': event['time'],
                'event_type': event_type,
                'price_at_crossover': event['price_at_crossover'],
                'sma_at_crossover': event['sma_at_crossover'],
                'price_vs_sma_pct': event['price_vs_sma_pct'],
                'sma_slope': event['sma_slope'],
                'candle_body': event['candle_body'],
                'volatility': event['volatility'],
                'reverted': reverted,
                'candles_to_reversion': candles_to_reversion,
                'max_adverse_move': max_adverse_move,
                'max_favorable_move': max_favorable_move,
                'max_adverse_move_pct': (max_adverse_move / event['price_at_crossover']) * 100,
                'max_favorable_move_pct': (max_favorable_move / event['price_at_crossover']) * 100
            })
        
        self.reversion_df = pd.DataFrame(reversion_data)
        
        # Calculate summary statistics
        reversion_rate = self.reversion_df['reverted'].mean() * 100
        avg_candles_to_reversion = self.reversion_df[self.reversion_df['reverted']]['candles_to_reversion'].mean()
        
        print(f"Reversion Analysis Complete:")
        print(f"  - Total events: {len(self.reversion_df)}")
        print(f"  - Reversion rate: {reversion_rate:.1f}%")
        print(f"  - Average candles to reversion: {avg_candles_to_reversion:.1f}")
        
        return self.reversion_df
    
    def prepare_ml_features(self):
        """Prepare features for machine learning"""
        print("Preparing ML features...")
        
        # Create features
        features = [
            'price_vs_sma_pct',      # Price position relative to SMA20
            'sma_slope',             # SMA20 slope
            'candle_body',           # Candle body size
            'volatility',            # Recent volatility
            'event_type_bullish'     # Event type (dummy variable)
        ]
        
        # Create dummy variable for event type
        self.reversion_df['event_type_bullish'] = (self.reversion_df['event_type'] == 'bullish').astype(int)
        
        # Prepare X and y
        X = self.reversion_df[features].copy()
        
        # Target variables
        y_reversion = self.reversion_df['reverted'].astype(int)  # Binary classification
        y_candles = self.reversion_df['candles_to_reversion'].fillna(self.lookahead_candles)  # Regression
        y_adverse_move = self.reversion_df['max_adverse_move_pct']  # Regression
        
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
        
        # 1. Reversion Classification Model (Random Forest)
        print("Training Reversion Classification Model...")
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train_scaled, y_reversion_train)
        
        # 2. Candles to Reversion Regression Model (XGBoost)
        print("Training Candles to Reversion Model...")
        xgb_regressor = XGBRegressor(n_estimators=100, random_state=42)
        xgb_regressor.fit(X_train_scaled, y_candles_train)
        
        # 3. Max Adverse Move Regression Model (XGBoost)
        print("Training Max Adverse Move Model...")
        xgb_adverse = XGBRegressor(n_estimators=100, random_state=42)
        xgb_adverse.fit(X_train_scaled, y_adverse_train)
        
        # Evaluate models
        self.evaluate_models(
            rf_classifier, xgb_regressor, xgb_adverse,
            X_test_scaled, y_reversion_test, y_candles_test, y_adverse_test,
            scaler
        )
        
        return rf_classifier, xgb_regressor, xgb_adverse, scaler
    
    def evaluate_models(self, rf_classifier, xgb_regressor, xgb_adverse, 
                       X_test, y_reversion_test, y_candles_test, y_adverse_test, scaler):
        """Evaluate model performance"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        # 1. Reversion Classification
        y_reversion_pred = rf_classifier.predict(X_test)
        y_reversion_proba = rf_classifier.predict_proba(X_test)[:, 1]
        
        print("\n1. REVERSION CLASSIFICATION (Random Forest)")
        print("-" * 40)
        print(f"Accuracy: {(y_reversion_pred == y_reversion_test).mean():.3f}")
        print("\nClassification Report:")
        print(classification_report(y_reversion_test, y_reversion_pred))
        
        # 2. Candles to Reversion Regression
        y_candles_pred = xgb_regressor.predict(X_test)
        candles_mse = mean_squared_error(y_candles_test, y_candles_pred)
        candles_r2 = r2_score(y_candles_test, y_candles_pred)
        
        print("\n2. CANDLES TO REVERSION REGRESSION (XGBoost)")
        print("-" * 40)
        print(f"Mean Squared Error: {candles_mse:.3f}")
        print(f"R² Score: {candles_r2:.3f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_candles_test - y_candles_pred)):.2f} candles")
        
        # 3. Max Adverse Move Regression
        y_adverse_pred = xgb_adverse.predict(X_test)
        adverse_mse = mean_squared_error(y_adverse_test, y_adverse_pred)
        adverse_r2 = r2_score(y_adverse_test, y_adverse_pred)
        
        print("\n3. MAX ADVERSE MOVE REGRESSION (XGBoost)")
        print("-" * 40)
        print(f"Mean Squared Error: {adverse_mse:.3f}")
        print(f"R² Score: {adverse_r2:.3f}")
        print(f"Mean Absolute Error: {np.mean(np.abs(y_adverse_test - y_adverse_pred)):.2f}%")
        
        # Feature importance
        print("\n4. FEATURE IMPORTANCE")
        print("-" * 40)
        feature_names = ['Price vs SMA20 %', 'SMA20 Slope', 'Candle Body', 'Volatility', 'Bullish Event']
        
        rf_importance = rf_classifier.feature_importances_
        xgb_importance = xgb_regressor.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'RF_Reversion': rf_importance,
            'XGB_Candles': xgb_importance
        })
        
        print(importance_df.sort_values('RF_Reversion', ascending=False))
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save analysis results"""
        print("\nSaving results...")
        
        # Save reversion analysis
        self.reversion_df.to_csv('sma_reversion_analysis.csv', index=False)
        
        # Save summary statistics
        summary_stats = {
            'total_events': len(self.reversion_df),
            'reversion_rate': self.reversion_df['reverted'].mean() * 100,
            'avg_candles_to_reversion': self.reversion_df[self.reversion_df['reverted']]['candles_to_reversion'].mean(),
            'avg_adverse_move': self.reversion_df['max_adverse_move_pct'].mean(),
            'avg_favorable_move': self.reversion_df['max_favorable_move_pct'].mean()
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv('sma_reversion_summary.csv', index=False)
        
        print("Results saved to:")
        print("  - sma_reversion_analysis.csv")
        print("  - sma_reversion_summary.csv")
    
    def create_visualizations(self):
        """Create visualizations of the analysis"""
        print("Creating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reversion Rate by Event Type
        reversion_by_type = self.reversion_df.groupby('event_type')['reverted'].agg(['mean', 'count'])
        reversion_by_type['mean'].plot(kind='bar', ax=axes[0,0], color=['red', 'green'])
        axes[0,0].set_title('Reversion Rate by Event Type')
        axes[0,0].set_ylabel('Reversion Rate')
        axes[0,0].set_ylim(0, 1)
        
        # 2. Distribution of Candles to Reversion
        reverted_events = self.reversion_df[self.reversion_df['reverted']]
        axes[0,1].hist(reverted_events['candles_to_reversion'], bins=20, alpha=0.7, color='blue')
        axes[0,1].set_title('Distribution of Candles to Reversion')
        axes[0,1].set_xlabel('Candles to Reversion')
        axes[0,1].set_ylabel('Frequency')
        
        # 3. Max Adverse Move Distribution
        axes[1,0].hist(self.reversion_df['max_adverse_move_pct'], bins=20, alpha=0.7, color='red')
        axes[1,0].set_title('Distribution of Max Adverse Move')
        axes[1,0].set_xlabel('Max Adverse Move (%)')
        axes[1,0].set_ylabel('Frequency')
        
        # 4. Price vs SMA20 at Crossover
        axes[1,1].scatter(self.reversion_df['price_vs_sma_pct'], 
                         self.reversion_df['candles_to_reversion'], 
                         alpha=0.6, c=self.reversion_df['reverted'], cmap='viridis')
        axes[1,1].set_title('Price vs SMA20 vs Candles to Reversion')
        axes[1,1].set_xlabel('Price vs SMA20 (%)')
        axes[1,1].set_ylabel('Candles to Reversion')
        
        plt.tight_layout()
        plt.savefig('sma_reversion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to: sma_reversion_analysis.png")

def main():
    """Main execution function"""
    print("="*60)
    print("20 SMA MEAN REVERSION MACHINE LEARNING ANALYSIS")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SMAMeanReversionAnalyzer('stock_data/XAUUSDm_M5_1year_data.csv', lookahead_candles=30)
    
    # Load and prepare data
    analyzer.load_and_prepare_data()
    
    # Detect crossover events
    analyzer.detect_crossover_events()
    
    # Analyze reversion events
    analyzer.analyze_reversion_events()
    
    # Prepare ML features
    X, y_reversion, y_candles, y_adverse_move, features = analyzer.prepare_ml_features()
    
    # Train models
    rf_classifier, xgb_regressor, xgb_adverse, scaler = analyzer.train_models(
        X, y_reversion, y_candles, y_adverse_move
    )
    
    # Create visualizations
    analyzer.create_visualizations()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Check the generated files for detailed results:")
    print("  - sma_reversion_analysis.csv")
    print("  - sma_reversion_summary.csv")
    print("  - sma_reversion_analysis.png")

if __name__ == "__main__":
    main() 