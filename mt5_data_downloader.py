#!/usr/bin/env python3
"""
MetaTrader 5 Data Downloader Script

This script connects to a MetaTrader 5 account and downloads historical price data
for specified symbols and timeframes, saving it into CSV files.

Author: Vsktech31
Repository: https://github.com/Vsktech31/1_Program_to_download_stock_data_from_MT5
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import warnings
import argparse
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

def load_configuration():
    """Load configuration from environment variables"""
    load_dotenv('environment.env')
    
    # Configuration variables
    config = {
        'symbols': ["XAUUSD"],  # Set to XAUUSD for gold
        'timeframe': mt5.TIMEFRAME_D1,  # Daily timeframe
        'start_year': 2004,  # 20 years ago
        'output_folder': "stock_data",
        'mt5_path': r"C:\Program Files\MetaTrader 5\terminal64.exe"
    }
    
    # Account credentials from environment variables
    config['account_number'] = os.getenv('MT5_ACCOUNT_NUMBER_DEMO')
    config['password'] = os.getenv('MT5_PASSWORD_DEMO')
    config['server'] = os.getenv('MT5_SERVER_DEMO')
    
    return config

def initialize_mt5(config):
    """Initialize and connect to MetaTrader 5"""
    print("Initializing MetaTrader 5...")
    
    if not mt5.initialize(path=config['mt5_path']):
        print(f"MT5 initialization failed: {mt5.last_error()}")
        return False
    
    # Login to account
    if not mt5.login(login=int(config['account_number']), 
                    password=config['password'], 
                    server=config['server']):
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    print("Successfully connected to MetaTrader 5")
    return True

def download_symbol_data(symbol, timeframe, start_date, end_date):
    """Download historical data for a specific symbol"""
    try:
        print(f"Downloading data for {symbol}...")
        
        # Get historical data
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"No data available for {symbol}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add additional calculated columns
        df['Returns'] = df['close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        return df
        
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return None

def save_data_to_csv(df, symbol, output_folder):
    """Save DataFrame to CSV file"""
    # Save with symbol and timeframe in filename for clarity
    filename = f"{output_folder}/{symbol}_D1_20years_data.csv"
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")
    return filename

def create_visualization(downloaded_data, output_folder):
    """Create and save price charts"""
    if not downloaded_data:
        return
    
    # Create subplots for each symbol
    fig, axes = plt.subplots(len(downloaded_data), 1, 
                            figsize=(15, 5*len(downloaded_data)))
    
    if len(downloaded_data) == 1:
        axes = [axes]
    
    for i, (symbol, df) in enumerate(downloaded_data.items()):
        if df is not None:
            axes[i].plot(df['time'], df['close'], label='Close Price', linewidth=1)
            axes[i].plot(df['time'], df['SMA_20'], label='SMA 20', alpha=0.7)
            axes[i].plot(df['time'], df['SMA_50'], label='SMA 50', alpha=0.7)
            
            axes[i].set_title(f'{symbol} Price Chart')
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Price')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{output_folder}/price_charts.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Price charts saved to: {plot_filename}")
    plt.show()

def print_summary(downloaded_data):
    """Print summary statistics"""
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    
    for symbol, df in downloaded_data.items():
        if df is not None:
            print(f"\n{symbol}:")
            print(f"  Records: {len(df)}")
            print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
            print(f"  Price range: ${df['close'].min():.5f} - ${df['close'].max():.5f}")
            print(f"  Average volume: {df['tick_volume'].mean():.0f}")
            print(f"  Average return: {df['Returns'].mean():.4f}")
            print(f"  Volatility: {df['Volatility'].mean():.4f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Download stock data from MetaTrader 5')
    parser.add_argument('--symbols', nargs='+', help='Symbols to download')
    parser.add_argument('--start-year', type=int, help='Start year for data')
    parser.add_argument('--output-folder', help='Output folder for data')
    parser.add_argument('--mt5-path', help='Path to MT5 terminal64.exe')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_configuration()
    
    # Override with command line arguments if provided
    if args.symbols:
        config['symbols'] = args.symbols
    if args.start_year:
        config['start_year'] = args.start_year
    if args.output_folder:
        config['output_folder'] = args.output_folder
    if args.mt5_path:
        config['mt5_path'] = args.mt5_path
    
    # Validate configuration
    if not all([config['account_number'], config['password'], config['server']]):
        print("Error: Please set up your MT5 credentials in environment.env file")
        print("Copy environment.env.example.txt to environment.env and fill in your details")
        sys.exit(1)
    
    print("Configuration:")
    print(f"  Symbols: {config['symbols']}")
    print(f"  Timeframe: {config['timeframe']}")
    print(f"  Start year: {config['start_year']}")
    print(f"  Output folder: {config['output_folder']}")
    print(f"  MT5 path: {config['mt5_path']}")
    
    # Create output directory
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])
        print(f"Created directory: {config['output_folder']}")
    
    # Initialize MT5
    if not initialize_mt5(config):
        print("Failed to initialize MT5. Please check your credentials and MT5 installation.")
        sys.exit(1)
    
    # Set date range
    start_date = datetime(config['start_year'], 1, 1)
    end_date = datetime.now()
    
    print(f"\nDownloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Download data for all symbols
    downloaded_data = {}
    
    for symbol in config['symbols']:
        df = download_symbol_data(symbol, config['timeframe'], start_date, end_date)
        
        if df is not None:
            save_data_to_csv(df, symbol, config['output_folder'])
            downloaded_data[symbol] = df
        else:
            print(f"Failed to download data for {symbol}")
    
    # Print summary
    print_summary(downloaded_data)
    
    # Create visualization
    create_visualization(downloaded_data, config['output_folder'])
    
    # Shutdown MT5
    mt5.shutdown()
    print("\nMetaTrader 5 connection closed.")
    print(f"All data has been saved to the '{config['output_folder']}' folder.")

if __name__ == "__main__":
    main() 