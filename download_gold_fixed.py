import os
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv

try:
    import MetaTrader5 as mt5
except ImportError:
    print("MetaTrader5 package is not installed. Please install it with 'pip install MetaTrader5'.")
    exit(1)

# Load environment variables
load_dotenv('environment.env')

ACCOUNT_NUMBER = os.getenv('MT5_ACCOUNT_NUMBER_DEMO')
PASSWORD = os.getenv('MT5_PASSWORD_DEMO')
SERVER = os.getenv('MT5_SERVER_DEMO')
MT5_PATH = r"C:\Program Files\MetaTrader\terminal64.exe"

if not all([ACCOUNT_NUMBER, PASSWORD, SERVER]):
    print("Error: Please set your MT5 credentials in environment.env.")
    exit(1)

try:
    ACCOUNT_NUMBER_INT = int(ACCOUNT_NUMBER)
except Exception:
    print("Error: ACCOUNT_NUMBER must be an integer.")
    exit(1)

# Initialize MT5
print("Initializing MetaTrader 5...")
if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 initialization failed: {mt5.last_error()}")
    exit(1)

# Login to MT5
print("Logging in to MT5...")
if not mt5.login(login=ACCOUNT_NUMBER_INT, password=PASSWORD, server=SERVER):
    print(f"MT5 login failed: {mt5.last_error()}")
    mt5.shutdown()
    exit(1)

print(f"Successfully connected to MT5: {SERVER} as {ACCOUNT_NUMBER}")

# Get account info
account_info = mt5.account_info()
if account_info is not None:
    print(f"Account: {account_info.login}, Server: {account_info.server}")

# Try different gold symbols
gold_symbols = ["XAUUSD", "XAUUSDm", "GOLD", "XAUUSD.a", "XAUUSD.b"]
available_symbol = None

for symbol in gold_symbols:
    print(f"Checking symbol: {symbol}")
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is not None and symbol_info.visible:
        print(f"Found available symbol: {symbol}")
        available_symbol = symbol
        break

if available_symbol is None:
    print("No gold symbol found. Trying to get all symbols...")
    symbols = mt5.symbols_get()
    if symbols:
        print("Available symbols (first 20):")
        for i, sym in enumerate(symbols[:20]):
            print(f"  {sym.name}")
        print("...")
    mt5.shutdown()
    exit(1)

# Set up parameters
SYMBOL = available_symbol
TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute timeframe
end_date = datetime.now()
start_date = end_date - timedelta(days=30)  # Try 30 days first
OUTPUT_FOLDER = "stock_data"

# Create output folder
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"Downloading {SYMBOL} data from {start_date.strftime('%Y-%m-%d %H:%M')} to {end_date.strftime('%Y-%m-%d %H:%M')}")

# Get historical data
rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)

if rates is None or len(rates) == 0:
    print(f"No data available for {SYMBOL}")
    print("Trying with a shorter time period...")
    start_date = end_date - timedelta(days=7)  # Try last 7 days
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)
    
    if rates is None or len(rates) == 0:
        print(f"Still no data available for {SYMBOL}")
        mt5.shutdown()
        exit(1)

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

print(f"Downloaded {len(df)} records for {SYMBOL}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")

# Save to CSV
csv_path = os.path.join(OUTPUT_FOLDER, f"{SYMBOL}_M5_data.csv")
df.to_csv(csv_path, index=False)
print(f"Saved data to: {csv_path}")

# Show sample data
print("\nSample data:")
print(df.head())

mt5.shutdown()
print("MT5 connection closed successfully.") 