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
MT5_PATH = r"C:\Program Files\MetaTrader\terminal64.exe"  # Updated path for your MT5 installation

if not all([ACCOUNT_NUMBER, PASSWORD, SERVER]):
    print("Error: Please set your MT5 credentials in environment.env.")
    exit(1)

try:
    ACCOUNT_NUMBER_INT = int(ACCOUNT_NUMBER)
except Exception:
    print("Error: ACCOUNT_NUMBER must be an integer.")
    exit(1)

SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5  # 5-minute timeframe
# Get 1 year of data from today
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
OUTPUT_FOLDER = "stock_data"

# Initialize MT5
try:
    initialized = mt5.initialize(path=MT5_PATH)
except AttributeError:
    print("Error: MetaTrader5 module does not have 'initialize'. Check your installation.")
    exit(1)

if not initialized:
    try:
        print(f"MT5 initialization failed: {mt5.last_error()}")
    except AttributeError:
        print("MT5 initialization failed: Unknown error (no last_error attribute)")
    exit(1)

try:
    login_result = mt5.login(login=ACCOUNT_NUMBER_INT, password=PASSWORD, server=SERVER)
except Exception as e:
    print(f"MT5 login error: {e}")
    try:
        mt5.shutdown()
    except AttributeError:
        pass
    exit(1)

if not login_result:
    try:
        print(f"MT5 login failed: {mt5.last_error()}")
    except AttributeError:
        print("MT5 login failed: Unknown error (no last_error attribute)")
    try:
        mt5.shutdown()
    except AttributeError:
        pass
    exit(1)

print(f"Connected to MT5: {SERVER} as {ACCOUNT_NUMBER}")

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

print(f"Downloading GOLD data ({SYMBOL}) from {start_date.date()} to {end_date.date()}...")
try:
    rates = mt5.copy_rates_range(SYMBOL, TIMEFRAME, start_date, end_date)
except AttributeError:
    print("Error: MetaTrader5 module does not have 'copy_rates_range'. Check your installation.")
    try:
        mt5.shutdown()
    except AttributeError:
        pass
    exit(1)

if rates is None or len(rates) == 0:
    print(f"No data available for {SYMBOL}")
    try:
        mt5.shutdown()
    except AttributeError:
        pass
    exit(1)

# Convert to DataFrame
df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')

# Save to CSV
csv_path = os.path.join(OUTPUT_FOLDER, f"{SYMBOL}_data.csv")
df.to_csv(csv_path, index=False)
print(f"Saved GOLD data to {csv_path}")

try:
    mt5.shutdown()
except AttributeError:
    pass
print("MT5 connection closed.") 