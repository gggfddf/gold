import os
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

if not mt5.initialize(path=MT5_PATH):
    print(f"MT5 initialization failed: {mt5.last_error()}")
    exit(1)

if not mt5.login(login=ACCOUNT_NUMBER_INT, password=PASSWORD, server=SERVER):
    print(f"MT5 login failed: {mt5.last_error()}")
    mt5.shutdown()
    exit(1)

symbols = mt5.symbols_get()
if symbols is None or len(symbols) == 0:
    print("No symbols found in your account.")
    mt5.shutdown()
    exit(1)

with open("available_symbols.txt", "w", encoding="utf-8") as f:
    for sym in symbols:
        f.write(f"{sym.name}\n")

print(f"Found {len(symbols)} symbols. List saved to available_symbols.txt.")
mt5.shutdown() 