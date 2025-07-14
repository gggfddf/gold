# MetaTrader 5 Data Downloader - Usage Guide

## Overview

This project provides tools to download historical financial data from MetaTrader 5 (MT5) and save it as CSV files for analysis. It includes both a Jupyter notebook and a standalone Python script.

## Files Overview

### Core Files
- **`Program_to_download_stock_data_from_MT5.ipynb`** - Jupyter notebook version
- **`mt5_data_downloader.py`** - Standalone Python script version
- **`requirements.txt`** - Python dependencies
- **`environment.env.example.txt`** - Template for environment variables

### Configuration Files
- **`.gitignore`** - Excludes sensitive files from version control
- **`README.md`** - Basic project documentation

### Output
- **`stock_data/`** - Directory where downloaded CSV files are saved

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install MetaTrader 5

Download and install MetaTrader 5 from the official website:
https://www.metatrader5.com/en/download

### 3. Set Up Environment Variables

1. Copy the example file:
   ```bash
   cp environment.env.example.txt environment.env
   ```

2. Edit `environment.env` with your MT5 credentials:
   ```
   MT5_ACCOUNT_NUMBER_DEMO="YOUR_ACCOUNT_NUMBER"
   MT5_PASSWORD_DEMO="YOUR_PASSWORD"
   MT5_SERVER_DEMO="YOUR_SERVER_NAME"
   ```

### 4. Configure MT5 Path

Update the `mt5_path` variable in the script to match your MT5 installation:
```python
mt5_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
```

## Usage

### Option 1: Jupyter Notebook

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `Program_to_download_stock_data_from_MT5.ipynb`

3. Run cells sequentially to:
   - Initialize MT5 connection
   - Download data for configured symbols
   - Generate visualizations
   - Save data to CSV files

### Option 2: Python Script

#### Basic Usage
```bash
python mt5_data_downloader.py
```

#### Advanced Usage with Command Line Arguments
```bash
# Download specific symbols
python mt5_data_downloader.py --symbols EURUSD GBPUSD USDJPY

# Set custom start year
python mt5_data_downloader.py --start-year 2019

# Custom output folder
python mt5_data_downloader.py --output-folder my_data

# Custom MT5 path
python mt5_data_downloader.py --mt5-path "C:\\Custom\\Path\\terminal64.exe"

# Combine multiple options
python mt5_data_downloader.py --symbols EURUSD GBPUSD --start-year 2021 --output-folder forex_data
```

## Configuration Options

### Default Settings
- **Symbols**: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD
- **Timeframe**: Daily (D1)
- **Start Year**: 2020
- **Output Folder**: stock_data

### Available Timeframes
```python
mt5.TIMEFRAME_M1    # 1 minute
mt5.TIMEFRAME_M5    # 5 minutes
mt5.TIMEFRAME_M15   # 15 minutes
mt5.TIMEFRAME_M30   # 30 minutes
mt5.TIMEFRAME_H1    # 1 hour
mt5.TIMEFRAME_H4    # 4 hours
mt5.TIMEFRAME_D1    # Daily
mt5.TIMEFRAME_W1    # Weekly
mt5.TIMEFRAME_MN1   # Monthly
```

### Popular Forex Symbols
- **Major Pairs**: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD
- **Minor Pairs**: EURGBP, EURJPY, GBPJPY, AUDNZD, CADCHF
- **Exotic Pairs**: USDTRY, EURTRY, USDZAR, EURZAR

## Output Data Format

Each downloaded symbol creates a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| time | Timestamp (datetime) |
| open | Opening price |
| high | Highest price |
| low | Lowest price |
| close | Closing price |
| tick_volume | Volume |
| spread | Spread |
| real_volume | Real volume |
| Returns | Price returns (calculated) |
| Volatility | 20-period volatility (calculated) |
| SMA_20 | 20-period simple moving average (calculated) |
| SMA_50 | 50-period simple moving average (calculated) |

## Troubleshooting

### Common Issues

1. **MT5 Initialization Failed**
   - Verify MT5 is installed and running
   - Check the `mt5_path` is correct
   - Ensure MT5 terminal is not blocked by firewall

2. **Login Failed**
   - Verify account credentials in `environment.env`
   - Check if account is active and not suspended
   - Ensure server name is correct

3. **No Data Available**
   - Check if symbol is available in your MT5 account
   - Verify the date range is within available data
   - Some symbols may have limited historical data

4. **Permission Errors**
   - Run as administrator if needed
   - Check folder permissions for output directory

### Error Messages

- **"MT5 initialization failed"**: Check MT5 installation and path
- **"MT5 login failed"**: Verify credentials and account status
- **"No data available for [symbol]"**: Symbol may not be available or date range invalid

## Data Analysis Examples

### Load and Analyze Data
```python
import pandas as pd

# Load downloaded data
df = pd.read_csv('stock_data/EURUSD_data.csv')
df['time'] = pd.to_datetime(df['time'])

# Basic statistics
print(df.describe())

# Plot price chart
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['close'])
plt.title('EURUSD Price Chart')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
```

### Calculate Technical Indicators
```python
# RSI
def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['RSI'] = calculate_rsi(df['close'])

# Bollinger Bands
df['BB_upper'] = df['SMA_20'] + (df['close'].rolling(20).std() * 2)
df['BB_lower'] = df['SMA_20'] - (df['close'].rolling(20).std() * 2)
```

## Security Notes

- Never commit `environment.env` to version control
- Keep your MT5 credentials secure
- Use demo accounts for testing
- Be aware of data usage limits from your broker

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your MT5 account has access to the requested symbols
3. Ensure your broker provides historical data for the requested timeframe

## License

This project is open source. Feel free to modify and distribute according to your needs. 