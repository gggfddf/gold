import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample XAUUSD data
start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)
dates = pd.date_range(start_date, end_date, freq='D')

# Simulate realistic XAUUSD data
n_days = len(dates)
base_price = 1800  # Starting around $1800/oz

# Generate price data with realistic movements
price_changes = np.random.normal(0, 0.02, n_days)  # 2% daily volatility
price_changes[0] = 0  # Start at base price

# Apply some trends and volatility clustering
for i in range(1, n_days):
    # Add some momentum (trend following)
    momentum = price_changes[i-1] * 0.1
    # Add volatility clustering
    volatility = abs(price_changes[i-1]) * 0.5 + 0.01
    price_changes[i] = np.random.normal(momentum, volatility)

# Calculate cumulative price
log_prices = np.cumsum(price_changes)
prices = base_price * np.exp(log_prices)

# Generate OHLC data
data = []
for i, (date, price) in enumerate(zip(dates, prices)):
    # Generate realistic intraday movements
    daily_volatility = np.random.uniform(0.005, 0.03)  # 0.5% to 3% daily range
    
    open_price = price * (1 + np.random.normal(0, 0.002))
    close_price = price * (1 + np.random.normal(0, 0.002))
    
    # Ensure high/low make sense
    high_price = max(open_price, close_price) * (1 + daily_volatility/2)
    low_price = min(open_price, close_price) * (1 - daily_volatility/2)
    
    # Generate volume (higher volume on volatile days)
    base_volume = 50000
    volume = int(base_volume * (1 + daily_volatility * 10) * np.random.uniform(0.5, 2.0))
    
    data.append({
        'date': date.strftime('%Y-%m-%d'),
        'open': round(open_price, 2),
        'high': round(high_price, 2),
        'low': round(low_price, 2),
        'close': round(close_price, 2),
        'volume': volume
    })

# Create DataFrame
df = pd.DataFrame(data)

# Add some realistic market behavior
# Weekend gaps (simulate Sunday night opening gaps)
for i in range(1, len(df)):
    if pd.to_datetime(df.iloc[i]['date']).weekday() == 0:  # Monday
        gap_size = np.random.normal(0, 0.01)  # 1% average gap
        df.iloc[i, df.columns.get_loc('open')] *= (1 + gap_size)

# Ensure OHLC relationships are valid
for i in range(len(df)):
    open_val = df.iloc[i]['open']
    close_val = df.iloc[i]['close']
    high_val = max(df.iloc[i]['high'], open_val, close_val)
    low_val = min(df.iloc[i]['low'], open_val, close_val)
    
    df.iloc[i, df.columns.get_loc('high')] = round(high_val, 2)
    df.iloc[i, df.columns.get_loc('low')] = round(low_val, 2)

# Save the sample data
df.to_csv('XAU_1d_data_clean.csv', index=False)
print(f"Generated {len(df)} days of sample XAUUSD data")
print("Sample data preview:")
print(df.head())
print("\nData saved to 'XAU_1d_data_clean.csv'")