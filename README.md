# MetaTrader 5 Data Downloader

This script connects to a MetaTrader 5 account and downloads historical price data for specified symbols and timeframes, saving it into CSV files.

## Prerequisites

* Python 3.x
* MetaTrader 5 Terminal installed

## Setup

1. **Clone the repository (or download the files):**  
   git clone https://github.com/Vsktech31/1_Program_to_download_stock_data_from_MT5.git  
   cd 1_Program_to_download_stock_data_from_MT5

2. **Install dependencies:**  
   pip install -r requirements.txt

3. **Set up environment variables:**  
   Copy `environment.env.example` to `environment.env`:  
   cp environment.env.example environment.env  
   Then, edit `environment.env` and fill in your actual MetaTrader 5 account details.  
   MT5_ACCOUNT_NUMBER_DEMO="YOUR_REAL_ACCOUNT_NUMBER"  
   MT5_PASSWORD_DEMO="YOUR_REAL_PASSWORD"  
   MT5_SERVER_DEMO="YOUR_REAL_SERVER_NAME"

4. **Configure the script:**  
   Open the Python script (e.g., `Program_to_download_stock_data_from_MT5.ipynb`) and adjust the following variables if needed:  
   * `SYMBOLS_TO_DOWNLOAD`  
   * `REQUEST_TIMEFRAME`  
   * `START_YEAR_DATA`  
   * `OUTPUT_DATA_FOLDER`  
   * `mt5_path` (path to your `terminal64.exe`)

## Running the Script

Run the codes in the file - Program_to_download_stock_data_from_MT5.ipynb

## About

Program_to_download_stock_data_from_MT5 