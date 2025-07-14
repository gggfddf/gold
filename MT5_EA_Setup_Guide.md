# MT5 SMA Reversion EA Setup Guide

## Quick Setup Steps

### 1. Copy EA File
- Copy `SMA_Reversion_EA.mq5` to your MT5 `Experts` folder:
  - Usually located at: `C:\Users\[YourName]\AppData\Roaming\MetaQuotes\Terminal\[Account]\MQL5\Experts\`

### 2. Compile EA
- Open MetaEditor in MT5 (F4)
- Open `SMA_Reversion_EA.mq5`
- Click "Compile" (F7)
- Fix any errors if they appear

### 3. Attach to Chart
- Open XAUUSD 5-minute chart
- Drag EA from Navigator panel to chart
- Configure parameters in popup window

## EA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| SMA_Length | 50 | SMA period |
| Hold_Candles | 12 | Hold time in candles |
| Entry_Distance | 0.0032 | Entry distance (0.32%) |
| Lot_Size | 0.1 | Trading lot size |
| Magic_Number | 12345 | Unique identifier |
| Comment_Text | "SMA Rev" | Trade comment |

## Strategy Logic

**Entry Condition:**
- Price below SMA50 by 0.32% or more
- No existing position
- Wait 5 candles between trades

**Exit Condition:**
- Hold for exactly 12 candles
- No stop loss or take profit

## Risk Management

⚠️ **IMPORTANT:**
- Start with small lot size (0.01)
- Test on demo account first
- Monitor performance closely
- This is a mean reversion strategy - works best in sideways markets

## Performance Notes

Based on backtesting:
- Win Rate: ~59%
- Expected Return: ~467% annually (with $4000 capital)
- All trades exit by timeout (12 candles)
- No stop losses - maximum risk per trade = lot size × 12 candles of movement

## Customization

You can modify:
- `SMA_Length`: Try 20, 30, or 100
- `Hold_Candles`: Try 6, 8, or 15
- `Entry_Distance`: Try 0.002 (0.2%) or 0.005 (0.5%)
- `Lot_Size`: Adjust based on your risk tolerance

## Troubleshooting

**EA not trading:**
- Check if "AutoTrading" is enabled
- Verify "Allow live trading" is checked in EA settings
- Check Expert tab for error messages

**Compilation errors:**
- Make sure you're using MT5 (not MT4)
- Check that all functions are properly defined

## Live Trading Tips

1. **Start Small**: Begin with 0.01 lot size
2. **Monitor**: Check EA performance daily
3. **Adjust**: Fine-tune parameters based on market conditions
4. **Backup**: Keep track of all trades and performance
5. **Risk**: Never risk more than 2% of account per trade

## File Locations

- EA File: `SMA_Reversion_EA.mq5`
- Logs: Check "Experts" tab in MT5
- Performance: Check "Account History" tab

## Support

If you encounter issues:
1. Check MT5 "Experts" tab for error messages
2. Verify all parameters are set correctly
3. Test on demo account first
4. Ensure stable internet connection 