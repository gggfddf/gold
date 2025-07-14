//+------------------------------------------------------------------+
//|                                              SMA_Reversion_EA.mq5 |
//|                                  Copyright 2024, AI Assistant     |
//|                                             https://www.example.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, AI Assistant"
#property link      "https://www.example.com"
#property version   "1.00"
#property description "SMA Reversion Strategy EA - Based on Pine Script"
#property description "Enter LONG when price is below SMA by specified distance"
#property description "Hold for fixed number of candles then exit"

//--- Input Parameters
input int      SMA_Length = 50;           // SMA Period
input int      Hold_Candles = 12;         // Hold Time (candles)
input double   Entry_Distance = 0.0032;   // Entry Distance (0.32% = 0.0032)
input double   Lot_Size = 0.1;            // Lot Size
input int      Magic_Number = 12345;      // Magic Number
input string   Comment_Text = "SMA Rev";  // Trade Comment

//--- Global Variables
int sma_handle;
double sma_buffer[];
bool position_open = false;
datetime last_trade_time = 0;
int entry_bar_index = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize SMA indicator
   sma_handle = iMA(_Symbol, PERIOD_CURRENT, SMA_Length, 0, MODE_SMA, PRICE_CLOSE);
   
   if(sma_handle == INVALID_HANDLE)
   {
      Print("Error creating SMA indicator handle");
      return(INIT_FAILED);
   }
   
   // Allocate array for SMA values
   ArraySetAsSeries(sma_buffer, true);
   
   Print("SMA Reversion EA initialized successfully");
   Print("SMA Length: ", SMA_Length);
   Print("Hold Candles: ", Hold_Candles);
   Print("Entry Distance: ", Entry_Distance * 100, "%");
   Print("Lot Size: ", Lot_Size);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handle
   if(sma_handle != INVALID_HANDLE)
      IndicatorRelease(sma_handle);
   
   Print("SMA Reversion EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Check if we have a position open
   if(PositionSelect(_Symbol))
   {
      // Check if it's our position
      if(PositionGetInteger(POSITION_MAGIC) == Magic_Number)
      {
         position_open = true;
         
         // Check exit condition (hold time)
         int current_bar = iBarShift(_Symbol, PERIOD_CURRENT, TimeCurrent());
         if(current_bar - entry_bar_index >= Hold_Candles)
         {
            ClosePosition();
         }
      }
      else
      {
         position_open = false;
      }
   }
   else
   {
      position_open = false;
   }
   
   // Check for new entry if no position is open
   if(!position_open)
   {
      CheckForEntry();
   }
}

//+------------------------------------------------------------------+
//| Check for entry conditions                                       |
//+------------------------------------------------------------------+
void CheckForEntry()
{
   // Get current price and SMA
   double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Copy SMA values
   if(CopyBuffer(sma_handle, 0, 0, 2, sma_buffer) < 2)
   {
      Print("Error copying SMA buffer");
      return;
   }
   
   double current_sma = sma_buffer[0];
   
   // Calculate distance from SMA
   double distance_from_sma = (current_sma - current_price) / current_sma;
   
   // Check entry condition: price below SMA by specified distance
   if(distance_from_sma >= Entry_Distance)
   {
      // Additional check: ensure we don't trade too frequently
      if(TimeCurrent() - last_trade_time > PeriodSeconds() * 5) // Wait at least 5 candles
      {
         OpenLongPosition();
      }
   }
}

//+------------------------------------------------------------------+
//| Open long position                                               |
//+------------------------------------------------------------------+
void OpenLongPosition()
{
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = Lot_Size;
   request.type = ORDER_TYPE_BUY;
   request.price = ask;
   request.deviation = 10;
   request.magic = Magic_Number;
   request.comment = Comment_Text;
   request.type_filling = ORDER_FILLING_FOK;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Long position opened successfully");
         Print("Entry Price: ", result.price);
         Print("Lot Size: ", Lot_Size);
         
         position_open = true;
         last_trade_time = TimeCurrent();
         entry_bar_index = iBarShift(_Symbol, PERIOD_CURRENT, TimeCurrent());
         
         // Log trade details
         LogTrade("OPEN", "LONG", result.price, Lot_Size, "Entry Distance: " + DoubleToString((SymbolInfoDouble(_Symbol, SYMBOL_BID) - sma_buffer[0]) / sma_buffer[0] * 100, 2) + "%");
      }
      else
      {
         Print("Error opening position: ", result.retcode);
      }
   }
   else
   {
      Print("Error sending order");
   }
}

//+------------------------------------------------------------------+
//| Close current position                                           |
//+------------------------------------------------------------------+
void ClosePosition()
{
   if(!PositionSelect(_Symbol))
      return;
   
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = PositionGetDouble(POSITION_VOLUME);
   request.type = ORDER_TYPE_SELL;
   request.price = bid;
   request.deviation = 10;
   request.magic = Magic_Number;
   request.comment = Comment_Text + " Exit";
   request.type_filling = ORDER_FILLING_FOK;
   
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("Position closed successfully");
         Print("Exit Price: ", result.price);
         
         position_open = false;
         
         // Calculate P&L
         double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
         double pnl = (result.price - entry_price) * PositionGetDouble(POSITION_VOLUME) * 100; // For gold (100 oz contract)
         
         // Log trade details
         LogTrade("CLOSE", "LONG", result.price, PositionGetDouble(POSITION_VOLUME), "P&L: $" + DoubleToString(pnl, 2));
      }
      else
      {
         Print("Error closing position: ", result.retcode);
      }
   }
   else
   {
      Print("Error sending close order");
   }
}

//+------------------------------------------------------------------+
//| Log trade information                                            |
//+------------------------------------------------------------------+
void LogTrade(string action, string direction, double price, double volume, string additional_info)
{
   string log_message = StringFormat("[%s] %s %s - Price: %.2f, Volume: %.2f, %s", 
                                    TimeToString(TimeCurrent()), action, direction, price, volume, additional_info);
   Print(log_message);
}

//+------------------------------------------------------------------+
//| Custom functions                                                 |
//+------------------------------------------------------------------+

//+------------------------------------------------------------------+
//| Get current SMA value                                            |
//+------------------------------------------------------------------+
double GetCurrentSMA()
{
   if(CopyBuffer(sma_handle, 0, 0, 1, sma_buffer) < 1)
   {
      Print("Error copying SMA buffer");
      return 0;
   }
   return sma_buffer[0];
}

//+------------------------------------------------------------------+
//| Get distance from SMA                                            |
//+------------------------------------------------------------------+
double GetDistanceFromSMA()
{
   double current_price = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double current_sma = GetCurrentSMA();
   
   if(current_sma == 0)
      return 0;
   
   return (current_sma - current_price) / current_sma;
}

//+------------------------------------------------------------------+
//| Expert Advisor Information Panel                                 |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   // This function can be used to add custom chart events if needed
}

//+------------------------------------------------------------------+
//| Custom indicator for visualization                               |
//+------------------------------------------------------------------+
void CreateInfoPanel()
{
   // Create text objects to display current status
   string panel_name = "SMA_Rev_Panel";
   
   // Remove existing panel
   ObjectDelete(0, panel_name);
   
   // Create new panel
   ObjectCreate(0, panel_name, OBJ_RECTANGLE_LABEL, 0, 0, 0);
   ObjectSetInteger(0, panel_name, OBJPROP_XDISTANCE, 10);
   ObjectSetInteger(0, panel_name, OBJPROP_YDISTANCE, 10);
   ObjectSetInteger(0, panel_name, OBJPROP_XSIZE, 200);
   ObjectSetInteger(0, panel_name, OBJPROP_YSIZE, 150);
   ObjectSetInteger(0, panel_name, OBJPROP_BGCOLOR, clrWhite);
   ObjectSetInteger(0, panel_name, OBJPROP_BORDER_COLOR, clrBlack);
   ObjectSetInteger(0, panel_name, OBJPROP_CORNER, CORNER_LEFT_UPPER);
   
   // Add text information
   string info_text = "SMA Reversion EA\n";
   info_text += "SMA: " + DoubleToString(GetCurrentSMA(), 2) + "\n";
   info_text += "Distance: " + DoubleToString(GetDistanceFromSMA() * 100, 2) + "%\n";
   info_text += "Position: " + (position_open ? "OPEN" : "CLOSED") + "\n";
   info_text += "Last Trade: " + TimeToString(last_trade_time);
   
   ObjectSetString(0, panel_name, OBJPROP_TEXT, info_text);
} 