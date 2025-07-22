//+------------------------------------------------------------------+
//|                                     Online Learning Catboost.mq5 |
//|                                    Copyright 2024, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"

#include <CatBoost.mqh>
CCatBoost *catboost;

input string model_name = "catboost.H1.onnx";
input string symbol = "EURUSD";
input ENUM_TIMEFRAMES timeframe = PERIOD_H1;
string common_path;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
//--- Check if the model file exists
  
   if (!FileIsExist(model_name, FILE_COMMON))
     {
       printf("%s Onnx file doesn't exist",__FUNCTION__);
       return INIT_FAILED;
     }
     
//--- Initialize a catboost model
   
  catboost = new CCatBoost(); 
  if (!catboost.Init(model_name, ONNX_COMMON_FOLDER))
    {
      printf("%s failed to initialize the catboost model, error = %d",__FUNCTION__,GetLastError());      
      return INIT_FAILED;
    }
      
//---

   if (!EventSetTimer(60)) //Execute the OnTimer function after every 60 seconds
     {
       printf("%s failed to set the event timer, error = %d",__FUNCTION__,GetLastError());
       return INIT_FAILED;
     }
    
    
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
    if (CheckPointer(catboost) != POINTER_INVALID)
      delete catboost;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
     MqlRates rates[];
     CopyRates(symbol, timeframe, 1, 1, rates); //copy the recent closed bar information
     
     vector x = {
                 (double)rates[0].time, 
                 rates[0].open, 
                 rates[0].high, 
                 rates[0].low, 
                 rates[0].close, 
                 (double)rates[0].tick_volume};
                 
     
     
     Comment(TimeCurrent(),"\nPredicted signal: ",catboost.predict_bin(x)==0?"Bearish":"Bullish");// if the predicted signal is 0 it means a bearish signal, otherwise it is a bullish signal
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
    if (CheckPointer(catboost) != POINTER_INVALID)
      delete catboost;
      
//---

     catboost = new CCatBoost(); 
     if (!catboost.Init(model_name, ONNX_COMMON_FOLDER))
       {
         printf("%s failed to initialize the catboost model, error = %d",__FUNCTION__,GetLastError());      
         return;
       }
       
     printf("%s New model loaded",TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
