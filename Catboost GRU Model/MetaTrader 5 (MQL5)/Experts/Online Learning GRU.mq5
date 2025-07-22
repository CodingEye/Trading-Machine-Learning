//+------------------------------------------------------------------+
//|                                          Online Learning GRU.mq5 |
//|                                    Copyright 2024, Omega Joctan. |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Omega Joctan."
#property link      "https://www.mql5.com/en/users/omegajoctan"
#property version   "1.00"

#include <preprocessing.mqh>
#include <GRU.mqh>

CGRU *gru;
StandardizationScaler *scaler;

//--- Arrays for temporary storage of the scaler values
double scaler_mean[], scaler_std[];

input string model_name = "gru.H1.onnx";
input int time_step = 10;
input string symbol = "EURUSD";
input ENUM_TIMEFRAMES timeframe = PERIOD_H1;

string mean_file;
string std_file;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   
   string base_name__ = model_name;
   
   if (StringReplace(base_name__,".onnx","")<0) //we followed this same file patterns while saving the binary files in python client
     {
       printf("%s Failed to obtain the parent name for the scaler files, error = %d",__FUNCTION__,GetLastError());
       return INIT_FAILED;
     }
    
    
   mean_file = base_name__ + ".standard_scaler_mean.bin";
   std_file = base_name__ + ".standard_scaler_scale.bin";
   
//--- Check if the model file exists

   if (!FileIsExist(model_name, FILE_COMMON))
     {
       printf("%s Onnx file doesn't exist",__FUNCTION__);
       return INIT_FAILED;
     }
  
//--- Initialize the GRU model from the common folder

     gru = new CGRU(); 
     if (!gru.Init(model_name, ONNX_COMMON_FOLDER))
       {
         printf("%s failed to initialize the gru model, error = %d",__FUNCTION__,GetLastError());      
         return INIT_FAILED;
       }

//--- Read the scaler files
   
   if (!readArray(mean_file, scaler_mean) || !readArray(std_file, scaler_std))
     {
       printf("%s failed to read scaler information",__FUNCTION__);
       return INIT_FAILED;
     }  
   
   
   scaler = new StandardizationScaler(scaler_mean, scaler_std); //Load the scaler class by populating it with values
   
//--- Set the timer

   if (!EventSetTimer(60))
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
   
    if (CheckPointer(gru) != POINTER_INVALID)
      delete gru;
    if (CheckPointer(scaler) != POINTER_INVALID)
      delete scaler;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
     MqlRates rates[];
     CopyRates(symbol, timeframe, 1, time_step, rates); //copy the recent closed bar information
     
     vector classes = {0,1}; //Beware of how classes are organized in the target variable. use numpy.unique(y) to determine this array
     
     matrix X = matrix::Zeros(time_step, 6); // 6 columns
     for (int i=0; i<time_step; i++)
       {         
         vector row = {
                 (double)rates[i].time, 
                 rates[i].open, 
                 rates[i].high, 
                 rates[i].low, 
                 rates[i].close, 
                 (double)rates[i].tick_volume};
         
         X.Row(row, i);
       }     
     
     X = scaler.transform(X); //it's important to normalize the data  
     Comment(TimeCurrent(),"\nPredicted signal: ",gru.predict_bin(X, classes)==0?"Bearish":"Bullish");// if the predicted signal is 0 it means a bearish signal, otherwise it is a bullish signal
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnTimer(void)
  {
//--- Delete the existing pointers in memory as the new ones are about to be created

    if (CheckPointer(gru) != POINTER_INVALID)
      delete gru;
    if (CheckPointer(scaler) != POINTER_INVALID)
      delete scaler;
      
//---
   
   
   if (!readArray(mean_file, scaler_mean) || !readArray(std_file, scaler_std))
     {
       printf("%s failed to read scaler information",__FUNCTION__);
       return;
     }  
   
   
   scaler = new StandardizationScaler(scaler_mean, scaler_std);
   
     gru = new CGRU(); 
     if (!gru.Init(model_name, ONNX_COMMON_FOLDER))
       {
         printf("%s failed to initialize the gru model, error = %d",__FUNCTION__,GetLastError());      
         return;
         
       }
     printf("%s New model loaded",TimeToString(TimeCurrent(), TIME_DATE|TIME_MINUTES));
  }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool readArray(string file_name, double &Array[])
 {
   int handle = FileOpen(file_name, FILE_SHARE_READ|FILE_COMMON|FILE_BIN); //File share read could be helpful for both Python client and metatrader 5 to read and write simultaneously
   if (handle == INVALID_HANDLE)
     {
       printf("Failed to read a binary file %s, error = %d",file_name,GetLastError());
       return false;
     }
   
   if (FileReadArray(handle, Array)==0)
     {
       printf("Binary file has zero contents");
     }
     
   FileClose(handle);
   
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+


