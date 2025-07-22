//+------------------------------------------------------------------+
//|                                                preprocessing.mqh |
//|                                    Copyright 2022, Fxalgebra.com |
//|                        https://www.mql5.com/en/users/omegajoctan |
//+------------------------------------------------------------------+
#property copyright "Copyright 2022, Fxalgebra.com"
#property link      "https://www.mql5.com/en/users/omegajoctan"


//+------------------------------------------------------------------+
//|                                                                  |
//|                                                                  |
//|               Standardization Scaler                             |
//|                                                                  |
//|                                                                  |
//+------------------------------------------------------------------+

class StandardizationScaler
  {
protected:
   vector mean, std;
   bool loaded_scaler;
   
   bool write_bin(vector &v,string file)
    {
      FileDelete(file);
      int handle = FileOpen(file,FILE_READ|FILE_WRITE|FILE_BIN,",");
      if (handle == INVALID_HANDLE)
       {
         printf("Invalid handle Err=%d",GetLastError());
         DebugBreak();
         return false;
       }
      
      double arr[];
      ArrayResize(arr, (int)v.Size());
      
      for (uint i=0; i<arr.Size(); i++)
       arr[i] = v[i];
      
      FileWriteArray(handle, arr);
      FileClose(handle);
     
     return true;
    }
    
   template<typename T>
   vector ArrayToVector(const T &Arr[])
     {
      vector v(ArraySize(Arr));
      
      for (int i=0; i<ArraySize(Arr); i++)
        v[i] = double(Arr[i]);
        
      return (v);
     }
 
public:
                     StandardizationScaler(void);
                     StandardizationScaler(const double &mean[], const double &std[]); //For Loading the pre-fitted scaler 
                    ~StandardizationScaler(void);
                    
                    virtual matrix fit_transform(const matrix &X);
                    virtual matrix transform(const matrix &X);
                    virtual vector transform(const vector &X);
                    
                    virtual bool   save(string save_dir);
                    
                    
                    virtual matrix inverse_transform(const matrix &X_scaled);
                    virtual vector inverse_transform(const vector &X_scaled);
  };
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::StandardizationScaler(void)
 {
   loaded_scaler = false;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::StandardizationScaler(const double &mean_[],const double &std_[])
 {
   this.mean = ArrayToVector(mean_);
   this.std = ArrayToVector(std_);
   
   loaded_scaler = true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
StandardizationScaler::~StandardizationScaler(void)
 {
 
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix StandardizationScaler::fit_transform(const matrix &X)
 { 
  
  if (loaded_scaler)
    {
      printf("% This is a loaded scaler | no need to fit to the new data, call another instance of a class",__FUNCTION__);
      return X;
    }
  
  this.mean.Resize(X.Cols());
  this.std.Resize(X.Cols());
  
    for (ulong i=0; i<X.Cols(); i++)
      { 
         this.mean[i] = X.Col(i).Mean();
         this.std[i] = X.Col(i).Std();
      }

//---
   return this.transform(X);
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector StandardizationScaler::inverse_transform(const vector &X_scaled)
 {
    vector X(X_scaled.Size());

    if (this.mean.Size() == 0 || this.std.Size() == 0) {
        printf("%s Call the fit_transform function first to fit the scaler or\n Load the pre-fitted scaler before attempting to transform the new data", __FUNCTION__);
        return X;
    }

    if (X_scaled.Size() != this.mean.Size()) {
        printf("%s Dimension mismatch between trained data sized=(%d) and the new data sized=(%d)", __FUNCTION__, this.mean.Size(), X_scaled.Size());
        return X;
    }

    for (ulong i = 0; i < X.Size(); i++) {
        X[i] = X_scaled[i] * (this.std[i] + 1e-10) + this.mean[i];
    }

    return X;
}
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
matrix StandardizationScaler::transform(const matrix &X)
 {
   matrix X_norm = X;
   
   for (ulong i=0; i<X.Rows(); i++)
     X_norm.Row(this.transform(X.Row(i)), i);
   
   return X_norm;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+

matrix StandardizationScaler::inverse_transform(const matrix &X_scaled)
 {
   matrix X = X_scaled;
   
   for (ulong i=0; i<X.Rows(); i++)
     X.Row(this.inverse_transform(X_scaled.Row(i)), i);
   
   return X;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
vector StandardizationScaler::transform(const vector &X)
 {
   vector v(X.Size());
   if (this.mean.Size()==0 || this.std.Size()==0)
     {
       printf("%s Call the fit_transform function first to fit the scaler or\n Load the pre-fitted scaler before attempting to transform the new data",__FUNCTION__);
       return v;
     }
   
   if (X.Size() != this.mean.Size())
     {
         printf("%s Dimension mismatch between trained data sized=(%d) and the new data sized=(%d)",__FUNCTION__,this.mean.Size(),X.Size());
         return v;
     }
   
   for (ulong i=0; i<v.Size(); i++)
      v[i] = (X[i] - this.mean[i]) / (this.std[i] + 1e-10);  
   
   return v;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool StandardizationScaler::save(string save_dir)
 {
//---save mean

   if (!write_bin(this.mean, save_dir+"\\mean.bin"))
     {
       printf("%s Failed Save the mean values of the Scaler",__FUNCTION__);
       return false;
     }
   
//--- save std

   if (!write_bin(this.std, save_dir+"\\std.bin"))
     {
       printf("%s Failed Save the Standard deviation values of the Scaler",__FUNCTION__);
       return false;
     }
     
   return true;
 }
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
 