import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import catboost_models
import os
import schedule
import time
import gru_models


if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()


# display info on the terminal settings and status
terminal_info=mt5.terminal_info()

if terminal_info==None:
    print("Failed to get the common datapath, error = ",mt5.last_error())
    mt5.shutdown()
    

terminal_info_dict = mt5.terminal_info()._asdict()
common_path = terminal_info_dict["commondata_path"]


# Getting the training data

def getData(start = 1, bars = 1000):

    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, start, bars)
    
    if len(rates) < bars: # if the received information is less than specified
        print("Failed to copy rates from MetaTrader 5, error = ",mt5.last_error())

    # create a pnadas DataFrame out of the obtained data

    df_rates = pd.DataFrame(rates)
                                                
    return df_rates



def trainAndSaveCatBoost():

    data = getData(start=1, bars=1000)

    # Check if we were able to receive some data

    if (len(data)<=0):
        print("Failed to obtain data from Metatrader5, error = ",mt5.last_error())
        mt5.shutdown()


    # Preparing the target variable

    data["future_open"] = data["open"].shift(-1) # shift one bar into the future
    data["future_close"] = data["close"].shift(-1)


    target = []
    for row in range(data.shape[0]):
        if data["future_close"].iloc[row] > data["future_open"].iloc[row]: # bullish signal
            target.append(1)
        else: # bearish signal
            target.append(0)

    data["target"] = target # add the target variable to the dataframe

    data = data.dropna() # drop empty rows


    X = data.drop(columns = ["spread","real_volume","future_close","future_open","target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

    catboost_model = catboost_models.CatBoostClassifierModel(X_train, X_test, y_train, y_test)
    catboost_model.train()

    # Save models in a specific location under the common parent folder

    models_path = os.path.join(common_path, "Files")

    if not os.path.exists(models_path): #if the folder exists
        os.makedirs(models_path) # Create the folder if it doesn't exist

    catboost_model.to_onnx(model_name=os.path.join(models_path, "catboost.H1.onnx"))



def create_sequences(X, Y, time_step):

    if len(X) != len(Y):
        raise ValueError("X and y must have the same length")
    
    X = np.array(X)
    Y = np.array(Y)
    
    Xs, Ys = [], []
    
    for i in range(X.shape[0] - time_step):
        Xs.append(X[i:(i + time_step), :])  # Include all features with slicing
        Ys.append(Y[i + time_step])
        
    return np.array(Xs), np.array(Ys)


def trainAndSaveGRU():

    data = getData(start=1, bars=1000)

    # Preparing the target variable

    data["future_open"] = data["open"].shift(-1)
    data["future_close"] = data["close"].shift(-1)


    target = []
    for row in range(data.shape[0]):
        if data["future_close"].iloc[row] > data["future_open"].iloc[row]:
            target.append(1)
        else:
            target.append(0)

    data["target"] = target

    data = data.dropna()

    # Check if we were able to receive some data

    if (len(data)<=0):
        print("Failed to obtain data from Metatrader5, error = ",mt5.last_error())
        mt5.shutdown()


    X = data.drop(columns = ["spread","real_volume","future_close","future_open","target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=False)

    ########### Preparing data for timeseries forecasting ###############

    time_step = 10 

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    x_train_seq, y_train_seq = create_sequences(X_train, y_train, time_step)
    x_test_seq, y_test_seq = create_sequences(X_test, y_test, time_step)

    ###### One HOt encoding #######

    y_train_encoded = to_categorical(y_train_seq)
    y_test_encoded = to_categorical(y_test_seq)


    gru = gru_models.GRUClassifier(time_step=time_step,
                                    X_train= x_train_seq, 
                                    y_train= y_train_encoded, 
                                    X_test= x_test_seq, 
                                    y_test= y_test_encoded
                                    )

    gru.train(
        batch_size=64, 
        learning_rate=0.001, 
        activation = "relu",
        epochs=1000,
        loss="binary_crossentropy",
        layers = 2,
        neurons = 50,
        verbose=1
        )
    
    # Save models in a specific location under the common parent folder

    models_path = os.path.join(common_path, "Files")

    if not os.path.exists(models_path): #if the folder exists
        os.makedirs(models_path) # Create the folder if it doesn't exist

    gru.to_onnx(model_name=os.path.join(models_path, "gru.H1.onnx"), standard_scaler=scaler)


schedule.every(1).minute.do(trainAndSaveCatBoost) #schedule catboost training
schedule.every(1).minute.do(trainAndSaveGRU) #scheduled GRU training

# Keep the script running to execute the scheduled tasks
while True:
    schedule.run_pending()
    time.sleep(60)  # Wait for 1 minute before checking again


mt5.shutdown()