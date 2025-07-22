import pandas as pd
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


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


# Getting the training chunks

def getData(start = 1, bars = 1000):

    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_H1, start, bars)
    
    # create DataFrame out of the obtained chunks

    df_rates = pd.DataFrame(rates)
                                                
    return df_rates


def trainIncrementally():

    # CatBoost model
    clf = CatBoostClassifier(
        task_type="CPU",
        iterations=2000,
        learning_rate=0.2,
        max_depth=1,
        verbose=0,
    )
    
    # Load full dataset
    big_data = getData(1, 10000)

    # Split into chunks of 1000 samples
    chunk_size = 1000
    chunks = [big_data[i:i + chunk_size].copy() for i in range(0, len(big_data), chunk_size)]  # Use .copy() here

    

    for i, chunk in enumerate(chunks):
        # Split features and target
            
        # Preparing the target variable

        chunk["future_open"] = chunk["open"].shift(-1)
        chunk["future_close"] = chunk["close"].shift(-1)


        target = []
        for row in range(chunk.shape[0]):
            if chunk["future_close"].iloc[row] > chunk["future_open"].iloc[row]:
                target.append(1)
            else:
                target.append(0)

        chunk["target"] = target

        chunk = chunk.dropna()

        # Check if we were able to receive some data

        if (len(chunk)<=0):
            print("Failed to obtain chunk from Metatrader5, error = ",mt5.last_error())
            mt5.shutdown()


        X = chunk.drop(columns = ["spread","real_volume","future_close","future_open","target"])
        y = chunk["target"]


        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)

        if i == 0:
            # Initial training
            clf.fit(X_train, y_train, eval_set=(X_val, y_val))

            y_pred = clf.predict(X_val)
            print(f"---> Acc score: {accuracy_score(y_pred=y_pred, y_true=y_val)}")
        else:
            # Incremental training
            clf.fit(X_train, y_train, init_model="model.cbm", eval_set=(X_val, y_val))

            y_pred = clf.predict(X_val)
            print(f"---> Acc score: {accuracy_score(y_pred=y_pred, y_true=y_val)}")
        
        # Save the model
        clf.save_model("model.cbm")
        print(f"Chunk {i + 1}/{len(chunks)} processed and model saved.")


# Run the training process
trainIncrementally()
