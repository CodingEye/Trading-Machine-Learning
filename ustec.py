import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score

# Define column names
column_names = ['Date', 'Time','Open', 'High', 'Low', 'Close', 'Volume']

# Get the Dataset
df = pd.read_csv('USTEC1.csv', na_values=['null'], header=None, names=column_names, index_col=0, parse_dates=[[0, 1]])
df.head()

#Step 3: Checking for Null Values by Printing the DataFrame Shape
#In this step, firstly, we will print the structure of the dataset. We’ll then check for null values in the data frame to ensure that there are none. The existence of null values in the dataset causes issues during training since they function as outliers, creating a wide variance in the training process.

# Print the shape of DataFrame and Check for Null Values
print("Dataframe Shape:", df.shape)
print("Null Value Present:", df.isnull().values.any())

#Plot the True Adj Close Value
df['Close'].plot()