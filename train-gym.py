import os
# Force CPU usage and suppress warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gymnasium as gym
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# Step 1: Load and prepare data
# -------------------------------
df = pd.read_csv('USTEC_historical_with_LWMAs.csv')
df['Date'] = pd.to_datetime(df['time'])
df.set_index('Date', inplace=True)

# Use correct column names from your CSV
df = df[['open', 'high', 'low', 'close', 'tick_volume', 'LWMA15', 'LWMA60', 'LWMA200']]

# Normalize the data
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

print("Data shape:", df_scaled.shape)
print("Columns:", df_scaled.columns.tolist())

# Drop any missing values
df_scaled.dropna(inplace=True)

# -------------------------------
# Step 2: Create Trading Environment
# -------------------------------
class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, window_size=50):
        super(CustomEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        self.current_step = window_size
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # Buy (1), Sell (2), Hold (0)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(window_size, 8), dtype=np.float32
        )
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        return self._next_observation(), {}
    
    def _next_observation(self):
        frame = self.df.iloc[self.current_step-self.window_size:self.current_step]
        return frame.values
        
    def step(self, action):
        self.current_step += 1
        
        done = self.current_step > len(self.df)-1
        truncated = False
        info = {}
            
        reward = 0
        if not done:
            next_price = self.df.iloc[self.current_step]['close']
            curr_price = self.df.iloc[self.current_step-1]['close']
            price_diff = next_price - curr_price
            
            if action == 1:  # Buy
                reward = price_diff
            elif action == 2:  # Sell
                reward = -price_diff
                
        return self._next_observation(), reward, done, truncated, info

# Create and wrap the environment
env = CustomEnv(df_scaled)
env = DummyVecEnv([lambda: env])

# -------------------------------
# Step 3: Train the Agent
# -------------------------------
model = A2C('MlpPolicy', env, 
            learning_rate=0.001,
            n_steps=5,
            tensorboard_log="./tensorboard_logs",
            verbose=1)

print("Starting training...")
model.learn(total_timesteps=10000)

# Save the trained model
model.save("ustec_trading_model")

# -------------------------------
# Step 4: Evaluate
# -------------------------------
obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward[0]

print(f"Total reward: {total_reward}")

# Close the environment
env.close()
