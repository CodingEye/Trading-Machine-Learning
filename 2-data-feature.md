USTEC_all_timeframes
This file contains raw row data (probably OHLCV: Open, High, Low, Close, Volume, etc.) for the USTEC symbol across different timeframes.
No indicator columns, just the basic market data.
USTEC_features
This file contains the same basic data, but also processed indicator columns (like LWMA_15, LWMA_60, LWMA_200, STOCH_K, STOCH_D, etc.).
These indicators are calculated using a script (xgboost_processer_v2.py) and added to the data.
2. Why Process and Add Indicator Values in Advance?
Your Question:
Why do we need to process and add indicator value as we can do it while running?

Short Answer:
For machine learning model training and backtesting, you need a static, reproducible dataset.
For live trading, you can calculate indicators on the fly, but for ML, you need them precomputed for all historical data.
Detailed Explanation:
A. Machine Learning Workflow
Training a Model:

You need a dataset where every row has all the features (including indicators) already calculated.
This allows you to train, validate, and test your model efficiently and reproducibly.
If you calculated indicators "on the fly" during training, you'd have to recalculate them every epoch/batch, which is slow and can lead to inconsistencies.
Preprocessing ensures that your features are consistent and that your model sees the same data every time.
Backtesting:

You want to simulate how your model would have performed in the past.
You need all indicator values for every historical bar, so you can "step through" the data as if it were live.
Precomputing indicators ensures your backtest is fast and accurate.
B. Live Trading
In live trading, you do calculate indicators on the fly (using the latest data).
But for historical data and model training, you need the indicators for every bar in the past, which is why you process and store them.
C. Technical Reasons
Efficiency:

Calculating indicators (especially long-period ones like LWMA_200) for every row, every time you train or test, is computationally expensive.
Preprocessing once and saving the results is much faster.
Consistency:

If you change your indicator calculation code, your historical features might change, breaking reproducibility.
By saving the processed features, you ensure your training/testing is always on the same data.
Feature Engineering:

You might want to try different indicators, or combinations, or transformations.
It's easier to preprocess and save them, then experiment with different models.
3. Summary Table
Use Case	Indicator Calculation	Why?
Model Training	Preprocessed (in CSV)	Fast, reproducible, consistent, efficient
Backtesting	Preprocessed (in CSV)	Fast, accurate, matches training data
Live Trading	On the fly	Uses latest data, only need current value
4. Bottom Line
You process and add indicator values to your historical data so that your machine learning models can train and backtest efficiently and reproducibly.
In live trading, you calculate indicators on the fly, but for ML, you need a static dataset with all features precomputed.