import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os

# --- Helper functions for metrics (calculate_cagr_from_pnl_atr will need to be adapted for USD) ---
# Adapted for USD balance
def calculate_cagr_from_balance_series(balance_series_usd, num_years):
    if num_years <= 0 or len(balance_series_usd) < 2: return 0.0
    initial_capital_usd = balance_series_usd.iloc[0]
    ending_value_usd = balance_series_usd.iloc[-1]
    if initial_capital_usd <= 0 : return 0.0 
    if ending_value_usd <= 0: return -1.0 
    return (ending_value_usd / initial_capital_usd) ** (1 / num_years) - 1

def calculate_sharpe_ratio(returns_series, risk_free_rate_per_trade=0): 
    if returns_series.empty or len(returns_series) < 2: return np.nan
    std_dev = returns_series.std()
    if std_dev == 0 or pd.isna(std_dev): return np.nan if returns_series.mean() - risk_free_rate_per_trade == 0 else np.inf * np.sign(returns_series.mean() - risk_free_rate_per_trade)
    excess_returns = returns_series - risk_free_rate_per_trade
    return excess_returns.mean() / std_dev

def calculate_sortino_ratio(returns_series, risk_free_rate_per_trade=0, target_return_per_trade=0):
    if returns_series.empty or len(returns_series) < 2 : return np.nan
    avg_return = returns_series.mean()
    downside_diff = target_return_per_trade - returns_series
    downside_returns_values = downside_diff[downside_diff > 0]
    if not len(downside_returns_values): return np.inf 
    downside_std = np.sqrt(np.mean(downside_returns_values**2))
    if downside_std == 0 or pd.isna(downside_std): return np.inf 
    return (avg_return - risk_free_rate_per_trade) / downside_std

# --- Comprehensive Metrics Calculation & Reporting Function (MODIFIED FOR USD) ---
def generate_report(trades_df, report_label, years_in_data_param, current_config_params_for_report, initial_account_balance_usd_param): # Renamed current_config to avoid clash
    print(f"\n--- Performance Report: {report_label} ---")
    print(f"--- Initial Account Balance: ${initial_account_balance_usd_param:.2f} ---")
    metrics = {'Label': report_label} # InitialBalanceUSD will be added from current_config_params_for_report

    if trades_df.empty or 'PNL_USD_Net' not in trades_df.columns or trades_df['PNL_USD_Net'].empty:
        print(f"No trades to analyze for {report_label}.")
        metrics.update({'TotalTrades': 0, 'Error': "No trades or PNL_USD_Net empty"})
        for k in ['WinRate', 'AvgPnL_USD', 'TotalPnL_USD', 'MAR_USD', 'MaxDrawdownAbs_USD', 'MaxDrawdownPct_Account',
                    'ProfitFactor_USD', 'PayoffRatio_USD', 'AvgWinAmount_USD', 'AvgLossAmount_USD',
                    'CAGR_Pct_USD', 'SharpeRatio_USD', 'SortinoRatio_USD', 'CalmarRatio_USD',
                    'LongestLossStreak', 'Expectancy_USD', 'AvgHoldingPeriod_Candles']:
            metrics[k] = 0.0 if k not in ['TotalTrades', 'LongestLossStreak', 'WinningTrades', 'LosingTrades'] else 0
        return metrics

    num_trades = len(trades_df)
    trades_df['PNL_USD_Net'] = trades_df['PNL_USD_Net'].fillna(0) 

    total_pnl_usd = trades_df['PNL_USD_Net'].sum()
    avg_pnl_usd = trades_df['PNL_USD_Net'].mean() if num_trades > 0 else 0.0
    
    wins_df = trades_df[trades_df['PNL_USD_Net'] > 0]
    losses_df = trades_df[trades_df['PNL_USD_Net'] < 0]
    winning_trades = len(wins_df); losing_trades = len(losses_df)
    win_rate = winning_trades / num_trades if num_trades > 0 else 0.0
    
    avg_win_amount_usd = wins_df['PNL_USD_Net'].mean() if winning_trades > 0 else 0.0
    avg_loss_amount_usd = abs(losses_df['PNL_USD_Net'].mean()) if losing_trades > 0 else 0.0
    if pd.isna(avg_win_amount_usd): avg_win_amount_usd = 0.0
    if pd.isna(avg_loss_amount_usd): avg_loss_amount_usd = 0.0

    profit_factor_numerator_usd = wins_df['PNL_USD_Net'].sum()
    profit_factor_denominator_usd = abs(losses_df['PNL_USD_Net'].sum())
    profit_factor_usd = (profit_factor_numerator_usd / profit_factor_denominator_usd) if profit_factor_denominator_usd > 1e-9 else (np.inf if profit_factor_numerator_usd > 1e-9 else 0.0)
    payoff_ratio_usd = (avg_win_amount_usd / avg_loss_amount_usd) if avg_loss_amount_usd > 1e-9 else (np.inf if avg_win_amount_usd > 0 else 0.0)
    expectancy_usd = (win_rate * avg_win_amount_usd) - ((1 - win_rate) * avg_loss_amount_usd) if num_trades > 0 else 0.0
    avg_holding_period = trades_df['Duration_Candles'].mean() if 'Duration_Candles' in trades_df and num_trades > 0 else 0.0

    if 'Account_Balance_After_Trade_USD' in trades_df.columns and not trades_df['Account_Balance_After_Trade_USD'].empty:
        equity_curve_usd_values = [initial_account_balance_usd_param] + trades_df['Account_Balance_After_Trade_USD'].tolist()
    else: 
        equity_curve_usd_values = [initial_account_balance_usd_param] + (initial_account_balance_usd_param + trades_df['PNL_USD_Net'].cumsum()).tolist()
    
    equity_curve_series_usd = pd.Series(equity_curve_usd_values)
    running_max_equity_usd = equity_curve_series_usd.cummax()
    drawdown_series_abs_usd = running_max_equity_usd - equity_curve_series_usd
    max_dd_abs_usd = drawdown_series_abs_usd.max() if not drawdown_series_abs_usd.empty else 0.0
    
    peak_equity_for_dd_pct_calc_usd = initial_account_balance_usd_param
    if not drawdown_series_abs_usd.empty and max_dd_abs_usd > 0:
        idx_max_dd_end = drawdown_series_abs_usd.idxmax()
        if idx_max_dd_end < len(running_max_equity_usd): 
             peak_equity_for_dd_pct_calc_usd = running_max_equity_usd.iloc[idx_max_dd_end]
        elif not running_max_equity_usd.empty:
             peak_equity_for_dd_pct_calc_usd = running_max_equity_usd.max()
    elif not running_max_equity_usd.empty:
        peak_equity_for_dd_pct_calc_usd = running_max_equity_usd.max()

    if peak_equity_for_dd_pct_calc_usd <= 1e-9: peak_equity_for_dd_pct_calc_usd = initial_account_balance_usd_param 
    max_dd_pct_account = (max_dd_abs_usd / peak_equity_for_dd_pct_calc_usd) * 100 if peak_equity_for_dd_pct_calc_usd > 0 else 0.0
    
    cagr_usd = calculate_cagr_from_balance_series(equity_curve_series_usd, years_in_data_param)
    annual_pnl_proxy_usd = total_pnl_usd / years_in_data_param if years_in_data_param > 0 else 0.0
    mar_ratio_usd = annual_pnl_proxy_usd / max_dd_abs_usd if max_dd_abs_usd > 1e-9 else (np.inf if annual_pnl_proxy_usd > 0 else 0.0)
    calmar_ratio_usd = (cagr_usd * 100 if cagr_usd is not None else -100.0) / max_dd_pct_account if max_dd_pct_account > 1e-9 else (np.inf if cagr_usd is not None and cagr_usd > 0 else 0.0)
    
    sharpe_ratio_val_usd = calculate_sharpe_ratio(trades_df['PNL_USD_Net'])
    sortino_ratio_val_usd = calculate_sortino_ratio(trades_df['PNL_USD_Net'])
    avg_trades_per_year_for_annualization = num_trades / years_in_data_param if years_in_data_param > 0 else 1.0
    annualization_factor = np.sqrt(max(1.0, avg_trades_per_year_for_annualization)) 
    sharpe_ratio_usd = sharpe_ratio_val_usd * annualization_factor if pd.notna(sharpe_ratio_val_usd) else np.nan
    sortino_ratio_usd = sortino_ratio_val_usd * annualization_factor if pd.notna(sortino_ratio_val_usd) else np.nan

    longest_win_streak = 0; current_ws = 0; longest_loss_streak = 0; current_ls = 0
    for pnl_val_usd in trades_df['PNL_USD_Net']:
        if pnl_val_usd > 0: current_ws += 1; current_ls = 0
        elif pnl_val_usd < 0: current_ls += 1; current_ws = 0
        else: current_ws = 0; current_ls = 0 
        longest_win_streak = max(longest_win_streak, current_ws)
        longest_loss_streak = max(longest_loss_streak, current_ls)

    metrics.update({
        'TotalTrades': num_trades, 'WinningTrades': winning_trades, 'LosingTrades': losing_trades,
        'WinRate': win_rate, 
        'TotalPnL_USD': total_pnl_usd, 'AvgPnL_USD': avg_pnl_usd, 'Expectancy_USD': expectancy_usd,
        'AvgHoldingPeriod_Candles': avg_holding_period, 
        'MaxDrawdownAbs_USD': max_dd_abs_usd, 'MaxDrawdownPct_Account': max_dd_pct_account,
        'CAGR_Pct_USD': cagr_usd * 100 if cagr_usd is not None else -100.0,
        'MAR_USD': mar_ratio_usd, 
        'SharpeRatio_USD': sharpe_ratio_usd, 'SortinoRatio_USD': sortino_ratio_usd,
        'ProfitFactor_USD': profit_factor_usd, 'CalmarRatio_USD': calmar_ratio_usd,
        'AvgWinAmount_USD': avg_win_amount_usd, 'AvgLossAmount_USD': avg_loss_amount_usd,
        'PayoffRatio_USD': payoff_ratio_usd, 
        'LongestWinStreak': longest_win_streak, 'LongestLossStreak': longest_loss_streak
    })

    if 'Raw_PNL_ATR' in trades_df.columns and num_trades > 0: # Keep ATR based raw PNL for diagnosis
        metrics['AvgRawPnL_ATR_ALL'] = trades_df['Raw_PNL_ATR'].mean()
        # Add other raw ATR metrics if needed for debugging underlying strategy logic

    print(f"{'Metric':<30}{'Value':<20}"); print("-" * 50)
    for k_met, v_met in metrics.items():
        if k_met == 'Label' or k_met in current_config_params_for_report: continue 
        is_float = isinstance(v_met, (float, np.floating)) and not k_met.endswith('Streak') and \
                   k_met not in ['TotalTrades', 'WinningTrades', 'LosingTrades'] and \
                   not k_met.startswith('AvgHolding') 
        
        format_str = "{v_met:<20.2f}" if k_met.endswith('USD') or k_met.startswith('MaxDrawdownAbs') or k_met.startswith('AvgP') or k_met.startswith('TotalP') or k_met.startswith('Expectancy') else "{v_met:<20.3f}"
        if k_met.endswith('Pct_USD') or k_met.endswith('Pct_Account') or k_met == 'WinRate' : format_str = "{v_met:<20.3%}"
        if is_float and pd.notna(v_met): print(f"{k_met:<30}{(format_str).format(v_met=v_met)}")
        else: print(f"{k_met:<30}{str(v_met):<20}")

    if not equity_curve_series_usd.empty and num_trades > 0:
        plt.figure(figsize=(12,6)); plt.plot(equity_curve_series_usd.index, equity_curve_series_usd.values) 
        plt.title(f'Account Equity Curve (USD) - {report_label} - {num_trades} Trades', fontsize=14)
        plt.xlabel('Trade Number'); plt.ylabel(f'Account Balance (USD)')
        plt.grid(True); safe_report_label = "".join(c if c.isalnum() else "_" for c in report_label)
        plot_filename = f"equity_curve_usd_{safe_report_label}.png"
        try: plt.savefig(plot_filename); print(f"Saved USD equity curve: {plot_filename}")
        except Exception as e_plot: print(f"Error saving USD equity curve plot {plot_filename}: {e_plot}")
        plt.close()
    elif num_trades > 0 : print(f"USD Equity curve not plotted for {report_label} due to empty equity series but trades exist.")
    return metrics

# --- POSITION SIZING FUNCTION ---
def calculate_position_size(
    current_account_balance_usd, risk_setting_mode, risk_value, 
    stop_loss_distance_price, entry_price, leverage,
    pip_size_in_price_units, pip_value_per_lot_account_currency,
    contract_size_units_per_lot, min_contract_size_lots, contract_step_lots
):
    if stop_loss_distance_price <= 1e-9: return 0.0 
    risk_amount_usd = 0.0
    if risk_setting_mode == 'percent_account': risk_amount_usd = current_account_balance_usd * (risk_value / 100.0)
    elif risk_setting_mode == 'fixed_usd': risk_amount_usd = risk_value
    else: return 0.0 
    if risk_amount_usd <= 0: return 0.0
    if risk_amount_usd >= current_account_balance_usd * 0.9: risk_amount_usd = current_account_balance_usd * 0.9 

    stop_loss_pips = stop_loss_distance_price / pip_size_in_price_units
    if stop_loss_pips <= 1e-5: return 0.0
    value_per_pip_per_lot = pip_value_per_lot_account_currency
    if value_per_pip_per_lot <= 0: return 0.0

    lots_calculated = risk_amount_usd / (stop_loss_pips * value_per_pip_per_lot)
    if lots_calculated < min_contract_size_lots: return 0.0 
    lots_calculated = np.floor(lots_calculated / contract_step_lots) * contract_step_lots
    if lots_calculated < min_contract_size_lots: return 0.0

    notional_value_in_base_currency_units = lots_calculated * contract_size_units_per_lot
    notional_value_account_currency = notional_value_in_base_currency_units * entry_price 
    margin_required_account_currency = notional_value_account_currency / leverage

    if margin_required_account_currency > current_account_balance_usd:
        while margin_required_account_currency > current_account_balance_usd and lots_calculated >= min_contract_size_lots:
            lots_calculated -= contract_step_lots
            if lots_calculated < min_contract_size_lots: lots_calculated = 0.0; break 
            notional_value_in_base_currency_units = lots_calculated * contract_size_units_per_lot
            notional_value_account_currency = notional_value_in_base_currency_units * entry_price
            margin_required_account_currency = notional_value_account_currency / leverage
        if lots_calculated < min_contract_size_lots: return 0.0
    
    if lots_calculated <= 1e-7: return 0.0
    return round(lots_calculated, int(-np.log10(contract_step_lots) if contract_step_lots > 0 else 2) )

# Load the data
#file_path = r"C:\Users\Bl4ckP3n9u1n\Documents\SigmaEA\GBPUSD_M15_202201030000_202506041200.csv"
# Define subfolder structure
data_subfolder = "historical_data"  # Subfolder for all historical price data
instrument_folder = "USTEC"         # Specific instrument folder

# Create full path
base_path = os.path.dirname(os.path.abspath(__file__))  # Gets current script directory
data_path = os.path.join(base_path, data_subfolder, instrument_folder)

# Create directories if they don't exist
os.makedirs(data_path, exist_ok=True)

# Update the file path
file_path = os.path.join(data_path, "USTEC_historical.csv")

# --- Financial Account & Instrument Configuration ---
INITIAL_ACCOUNT_BALANCE_USD = 100.0
LEVERAGE = 30 
ACCOUNT_CURRENCY = "USD" 
INSTRUMENT_NAME = "GBPUSD"
CONTRACT_SIZE_UNITS_PER_LOT = 100000  
PIP_DECIMAL_PLACES = 4 
PIP_SIZE_IN_PRICE_UNITS = 10**(-PIP_DECIMAL_PLACES) 
PIP_VALUE_PER_LOT_ACCOUNT_CURRENCY = PIP_SIZE_IN_PRICE_UNITS * CONTRACT_SIZE_UNITS_PER_LOT
MIN_CONTRACT_SIZE_LOTS = 0.01
CONTRACT_STEP_LOTS = 0.01

# --- ITERABLE FINANCIAL PARAMETERS ---
risk_setting_modes_to_test = ['percent_account'] 
risk_percent_account_values_range = [1.0, 2.0, 5.0, 10.0] 
# risk_fixed_usd_values_range = [1.0, 2.0, 5.0, 10.0] 
iterated_spread_pips_rt_range = [0.0, 0.5, 1.0, 2.0, 3.0] 

# --- BASE STRATEGY PARAMETERS ---
base_strategy_params = {
    'donchian_period': 20, 'chandelier_k': 1.5, 'chandelier_atr_dynamic': False, 
    'hard_sl_mult': 2.0, 'time_stop': 45, 'pl_trigger': 1.0, 'pl_level': 0.3,
    'profit_lock_active': True, 'commission_atr': 0.01, 
    'max_consecutive_losses_for_half_size': 2, 
    'wins_to_recover_full_size': 1,
    'strategy_label_base': "Iter19_Monetary" # Base label for the strategy
}

try:
    df = pd.read_csv(file_path, delimiter='\t')
    expected_columns = ['<DATE>', '<TIME>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<TICKVOL>', '<VOL>', '<SPREAD>']
    if all(col in df.columns for col in expected_columns):
        df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Tickvol', 'Volume', 'Spread_Original_Data'] 
        df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
        df = df.set_index('Timestamp')
        df['Effective_Volume'] = df['Volume'].where(df['Volume'] > 0, df['Tickvol'])
        df_ohlc_raw = df[['Open', 'High', 'Low', 'Close', 'Effective_Volume']].copy()
        df_ohlc_raw.rename(columns={'Effective_Volume': 'Volume'}, inplace=True)
    else: print("Columns do not match expected MT5 export format."); df_ohlc_raw = None
except Exception as e: print(f"Error loading {file_path}: {e}"); df_ohlc_raw = None

if df_ohlc_raw is not None:
    df_ohlc_prepared = df_ohlc_raw.copy()
    period_atr = 14
    df_ohlc_prepared['H-L'] = df_ohlc_prepared['High'] - df_ohlc_prepared['Low']
    df_ohlc_prepared['H-PC'] = abs(df_ohlc_prepared['High'] - df_ohlc_prepared['Close'].shift(1))
    df_ohlc_prepared['L-PC'] = abs(df_ohlc_prepared['Low'] - df_ohlc_prepared['Close'].shift(1))
    df_ohlc_prepared['TR_temp'] = df_ohlc_prepared[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_ohlc_prepared['ATR_calc'] = df_ohlc_prepared['TR_temp'].rolling(window=period_atr).mean()
    df_ohlc_prepared['ATR_Smooth_3'] = df_ohlc_prepared['ATR_calc'].rolling(window=3).mean()
    df_ohlc_prepared['ATR_Slope_Smooth_3'] = df_ohlc_prepared['ATR_Smooth_3'].diff()
    df_ohlc_prepared['ATR_100_Rolling_Q10'] = df_ohlc_prepared['ATR_calc'].rolling(window=100).quantile(0.10)
    donchian_period = base_strategy_params['donchian_period'] 
    df_ohlc_prepared['Donchian_High'] = df_ohlc_prepared['High'].rolling(window=donchian_period).max().shift(1)
    df_ohlc_prepared['Donchian_Low'] = df_ohlc_prepared['Low'].rolling(window=donchian_period).min().shift(1)
    df_ohlc_prepared['ATR_100_Rolling_Q33'] = df_ohlc_prepared['ATR_calc'].rolling(window=100).quantile(0.33)
    df_ohlc_prepared['ATR_100_Rolling_Q66'] = df_ohlc_prepared['ATR_calc'].rolling(window=100).quantile(0.66)
    df_ohlc_prepared.dropna(subset=['ATR_calc', 'ATR_Slope_Smooth_3', 'ATR_100_Rolling_Q10', 
                                    'Donchian_High', 'Donchian_Low',
                                    'ATR_100_Rolling_Q33', 'ATR_100_Rolling_Q66'], inplace=True)
    print(f"Data prepared: {len(df_ohlc_prepared)} rows from {df_ohlc_prepared.index.min()} to {df_ohlc_prepared.index.max()}")

    base_signals_list = []
    if not df_ohlc_prepared.empty:
        start_idx_for_signals = 1 
        if len(df_ohlc_prepared) > start_idx_for_signals + 1: 
            for i in range(start_idx_for_signals, len(df_ohlc_prepared) - 1):
                signal_candle = df_ohlc_prepared.iloc[i]; prev_candle = df_ohlc_prepared.iloc[i-1]; t_plus_1_candle = df_ohlc_prepared.iloc[i+1]
                current_atr_val_signal = signal_candle['ATR_calc']; atr_slope_val_signal = signal_candle['ATR_Slope_Smooth_3']
                atr_q10_signal = signal_candle['ATR_100_Rolling_Q10']
                if pd.isna(current_atr_val_signal) or current_atr_val_signal <= 1e-9 or \
                   pd.isna(atr_slope_val_signal) or pd.isna(atr_q10_signal) or \
                   any(pd.isna(val) for val in [signal_candle['Donchian_High'], signal_candle['Donchian_Low'],
                                                prev_candle['Close'], prev_candle['Donchian_High'], prev_candle['Donchian_Low'],
                                                t_plus_1_candle['Open'], t_plus_1_candle['Close'], 
                                                signal_candle['High'], signal_candle['Low']]): continue
                if not (atr_slope_val_signal > 0): continue
                if current_atr_val_signal < atr_q10_signal: continue
                direction = 0
                if signal_candle['Close'] > signal_candle['Donchian_High'] and not (prev_candle['Close'] > prev_candle['Donchian_High']):
                    if t_plus_1_candle['Close'] > signal_candle['High']: direction = 1
                elif signal_candle['Close'] < signal_candle['Donchian_Low'] and not (prev_candle['Close'] < prev_candle['Donchian_Low']):
                    if t_plus_1_candle['Close'] < signal_candle['Low']: direction = -1
                if direction != 0:
                    atr_q33_signal = signal_candle['ATR_100_Rolling_Q33']; atr_q66_signal = signal_candle['ATR_100_Rolling_Q66']
                    vol_regime = 'Mid Vol'
                    if pd.notna(atr_q33_signal) and current_atr_val_signal < atr_q33_signal: vol_regime = 'Low Vol (Q10-Q33)'
                    elif pd.notna(atr_q66_signal) and current_atr_val_signal > atr_q66_signal: vol_regime = 'High Vol'
                    base_signals_list.append({
                        'Timestamp': signal_candle.name, 'Signal_Iloc_Index': df_ohlc_prepared.index.get_loc(signal_candle.name),
                        'Entry_Price_Actual_Base': t_plus_1_candle['Open'], 
                        'Direction': direction,
                        'ATR_at_Signal': current_atr_val_signal, 'Vol_Regime_Signal': vol_regime,
                    })
    all_qualified_signals_df = pd.DataFrame(base_signals_list)
    print(f"--- Generated {len(all_qualified_signals_df)} qualified signals ---")

    financial_param_sets = []
    for r_mode in risk_setting_modes_to_test:
        risk_values_to_iterate = []
        if r_mode == 'percent_account': risk_values_to_iterate = risk_percent_account_values_range
        # elif r_mode == 'fixed_usd': risk_values_to_iterate = risk_fixed_usd_values_range
        for r_val in risk_values_to_iterate:
            for s_pips_rt in iterated_spread_pips_rt_range:
                financial_param_sets.append({
                    'risk_setting_mode': r_mode, 'risk_value': r_val, 'spread_pips_rt': s_pips_rt,
                })
    print(f"\n--- Generated {len(financial_param_sets)} financial parameter sets for testing. ---")
    if not financial_param_sets: print("WARNING: No financial parameter sets generated.")

    all_runs_metrics_summary_list = [] # Changed name to avoid confusion

    for fin_params_iter in financial_param_sets: # Changed name to avoid clash
        # Combine base strategy, fixed financial, and iterated financial parameters
        current_run_config = {
            **base_strategy_params, 
            'initial_account_balance_usd': INITIAL_ACCOUNT_BALANCE_USD, # Add fixed financial params
            'leverage': LEVERAGE,
            **fin_params_iter # Add iterated financial params
        }
        
        config_label = (f"{current_run_config['strategy_label_base']}_"
                        f"Risk_{current_run_config['risk_setting_mode'][:4]}{current_run_config['risk_value']}_"
                        f"Spread_{current_run_config['spread_pips_rt']}pips_Comm_{current_run_config['commission_atr']}atr")
        current_run_config['Label'] = config_label # Store the full label in the config itself for reporting

        print(f"\n--- Running Backtest for Config: {config_label} ---")
        current_account_balance_usd = current_run_config['initial_account_balance_usd']
        trade_results_list_for_config = []         
        
        if not all_qualified_signals_df.empty:
            consecutive_losses_count = 0 
            for trade_idx, signal_row in all_qualified_signals_df.iterrows():
                trade_id_timestamp = signal_row['Timestamp']
                atr_for_this_trade = signal_row['ATR_at_Signal']
                direction = signal_row['Direction']
                vol_regime_at_entry = signal_row['Vol_Regime_Signal']
                half_spread_price_adj = (current_run_config['spread_pips_rt'] / 2.0) * PIP_SIZE_IN_PRICE_UNITS
                base_entry_price = signal_row['Entry_Price_Actual_Base']
                entry_price = base_entry_price + (half_spread_price_adj * direction) # Apply one way based on direction perception

                if pd.isna(atr_for_this_trade) or atr_for_this_trade <= 1e-9 or pd.isna(entry_price): continue
                current_risk_value_for_trade = current_run_config['risk_value']
                if consecutive_losses_count >= current_run_config['max_consecutive_losses_for_half_size']:
                    current_risk_value_for_trade /= 2.0
                stop_loss_distance_price = current_run_config['hard_sl_mult'] * atr_for_this_trade
                position_size_lots = calculate_position_size(
                    current_account_balance_usd, current_run_config['risk_setting_mode'],
                    current_risk_value_for_trade, stop_loss_distance_price, entry_price, 
                    current_run_config['leverage'], PIP_SIZE_IN_PRICE_UNITS,
                    PIP_VALUE_PER_LOT_ACCOUNT_CURRENCY, CONTRACT_SIZE_UNITS_PER_LOT,
                    MIN_CONTRACT_SIZE_LOTS, CONTRACT_STEP_LOTS
                )
                if position_size_lots <= 0: continue 

                pnl_usd_net = 0.0; raw_pnl_atr_for_trade = 0.0
                exit_price_trade = np.nan; exit_reason_trade_summary = "Unknown"
                highest_high_for_chandelier_calc = entry_price 
                lowest_low_for_chandelier_calc = entry_price   
                mfe_tracker_high = entry_price; mfe_tracker_low = entry_price  
                locked_profit_stop_price = None 
                initial_hard_sl_price = entry_price - (direction * stop_loss_distance_price) 
                trade_duration_candles = 0; signal_iloc_in_df = signal_row['Signal_Iloc_Index']

                for k_trade in range(current_run_config['time_stop']): 
                    trade_duration_candles = k_trade + 1
                    current_candle_iloc = signal_iloc_in_df + 1 + k_trade
                    if current_candle_iloc >= len(df_ohlc_prepared):
                        last_available_close = df_ohlc_prepared.iloc[-1]['Close']
                        exit_price_trade = last_available_close - (half_spread_price_adj * direction) # Apply exit spread
                        exit_reason_trade_summary = "Data_End_In_Trade"; break
                    current_candle = df_ohlc_prepared.iloc[current_candle_iloc]
                    if any(pd.isna(current_candle[col]) for col in ['Open', 'High', 'Low', 'Close']): # Basic NaN check
                        exit_price_trade = (current_candle['Open'] if pd.notna(current_candle['Open']) else entry_price) - (half_spread_price_adj * direction)
                        exit_reason_trade_summary = "NaN_Data_In_Trade"; break

                    atr_for_chandelier_calc = atr_for_this_trade 
                    chandelier_stop_val_current_bar = 0.0 
                    if direction == 1: chandelier_stop_val_current_bar = highest_high_for_chandelier_calc - current_run_config['chandelier_k'] * atr_for_chandelier_calc
                    else: chandelier_stop_val_current_bar = lowest_low_for_chandelier_calc + current_run_config['chandelier_k'] * atr_for_chandelier_calc
                    if direction == 1:
                        mfe_tracker_high = max(mfe_tracker_high, current_candle['High'])
                        current_mfe_atr_for_pl = (mfe_tracker_high - entry_price) / atr_for_this_trade if atr_for_this_trade > 0 else 0
                    else: 
                        mfe_tracker_low = min(mfe_tracker_low, current_candle['Low'])
                        current_mfe_atr_for_pl = (entry_price - mfe_tracker_low) / atr_for_this_trade if atr_for_this_trade > 0 else 0
                    if current_run_config.get('profit_lock_active', True): 
                        if locked_profit_stop_price is None and current_mfe_atr_for_pl >= current_run_config['pl_trigger']:
                            locked_profit_stop_price = entry_price + (current_run_config['pl_level'] * atr_for_this_trade * direction)
                    current_effective_sl = initial_hard_sl_price
                    if direction == 1:
                        current_effective_sl = max(initial_hard_sl_price, chandelier_stop_val_current_bar) 
                        if current_run_config.get('profit_lock_active', True) and locked_profit_stop_price is not None: 
                            current_effective_sl = max(current_effective_sl, locked_profit_stop_price)
                    else: 
                        current_effective_sl = min(initial_hard_sl_price, chandelier_stop_val_current_bar)
                        if current_run_config.get('profit_lock_active', True) and locked_profit_stop_price is not None:
                            current_effective_sl = min(current_effective_sl, locked_profit_stop_price)

                    exit_hit_this_bar = False; temp_exit_reason_bar = ""; is_profit_lock_exit_type = False 
                    tentative_exit_price_no_exit_spread = np.nan
                    if direction == 1 and current_candle['Low'] <= current_effective_sl:
                        exit_hit_this_bar = True
                        if current_run_config.get('profit_lock_active', True) and locked_profit_stop_price is not None and \
                           abs(current_effective_sl - locked_profit_stop_price) < (1e-9 * max(1, atr_for_this_trade)):
                            temp_exit_reason_bar = "ProfitLockEx"; tentative_exit_price_no_exit_spread = locked_profit_stop_price; is_profit_lock_exit_type = True
                        elif abs(current_effective_sl - chandelier_stop_val_current_bar) < (1e-9 * max(1, atr_for_this_trade)):
                            temp_exit_reason_bar = "ChandelierEx"; tentative_exit_price_no_exit_spread = chandelier_stop_val_current_bar
                        else: temp_exit_reason_bar = "HardSLEx"; tentative_exit_price_no_exit_spread = initial_hard_sl_price
                        if pd.notna(tentative_exit_price_no_exit_spread):
                            if is_profit_lock_exit_type:
                                if current_candle['Open'] < base_entry_price: # Compare Open to original base_entry for PL gap
                                    tentative_exit_price_no_exit_spread = min(tentative_exit_price_no_exit_spread, current_candle['Open'])
                            elif current_candle['Open'] < tentative_exit_price_no_exit_spread: 
                                tentative_exit_price_no_exit_spread = current_candle['Open']
                    elif direction == -1 and current_candle['High'] >= current_effective_sl:
                        exit_hit_this_bar = True
                        if current_run_config.get('profit_lock_active', True) and locked_profit_stop_price is not None and \
                           abs(current_effective_sl - locked_profit_stop_price) < (1e-9 * max(1, atr_for_this_trade)):
                            temp_exit_reason_bar = "ProfitLockEx"; tentative_exit_price_no_exit_spread = locked_profit_stop_price; is_profit_lock_exit_type = True
                        elif abs(current_effective_sl - chandelier_stop_val_current_bar) < (1e-9 * max(1, atr_for_this_trade)):
                            temp_exit_reason_bar = "ChandelierEx"; tentative_exit_price_no_exit_spread = chandelier_stop_val_current_bar
                        else: temp_exit_reason_bar = "HardSLEx"; tentative_exit_price_no_exit_spread = initial_hard_sl_price
                        if pd.notna(tentative_exit_price_no_exit_spread):
                            if is_profit_lock_exit_type:
                                if current_candle['Open'] > base_entry_price: # Compare Open to original base_entry for PL gap
                                    tentative_exit_price_no_exit_spread = max(tentative_exit_price_no_exit_spread, current_candle['Open'])
                            elif current_candle['Open'] > tentative_exit_price_no_exit_spread:
                                tentative_exit_price_no_exit_spread = current_candle['Open']
                    
                    if exit_hit_this_bar and pd.notna(tentative_exit_price_no_exit_spread):
                        exit_price_trade = tentative_exit_price_no_exit_spread - (half_spread_price_adj * direction) # Apply exit spread
                        exit_reason_trade_summary = temp_exit_reason_bar + f"_Bar{trade_duration_candles}"; break 
                    if trade_duration_candles >= current_run_config['time_stop']:
                        tentative_exit_price_no_exit_spread = current_candle['Close'] # Time stop uses close before spread
                        exit_price_trade = tentative_exit_price_no_exit_spread - (half_spread_price_adj * direction)
                        exit_reason_trade_summary = "TimeStop"; break
                    if direction == 1: highest_high_for_chandelier_calc = max(highest_high_for_chandelier_calc, current_candle['High'])
                    else: lowest_low_for_chandelier_calc = min(lowest_low_for_chandelier_calc, current_candle['Low'])
                
                if pd.notna(exit_price_trade):
                    pnl_price_change_gross = (exit_price_trade - entry_price) * direction
                    pnl_usd_gross = (pnl_price_change_gross / PIP_SIZE_IN_PRICE_UNITS) * position_size_lots * PIP_VALUE_PER_LOT_ACCOUNT_CURRENCY
                    commission_cost_price_units = current_run_config['commission_atr'] * atr_for_this_trade
                    commission_cost_usd = (commission_cost_price_units / PIP_SIZE_IN_PRICE_UNITS) * position_size_lots * PIP_VALUE_PER_LOT_ACCOUNT_CURRENCY
                    pnl_usd_net = pnl_usd_gross - commission_cost_usd
                    if pd.notna(tentative_exit_price_no_exit_spread): # Ensure tentative was set for raw ATR calc
                         raw_pnl_atr_for_trade = (tentative_exit_price_no_exit_spread - base_entry_price) * direction / atr_for_this_trade if atr_for_this_trade > 0 else 0
                else: pnl_usd_net = 0.0; raw_pnl_atr_for_trade = 0.0 # Should not happen
                
                current_account_balance_usd += pnl_usd_net
                trade_results_list_for_config.append({
                    'Timestamp': trade_id_timestamp, 'PNL_USD_Net': pnl_usd_net, 'Raw_PNL_ATR': raw_pnl_atr_for_trade,
                    'Position_Size_Lots': position_size_lots, 'Account_Balance_After_Trade_USD': current_account_balance_usd,
                    'Direction': direction, 'Vol_Regime': vol_regime_at_entry, 'Duration_Candles': trade_duration_candles, 
                    'Exit_Reason': exit_reason_trade_summary, 'Entry_Price_With_Spread': entry_price, 
                    'Exit_Price_With_Spread': exit_price_trade, 'ATR_at_Entry_Signal': atr_for_this_trade,
                })
                if pnl_usd_net < 0: consecutive_losses_count += 1
                elif pnl_usd_net > 0 :
                    if current_run_config['wins_to_recover_full_size'] == 1: consecutive_losses_count = 0
        
        current_results_df = pd.DataFrame(trade_results_list_for_config)
        years_in_data_global = (df_ohlc_prepared.index.max() - df_ohlc_prepared.index.min()).days / 365.25 if len(df_ohlc_prepared) > 1 else 1.0/365.25
        
        metrics_for_this_run = generate_report(current_results_df, config_label, years_in_data_global, 
                                               current_run_config, # Pass full config for report context
                                               current_run_config['initial_account_balance_usd'])
        
        # Store all config parameters along with metrics for this run
        run_summary_data = {**current_run_config, **metrics_for_this_run}
        all_runs_metrics_summary_list.append(run_summary_data)
        
    if all_runs_metrics_summary_list:
        summary_df = pd.DataFrame(all_runs_metrics_summary_list)
        summary_filename = f"all_runs_summary_{base_strategy_params['strategy_label_base']}.csv"
        
        # Define desired order of columns for the summary CSV
        # Start with key identifying parameters, then key metrics
        cols_order_preference = [
            'Label', 'risk_setting_mode', 'risk_value', 'spread_pips_rt', 'commission_atr',
            'initial_account_balance_usd', 'leverage',
            'donchian_period', 'chandelier_k', 'hard_sl_mult', 'pl_trigger', 'pl_level', 'time_stop', # Key strategy params
            'TotalTrades', 'WinRate', 'TotalPnL_USD', 'AvgPnL_USD', 
            'MaxDrawdownAbs_USD', 'MaxDrawdownPct_Account', 'MAR_USD', 'CAGR_Pct_USD',
            'ProfitFactor_USD', 'PayoffRatio_USD', 'Expectancy_USD',
            'AvgWinAmount_USD', 'AvgLossAmount_USD',
            'LongestWinStreak', 'LongestLossStreak', 'AvgHoldingPeriod_Candles',
            'SharpeRatio_USD', 'SortinoRatio_USD', 'CalmarRatio_USD',
            'AvgRawPnL_ATR_ALL' # Keep this for diagnosis
        ]
        # Get actual columns present in summary_df, maintaining preferred order
        existing_cols_in_order = [col for col in cols_order_preference if col in summary_df.columns]
        # Add any other columns from summary_df not in preferred list (e.g., 'Error')
        remaining_cols = [col for col in summary_df.columns if col not in existing_cols_in_order]
        final_cols_for_csv = existing_cols_in_order + remaining_cols
        
        summary_df = summary_df[final_cols_for_csv] # Reorder
        summary_df.sort_values(by='MAR_USD', ascending=False, inplace=True)
        summary_df.to_csv(summary_filename, index=False, float_format='%.3f')
        print(f"\n--- Saved consolidated summary of all runs to {summary_filename} ---")
        print(f"\n--- Top Performing Runs (sorted by MAR_USD) ---")
        print(summary_df.head(20).to_string(index=False))
        if any(summary_df['TotalTrades'] < 10): # Example threshold
            print("\nWarning: Some runs have very few trades. Interpret metrics for these runs with caution.")
    else:
        print("No metrics were generated from any runs.")
else:
    print("DataFrame df_ohlc_prepared is not available. Cannot proceed with analysis.")