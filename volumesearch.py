from datetime import timedelta
import json

import numpy as np
import pandas as pd
from gridsearch import calculate_buy_and_hold, get_all_data_pkl, create_data_hash
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.base import BaseEstimator
import optuna
from cache import get_redis_server, get_redis_server_url
import pickle
import xxhash
import time as timemod

# Set the option to display all columns
pd.set_option('display.max_columns', None)

global_hyperopt_version_cache_prefix = 'Volumne_V1_'

donchian_high_x_range = [4, 100, 2]
donchian_low_x_range = [4, 100, 2]
ema_x_range = [50, 600, 50] # long
ema_y_range = [2, 40, 2] # short
macd_x_range = [5, 200, 5] # signal line
volume_x_range = [4, 100, 2]

# TODO: time in market could be another good factor to check for a final result. less time in market is better
# TODO: divide the result by number of sells or something... less sells is normally better.
def calculate_strategy(data, ema_x, ema_y, macd_x, volume_x, donchian_high_x, donchian_low_x, initial_capital=10000, order_size=1000, max_open_orders=10):
    # Initialize capital and open orders
    available_capital = initial_capital
    open_orders = []

    # Ensure 'pair' is retained during resampling
    data = data.set_index(['open_time', 'pair'])
    resampled = data.groupby('pair').resample('1h', level=0).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).reset_index()

    # Calculate EMA, MACD, volume oscillator, and Donchian channels
    resampled['ema_x'] = resampled.groupby('pair')['close'].transform(lambda x: x.ewm(span=ema_x, adjust=False).mean())
    resampled['ema_y'] = resampled.groupby('pair')['close'].transform(lambda x: x.ewm(span=ema_y, adjust=False).mean())
    resampled['MACD_x'] = resampled['ema_y'] - resampled['ema_x']
    resampled['Signal_Line_x'] = resampled.groupby('pair')['MACD_x'].transform(lambda x: x.ewm(span=macd_x, adjust=False).mean())
    resampled['macd_histogram_x'] = resampled['MACD_x'] - resampled['Signal_Line_x']
    resampled['volume_osci_x'] = resampled.groupby('pair')['volume'].transform(lambda x: x - x.ewm(span=volume_x, adjust=False).mean())
    resampled['donchian_high_x'] = resampled.groupby('pair')['high'].transform(lambda x: x.rolling(window=donchian_high_x).max())
    resampled['donchian_low_x'] = resampled.groupby('pair')['low'].transform(lambda x: x.rolling(window=donchian_low_x).min())

    # Calculate MACD change and volume oscillator condition
    resampled['macd_change'] = (resampled['MACD_x'].shift(1) < 0) & (resampled['MACD_x'] > 0)
    resampled['volume_osci_above_zero'] = resampled['volume_osci_x'] > 0

    # Check conditions for long trade
    resampled['long_trade'] = (resampled['close'] > resampled['ema_x']) & resampled['macd_change'] & resampled['volume_osci_above_zero']

    # Calculate Stop Loss at lower Donchian line
    resampled['stop_loss'] = resampled['donchian_low_x']

    # Adjust lower Donchian line upwards
    resampled['donchian_low_adjusted'] = resampled.groupby('pair')['donchian_low_x'].cummax()

    # Check conditions for re-entry
    resampled['macd_previously_negative'] = resampled['MACD_x'].shift(1) < 0
    resampled['re_entry'] = resampled['long_trade'] & resampled['macd_previously_negative']

    # Initialize columns for tracking
    resampled['buy_price'] = None
    resampled['sell_price'] = None
    resampled['order_active'] = False
    resampled['profit'] = 0

    for index, row in resampled.iterrows():
        if row['long_trade'] and available_capital >= order_size and len(open_orders) < max_open_orders:
            # Open a new order
            resampled.at[index, 'buy_price'] = row['close']
            resampled.at[index, 'order_active'] = True
            available_capital -= order_size
            open_orders.append((index, row['pair'], row['close']))
            # print(f"Opening order at {index}, Pair: {row['pair']}, Close: {row['close']}, Available capital: {available_capital}, Open orders: {len(open_orders)}")

        # Check existing orders for closing condition
        for order in open_orders[:]:
            order_index, order_pair, buy_price = order
            if row['pair'] == order_pair and row['low'] < resampled.at[order_index, 'donchian_low_adjusted']:
                # Close the order
                resampled.at[order_index, 'sell_price'] = row['low']
                profit = (row['low'] - buy_price) * (order_size / buy_price)
                resampled.at[order_index, 'profit'] = profit
                available_capital += (order_size + profit)
                resampled.at[order_index, 'order_active'] = False
                open_orders.remove(order)
                # print(f"Closing order at {index}, Pair: {row['pair']}, Low: {row['low']}, Profit: {profit}, Available capital: {available_capital}, Open orders: {len(open_orders)}")

    # Force sell all open orders at the last available close price
    if open_orders:
        last_close_data = resampled.iloc[-1]
        for order in open_orders:
            order_index, order_pair, buy_price = order
            last_close = resampled[(resampled['pair'] == order_pair)].iloc[-1]['close']
            resampled.at[order_index, 'sell_price'] = last_close
            profit = (last_close - buy_price) * (order_size / buy_price)
            resampled.at[order_index, 'profit'] = profit
            available_capital += (order_size + profit)
            resampled.at[order_index, 'order_active'] = False
            # print(f"Force closing order at {order_index}, Pair: {order_pair}, Close: {last_close}, Profit: {profit}, Available capital: {available_capital}, Open orders: 0")
        open_orders.clear()

    total_profit = resampled['profit'].sum()

    return total_profit


def volumesearch(pairs, start_time, end_time, candle_directory, cache_directory):

    all_data = get_all_data_pkl(pairs, start_time, end_time, candle_directory, cache_directory)
        
    bnh_profit = calculate_buy_and_hold(all_data)

    print('Buy and Hold profit:', bnh_profit)
    
    def objective(trial):
        # Define the hyperparameters using the trial object
        donchian_high_x = trial.suggest_int('donchian_high_x', donchian_high_x_range[0], donchian_high_x_range[1], step=donchian_high_x_range[2])
        donchian_low_x = trial.suggest_int('donchian_low_x', donchian_low_x_range[0], donchian_low_x_range[1], step=donchian_low_x_range[2])
        ema_x = trial.suggest_int('ema_x', ema_x_range[0], ema_x_range[1], step=ema_x_range[2])
        ema_y = trial.suggest_int('ema_y', ema_y_range[0], ema_y_range[1], step=ema_y_range[2])
        macd_x = trial.suggest_int('macd_x', macd_x_range[0], macd_x_range[1], step=macd_x_range[2])
        volume_x = trial.suggest_int('volume_x', volume_x_range[0], volume_x_range[1], step=volume_x_range[2])
        
        # Calculate the profit using the calculate_strategy function
        X = all_data.copy(deep=True)
        profit = calculate_strategy(X, ema_x, ema_y, macd_x, volume_x, donchian_high_x, donchian_low_x)

        return profit

    # Create a study object and optimize the objective
    storage = optuna.storages.JournalStorage(optuna.storages.JournalRedisStorage(get_redis_server_url()))
    study = optuna.create_study(study_name='volumesearch_v2_' + str(xxhash.xxh64(json.dumps(pairs).encode()).hexdigest()) + start_time.strftime('%Y%m%d') + end_time.strftime('%Y%m%d'), direction='maximize', storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=5000, n_jobs=10)

    # Get the best score and best parameters
    best_score = study.best_value
    best_params = study.best_params

    print('Best score:', best_score)
    print('Best parameters:', best_params)

    return best_score