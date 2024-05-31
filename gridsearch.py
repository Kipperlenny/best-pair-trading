from datetime import datetime, timedelta
import itertools
import os
import pickle
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.base import BaseEstimator
import pandas as pd
from dateutil.relativedelta import relativedelta
from trading import get_best_channel_pair
from historic_data import load_month_data
import numpy as np
from sklearn.model_selection import GridSearchCV
from pandas.tseries.offsets import DateOffset
from cache import create_data_hash, create_cache_key
import sys
import time as timemod
from cache import get_redis_server
import xxhash

# parameters to hyperopt
'''price_jump_threshold_range = np.arange(0.05, 0.15, 0.01)
last_price_treshold_range = np.arange(0.40, 0.60, 0.05)
wished_profit_range = np.arange(1.01, 1.05, 1.1)
low_to_high_threshold_range = [5, 10, 50, 100, 250]
std_dev_threshold_range = [10, 50, 150, 300, 600]
rolling_window_number_range = range(10, 30, 2)
std_for_BB_range = np.arange(1, 3, 0.1)
moving_average_type_range = ['SMA', 'EMA']'''

price_jump_threshold_range = [0.02, 0.1, 0.2]# [0.2] # if price in one 5min candle jumps more than this percent, remove the pair from consideration
last_price_treshold_range = [0.1, 0.25, 0.5] # [0.5] #np.arange(0.4, 0.6, 0.05) # if the last price is more than this percent of the lower BB, remove the pair from consideration
wished_profit_range = [1.01, 1.03, 1.1] #np.arange(1.01, 1.05, 0.01)
rolling_window_number_range = [2, 10, 50, 500, 1000] # [50]
std_for_BB_range = [1, 2, 3] # [2] #np.arange(1, 3, 0.5)
moving_average_type_range = ['SMA', 'EMA'] # ['EMA']

global_hyperopt_version_cache_prefix = 'V1_'

class LineCountingStream:
    def __init__(self, stdout):
        self.stdout = stdout
        self.line_count = 0

    def write(self, text):
        self.line_count += text.count('\n')
        self.stdout.write(text)

    def reset_line_count(self):
        self.line_count = 0

    def get_line_count(self):
        return self.line_count

    def flush(self):
        self.stdout.flush()

# Save the original stdout
original_stdout = sys.stdout
    
# Redirect stdout to the line counting stream
line_counting_stream = LineCountingStream(original_stdout)
sys.stdout = line_counting_stream
    
class StrategyEstimator(BaseEstimator):
    def __init__(self, data_dict, n_jobs, max_jobs, price_jump_threshold=None, last_price_treshold=None, rolling_window_number=None, std_for_BB=None, moving_average_type=None, wished_profit=None):
        self.max_jobs = max_jobs
        self.n_jobs = n_jobs
        self.data_dict = data_dict
        self.price_jump_threshold = price_jump_threshold
        self.last_price_treshold = last_price_treshold
        self.rolling_window_number = rolling_window_number
        self.std_for_BB = std_for_BB
        self.moving_average_type = moving_average_type
        self.wished_profit = wished_profit

    def fit(self, X, y=None):
        X = X.copy(deep=True)
        grouped_X = X.groupby('open_time')
        self.strategy_result_ = calculate_strategy(grouped_X, X, self.data_dict, self.n_jobs, self.max_jobs, self.price_jump_threshold, self.last_price_treshold, self.rolling_window_number, self.std_for_BB, self.moving_average_type, self.wished_profit)
        return self

    def score(self, X, y=None):
        return self.strategy_result_
    
    def predict(self, X):
        X = X.copy(deep=True)
        # Assuming that the strategy result can be calculated for new data as well
        grouped_X = X.groupby('open_time')
        return calculate_strategy(grouped_X, X, 1, 1, self.price_jump_threshold, self.last_price_treshold, self.rolling_window_number, self.std_for_BB, self.moving_average_type, self.wished_profit)

def calculate_strategy(grouped_data, data, data_dict, n_jobs, max_jobs, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit):
    r = get_redis_server()
    
    # Convert the DataFrameGroupBy object to a binary string using pickle
    grouped_data_str = pickle.dumps(grouped_data)

    # Create a hash of the binary string using xxhash
    grouped_data_hash = xxhash.xxh64(grouped_data_str).hexdigest()

    data_hash = create_data_hash(data)
    params_hash = xxhash.xxh64(str((price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit))).hexdigest()

    cache_key = global_hyperopt_version_cache_prefix + str(grouped_data_hash) + str(data_hash) + str(params_hash)

    start_time = timemod.time()
    # Check if the result is in the cache
    result = r.get(cache_key)
    # print(grouped_data_hash, str((price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit)), cache_key, result)
    if result is not None:
        # print("Time taken for cache check: ", timemod.time() - start_time)
        print(sys.stdout.get_line_count() * n_jobs, '/', max_jobs, 'cache result:', result, 'parameters:', price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit)
        return float(result.decode('utf-8'))
    
    open_orders = []
    sells = 0
    result = 0
    best_pair = None

    # Convert grouped_data to a list
    grouped_data_list = list(grouped_data)
    last_time = grouped_data_list[-1][0]
    first_time = grouped_data_list[0][0]

    start_time = timemod.time()

    # Now you can iterate over grouped_data_skipped
    for time, group in grouped_data:
        # Skip the first 1000 entries (each entry is 5 minutes)
        if time < first_time + timedelta(minutes=1000*5):
            continue

        minute = time.minute
        hour = time.hour
        day = time.day

        # only check for buying every hour
        if minute < 5 and len(open_orders) < 10:
            pair_list = data.loc[data['open_time'] == time, 'pair_hourly'].iloc[0]
            channel_pair_cache_key = create_cache_key(grouped_data_hash, pair_list, time, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type)
            best_pair = get_best_channel_pair(data, channel_pair_cache_key, pair_list, time, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type)

            if best_pair and best_pair is not None:
                if not any(order_pair == best_pair for order_pair, _ in open_orders):
                    open_orders.append((best_pair, time))

        # start_time_inner = timemod.time()

        for order in open_orders.copy():
            order_pair, buy_time = order

            if time > buy_time and time <= last_time:
                # Look up the buy and sell data in the dictionary
                buy_data = data_dict.get((order_pair, buy_time))
                sell_data = data_dict.get((order_pair, time))

                if buy_data is None or sell_data is None:
                    open_orders.remove(order)
                    continue

                buy_price = float(buy_data[0])
                sell_price = float(sell_data[0])

                if buy_price == 0:
                    open_orders.remove(order)
                    continue

                units_bought = 1000 / buy_price
                sell_value = units_bought * sell_price

                if sell_value > 1000 * float(wished_profit) and (best_pair is None or order_pair != best_pair):
                    result += sell_value - 1000
                    sells += 1
                    open_orders.remove(order)

        # print("Time taken for open_orders: ", timemod.time() - start_time_inner)

    # print("Time taken for grouped_data_list: ", timemod.time() - start_time)
    
    # after all candles have been processed, calculate the value of open positions
    for order in open_orders:
        order_pair, buy_time = order

        # Look up the buy and end data in the dictionary
        buy_data = data_dict.get((order_pair, buy_time))
        end_data = data_dict.get((order_pair, time))

        if buy_data is None or end_data is None:
            continue

        buy_value = float(buy_data[0])
        end_value = float(end_data[0])

        units_bought = 1000 / buy_value
        end_value_units = units_bought * end_value

        sells += 1
        result += end_value_units - 1000

    print(sys.stdout.get_line_count() * n_jobs, '/', max_jobs, 'sells', sells, 'result:', result, 'parameters:', price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit)
        
    r.set(cache_key, result)

    # TODO: time in market could be another good factor to check for a final result. less time in market is better
    return result # TODO: divide the result by number of sells or something... less sells is normally better.

def get_all_data_pkl(pairs, start_time, end_time, directory):
        # Create a string that represents the function parameters
    params_str = str(pairs) + start_time.strftime('%Y%m%d') + end_time.strftime('%Y%m%d')

    # Create a hash of the parameters string
    params_hash = xxhash.xxh64(params_str.encode()).hexdigest()

    # Create a unique filename based on the hash
    filename = f"historical_channel_data/all_data_{params_hash}.pkl"

    # If the file exists, load it and return
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
           all_data = pickle.load(f)
        return all_data

    train_data = {}
    start_month = start_time.replace(day=1)
    end_month = end_time.replace(day=1)
    while start_month <= end_month:
        for pair in pairs:
            data = load_month_data(pair, start_month, directory, minimal=True)
            if data is not None:
                # Remove data that is after the end_time
                data = data.loc[data.index <= end_time]
                if pair not in train_data:
                    train_data[pair] = data
                else:
                    train_data[pair] = pd.concat([train_data[pair], data])
        start_month += relativedelta(months=1)

    # Python code
    all_data = {}
    for pair, data in train_data.items():
        data = data.reset_index()
        data['pair'] = pair
        all_data[pair] = data

    all_data = pd.concat(all_data.values(), ignore_index=True)

    # Convert 'open_time' to datetime and round to the nearest hour
    all_data['open_time'] = pd.to_datetime(all_data['open_time']).dt.round('h')

    # Calculate the list of unique pairs for each hour
    hourly_pairs = all_data.groupby('open_time')['pair'].unique().reset_index()

    # Merge 'hourly_pairs' into 'all_data'
    all_data = pd.merge(all_data, hourly_pairs, on='open_time', how='left', suffixes=('', '_hourly'))

    # Save all_data to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(all_data, f)

    return all_data

def gridsearch(pairs, start_time, end_time, directory):
    # load('gridsearch.pkl')

    all_data = get_all_data_pkl(pairs, start_time, end_time, directory)
        
    bnh_profit = calculate_buy_and_hold(all_data)

    print('Buy and Hold profit:', bnh_profit)

    param_grid = {
        'price_jump_threshold': price_jump_threshold_range,
        'last_price_treshold': last_price_treshold_range,
        'rolling_window_number': rolling_window_number_range,
        'std_for_BB': std_for_BB_range,
        'moving_average_type': moving_average_type_range,
        'wished_profit': wished_profit_range
    }

    # Create a dictionary that maps from (pair, open_time) to (close, open_time)
    data_dict = {(row['pair'], row['open_time']): (row['close'], row['open_time']) for _, row in all_data.iterrows()}

    # make data_dict unique
    data_dict = {k: v for k, v in data_dict.items() if v is not None}

    splits = 2
    n_jobs = 5 # should work with 64 GB memory and 180 days of data 

    # Calculate the total number of jobs
    total_jobs = len(ParameterGrid(param_grid)) * splits

    # Create a StrategyEstimator instance
    estimator = StrategyEstimator(n_jobs=n_jobs, max_jobs=total_jobs, data_dict=data_dict)

    # Create a TimeSeriesSplit object
    tscv = TimeSeriesSplit(n_splits=splits)

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator, param_grid, cv=tscv, n_jobs=n_jobs)

    # Fit the GridSearchCV object to the data
    sys.stdout.reset_line_count()
    grid_search.fit(all_data)

    # Get the best score and best parameters
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    # Print the score for each parameter combination
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        print(mean_score, params)

    print('Best score:', best_score)
    print('Best parameters:', best_params)

    # dump('gridsearch.pkl')


def calculate_buy_and_hold(data):
    # Get the candle data for the BTCUSDT pair
    btcusdt_data = data.loc[data['pair'] == 'BTCUSDT']

    # Get the close price of the first candle we process (index 1000)
    buy_price = float(btcusdt_data.iloc[1000]['close'])

    # Get the close price of the last candle
    sell_price = float(btcusdt_data.iloc[-1]['close'])

    print("Bought BTC for", buy_price, "and sold for", sell_price)

    # Calculate the number of units bought with 10000$ (10k because we will use 10k for trading later (1000$ per trade))
    units_bought = 10000 / buy_price

    # Calculate the sell value based on the number of units
    sell_value = units_bought * sell_price

    # Calculate the profit
    profit = sell_value - 10000

    return profit