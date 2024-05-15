from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator
import pandas as pd
from dateutil.relativedelta import relativedelta
from trading import get_best_channel_pair
from historic_data import load_month_data
import numpy as np
from sklearn.model_selection import GridSearchCV


# parameters to hyperopt
price_jump_threshold_range = np.arange(0.05, 0.15, 0.01)
last_price_treshold_range = np.arange(0.40, 0.60, 0.01)
wished_profit_range = np.arange(1.01, 1.05, 1.1)
low_to_high_threshold_range = [5, 10, 50, 100, 250]
std_dev_threshold_range = [10, 50, 150, 300, 600]
rolling_window_number_range = range(10, 30, 2)
std_for_BB_range = np.arange(1, 3, 0.1)
moving_average_type_range = ['SMA', 'EMA']

class StrategyEstimator(BaseEstimator):
    def __init__(self, price_jump_threshold=None, last_price_treshold=None, rolling_window_number=None, std_for_BB=None, moving_average_type=None, wished_profit=None, low_to_high_threshold=None, std_dev_threshold=None):
        self.price_jump_threshold = price_jump_threshold
        self.last_price_treshold = last_price_treshold
        self.rolling_window_number = rolling_window_number
        self.std_for_BB = std_for_BB
        self.moving_average_type = moving_average_type
        self.wished_profit = wished_profit
        self.low_to_high_threshold = low_to_high_threshold
        self.std_dev_threshold = std_dev_threshold

    def fit(self, X, y=None):
        self.strategy_result_ = calculate_strategy(X, self.price_jump_threshold, self.last_price_treshold, self.rolling_window_number, self.std_for_BB, self.moving_average_type, self.wished_profit, self.low_to_high_threshold, self.std_dev_threshold)
        return self

    def score(self, X, y=None):
        return self.strategy_result_
    
    def predict(self, X):
        # Assuming that the strategy result can be calculated for new data as well
        return calculate_strategy(X, self.price_jump_threshold, self.last_price_treshold, self.rolling_window_number, self.std_for_BB, self.moving_average_type, self.wished_profit, self.low_to_high_threshold, self.std_dev_threshold)
    
def calculate_strategy(data, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit, low_to_high_threshold, std_dev_threshold):
    # find best pair for each 5min candle in data, as long as 10 open orders are not reached
    # if a best pair is found, find the selling price based on wished_profit

    open_orders = []
    result = 0

    # Get the maximum length of all candle_data
    max_length = len(data)

    # Initialize best_pair before the loop
    best_pair = None

    # Loop over all candles
    for index in range(1000, max_length):

        # only check for buying every hour
        if index % 12 == 0:
            print(data.iloc[index]["close_time"])

            # Initialize a dictionary to store the last 1000 candles for each pair
            last_1000_candles_dict = {}

            # Get the last 1000 candles
            start_position = max(0, index - 999)
            last_1000_candles = data.iloc[start_position:index + 1]

            # Pass the last 1000 candles to the function
            best_pair = get_best_channel_pair(last_1000_candles, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, low_to_high_threshold, std_dev_threshold)
            
            if best_pair:
                # Check if there's already an open order for the best pair
                if not any(order_pair == best_pair for order_pair, _ in open_orders):
                    print("buying", best_pair, "at", float(data.iloc[index]["close"]))
                    open_orders.append((best_pair, index))  # store the pair and the index at which it was bought

        for order in open_orders.copy():
            order_pair, buy_index = order

            # check if the current candle is after the buy_index for this order
            if index >= buy_index:
                buy_price = float(data.iloc[buy_index]["close"])
                sell_price = float(data.iloc[index]["close"])

                if buy_price == 0:
                    open_orders.remove(order)  # remove the order from open_orders
                    continue

                # calculate the number of units bought
                units_bought = 1000 / buy_price

                # calculate the sell value based on the number of units
                sell_value = units_bought * sell_price

                if sell_value > 1000 * float(wished_profit) and (best_pair is None or order_pair != best_pair):
                    print("selling", order_pair, "at", sell_price)
                    result += sell_value - 1000  # subtract the initial investment
                    open_orders.remove(order)  # remove the order from open_orders
                    break

    # after all candles have been processed, calculate the value of open positions
    for order in open_orders:
        pair, buy_index = order

        # calculate the value of the position at the end of the trading period
        end_value = float(data.iloc[-1]["close"])
        buy_value = float(data.iloc[buy_index]["close"])

        # calculate the number of units bought
        units_bought = 1000 / buy_value

        # calculate the end value based on the number of units
        end_value_units = units_bought * end_value

        # subtract the initial investment from the end value and add to result
        print("end selling", pair, "at", end_value)
        result += end_value_units - 1000
    
    return result

def gridsearch(pairs, start_time, end_time, directory):
    train_data = {}
    start_month = start_time.replace(day=1)
    end_month = end_time.replace(day=1)
    while start_month <= end_month:
        for pair in pairs:
            data = load_month_data(pair, start_month, directory)
            if data is not None:
                if pair not in train_data:
                    train_data[pair] = data
                else:
                    train_data[pair] = pd.concat([train_data[pair], data])
        start_month += relativedelta(months=1)

    for pair, data in train_data.items():
        data['pair'] = pair
    all_data = pd.concat(train_data.values(), ignore_index=True)

    param_grid = {
        'price_jump_threshold': price_jump_threshold_range,
        'last_price_treshold': last_price_treshold_range,
        'rolling_window_number': rolling_window_number_range,
        'std_for_BB': std_for_BB_range,
        'moving_average_type': moving_average_type_range,
        'wished_profit': wished_profit_range,
        'low_to_high_threshold': low_to_high_threshold_range,
        'std_dev_threshold': std_dev_threshold_range
    }

    # Create a StrategyEstimator instance
    estimator = StrategyEstimator()

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator, param_grid, cv=5, n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(all_data)

    # Get the best score and best parameters
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_

    print('Best score:', best_score)
    print('Best parameters:', best_params)