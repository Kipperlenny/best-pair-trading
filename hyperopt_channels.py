# hyperopting.py
import argparse
import pandas as pd
import pickle
from datetime import datetime, timedelta
from binance import AsyncClient
import asyncio
import os
from dotenv import load_dotenv
import requests
import numpy as np
from plot import create_plots
from hyperopt import update_historical_data, download_historical_data, create_or_get_params_file, calculation, filter_candle_and_pair_data
from trading import get_best_channel_pair

# parameters to hyperopt
price_jump_threshold_range = np.arange(0.05, 0.15, 0.01)
last_price_treshold_range = np.arange(0.40, 0.60, 0.01)
wished_profit_range = np.arange(1.01, 1.05, 1.1)
rolling_window_number_range = range(10, 30, 2)
std_for_BB_range = np.arange(1, 3, 0.1)
moving_average_type_range = ['SMA', 'EMA']

train_test_split = 0.8

def calculate_buy_and_hold(data):
    # Get the candle data for the BTCUSDT pair
    btcusdt_data = data['BTCUSDT']

    # Get the close price of the first candle we process (index 1000)
    buy_price = float(btcusdt_data.iloc[1000]['close'])

    # Get the close price of the last candle
    sell_price = float(btcusdt_data.iloc[-1]['close'])

    # Calculate the number of units bought with 1000$
    units_bought = 1000 / buy_price

    # Calculate the sell value based on the number of units
    sell_value = units_bought * sell_price

    # Calculate the profit
    profit = sell_value - 1000

    return profit

def calculate_strategy(data, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit):
    # find best pair for each 5min candle in data, as long as 10 open orders are not reached
    # if a best pair is found, find the selling price based on wished_profit

    open_orders = []
    result = 0

    # Get the maximum length of all candle_data
    max_length = max(len(candle_data) for candle_data in data.values())

    # Loop over all candles
    for index in range(1000, max_length):

        # only check for buying every hour if we have any open_positions
        if index % 12 == 0 or len(open_orders) == 0:
            # print(f"Processing candle at index: {index} / {max_length}")

            # Initialize a dictionary to store the last 1000 candles for each pair
            last_1000_candles_dict = {}

            # For each pair, get the last 1000 candles
            for pair, candle_data in data.items():
                if index < len(candle_data):
                    # Get the current index position in the DataFrame
                    current_position = index

                    # Calculate the start position for slicing
                    start_position = max(0, current_position - 999)

                    # Get the last 1000 candles up to and including the current one
                    last_1000_candles = candle_data.iloc[start_position:current_position + 1]

                    # Store the last 1000 candles in the dictionary
                    last_1000_candles_dict[pair] = last_1000_candles

            # Pass the last 1000 candles to the function
            best_pair = get_best_channel_pair(data, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, last_1000_candles_dict)
            
            if best_pair:
                # Check if there's already an open order for the best pair
                if not any(order_pair == best_pair for order_pair, _ in open_orders):
                    print("buying", best_pair, "at", float(data[best_pair].iloc[index]["close"]))
                    open_orders.append((best_pair, index)) # store the pair and the index at which it was bought

        # check each open order for a sell opportunity
        for order in open_orders:
            order_pair, buy_index = order
            order_candle_data = data[order_pair]

            # check if the current candle is after the buy_index for this order
            if index >= buy_index:
                buy_price = float(order_candle_data.iloc[buy_index]["close"])
                sell_price = float(order_candle_data.iloc[index]["close"])

                # calculate the number of units bought
                units_bought = 1000 / buy_price

                # calculate the sell value based on the number of units
                sell_value = units_bought * sell_price

                if sell_value > 1000 * float(wished_profit):
                    print("selling", order_pair, "at", sell_price)
                    result += sell_value - 1000  # subtract the initial investment
                    open_orders.remove(order)  # remove the order from open_orders
                    break

    # after all candles have been processed, calculate the value of open positions
    for order in open_orders:
        pair, buy_index = order
        candle_data = data[pair]

        # calculate the value of the position at the end of the trading period
        end_value = float(candle_data.iloc[-1]["close"])
        buy_value = float(candle_data.iloc[buy_index]["close"])

        # calculate the number of units bought
        units_bought = 1000 / buy_value

        # calculate the end value based on the number of units
        end_value_units = units_bought * end_value

        # subtract the initial investment from the end value and add to result
        print("end selling", pair, "at", end_value)
        result += end_value_units - 1000
    
    return result

def train(train_data, file_name, rerun = False, args = {}):

    # Initialize the best score and best parameters
    best_score = float('inf')
    best_params = None
    bnh_profit = calculate_buy_and_hold(train_data)

    print('Buy and Hold profit:', bnh_profit)

    # Iterate over all combinations of parameters
    for wished_profit in wished_profit_range:
        for price_jump_threshold in price_jump_threshold_range:
            for last_price_treshold in last_price_treshold_range:
                for rolling_window_number in rolling_window_number_range:
                    for std_for_BB in std_for_BB_range:
                        for moving_average_type in moving_average_type_range:
                            # Calculate the trading strategy with the current parameters
                            strategy_result = calculate_strategy(train_data, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit)
                            
                            print("result", strategy_result, "for", price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit, "buy and hold", bnh_profit)

                            # If the score is better than the best score, update the best score and best parameters
                            if strategy_result > best_score:
                                best_score = strategy_result
                                best_params = (price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type, wished_profit)

    print('Best score:', best_score)
    print('Best parameters:', best_params)
    
    return best_params

async def main(args):

    # Initialize the Binance client
    client = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET"))

    # first we get all USDT pairs from binance
    all_pairs = await client.get_exchange_info()

    usdt_pairs = {}
    for pair in all_pairs["symbols"]:
        if pair['status'] == 'TRADING' and pair['quoteAsset'] == 'USDT' and any('SPOT' in subarray for subarray in pair['permissionSets']):
            # remove unneeded stuff from pair
            del pair['permissions']
            del pair['allowedSelfTradePreventionModes']
            del pair['defaultSelfTradePreventionMode']
            usdt_pairs[pair['symbol']] = pair

    start_time = int(datetime(2023, 10, 1).timestamp() * 1000)

    # for testing, only use the first 5 pairs
    usdt_pairs = dict(list(usdt_pairs.items())[:5]) # TODO: remove this line
    
    # Download the historical data
    data = {}
    if os.path.exists('historical_channel_data.pkl'):
        print("Updating historical data")
        # use update_historical_data from hyperopt.py
        data = await update_historical_data(client, usdt_pairs.keys(), '5m', 'historical_channel_data.pkl')
    else:
        print("Downloading historical data")
        data = await download_historical_data(client, usdt_pairs.keys(), start_time, '5m', 'historical_channel_data.pkl')

    if args.start_candle != "" and args.end_candle != "":
        print("Filter historical data to start and end args")

        # Parse the start and end candles into datetime objects
        start_candle = datetime.strptime(args.start_candle, "%Y-%m-%d %H:%M") if args.start_candle else None
        end_candle = datetime.strptime(args.end_candle, "%Y-%m-%d %H:%M") if args.end_candle else None

        # Filter the data
        for pair in data:
            pair_start_candle = data[pair].index.asof(start_candle) if start_candle else None
            pair_end_candle = data[pair].index.asof(end_candle) if end_candle else None
            data[pair] = data[pair].loc[pair_start_candle:pair_end_candle]

    # split data into training and test data
    train_data = {}
    test_data = {}
    print("Splitting data into training and test data")
    for pair in data:
        train_size = int(len(data[pair]) * train_test_split)
        train_data[pair] = data[pair].iloc[:train_size]
        test_data[pair] = data[pair].iloc[train_size:]

    print("train_data", len(train_data), "from", list(train_data.values())[0].iloc[0]["close_time"], "to", list(train_data.values())[0].iloc[-1]["close_time"], "days", (list(train_data.values())[0].iloc[-1]["close_time"] - list(train_data.values())[0].iloc[0]["close_time"]).days)
    print("test_data", len(test_data), "from", list(test_data.values())[0].iloc[0]["close_time"], "to", list(test_data.values())[0].iloc[-1]["close_time"], "days", (list(test_data.values())[0].iloc[-1]["close_time"] - list(test_data.values())[0].iloc[0]["close_time"]).days)

    filename_friendly_from = list(data.values())[0].iloc[0]["close_time"].strftime("%Y-%m-%d_%H%M%S%f")
    filename_friendly_to = list(data.values())[0].iloc[-1]["close_time"].strftime("%Y-%m-%d_%H%M%S%f")
    file_name = "df_channels_params_" + filename_friendly_from + "_" + filename_friendly_to

    print("working and saving on file:", file_name)

    # hyperopt on the train data
    print("Hyperopting on the train data")
    best_params = train(train_data, file_name + '_train', args)

    # test params on test data
    print("Testing params on test data")
    test_result = calculate_strategy(test_data, *best_params)

    # end script with success message
    print("Test result:", test_result)
    print("Hyperopting script completed successfully")

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='hyperopting script')

    parser.add_argument('--start_candle', type=str, default="", required=False, help='Start candle in the format "YYYY-MM-DD HH:MM" for the whole data')
    parser.add_argument('--end_candle', type=str, default="", required=False, help='End candle in the format "YYYY-MM-DD HH:MM" for the whole data')

    # Parse the arguments
    args = parser.parse_args()
    # Convert args.start_candle and args.end_candle to datetime objects
    start_candle_date = datetime.strptime(args.start_candle, "%Y-%m-%d %H:%M")if args.start_candle else None
    end_candle_date = datetime.strptime(args.end_candle, "%Y-%m-%d %H:%M") if args.end_candle else None

    # Compare start_candle_date to the datetime object for 2023-10-01
    if args.start_candle and start_candle_date < datetime(2023, 10, 1):
        print("Start candle must be after 2023-10-1")
        exit()

    # Compare end_candle_date to start_candle_date + 1 week
    if end_candle_date and end_candle_date < start_candle_date + timedelta(weeks=1):
        print("End candle must be at least one week after start candle")
        exit()

    # Compare end_candle_date to the current date and time
    if end_candle_date and end_candle_date > datetime.now():
        print("End candle cannot be later than now")
        exit()
        
    asyncio.run(main(args))