# backtesting.py
import pandas as pd
import pickle
from datetime import datetime, timedelta
from binance import AsyncClient, ThreadedWebsocketManager, ThreadedDepthCacheManager
from binance.exceptions import BinanceAPIException
import asyncio
import os
from dotenv import load_dotenv
import requests
import numpy as np
import matplotlib.pyplot as plt
from plot import create_plots
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ProcessPoolExecutor


# Load the .env file
load_dotenv()

# Set a global default timeout for all requests
requests.adapters.DEFAULT_RETRIES = 5

async def download_historical_data(client, pairs, start_time, interval='5m'):
    data = {}

    for pair in pairs:
        klines = await client.get_historical_klines(symbol=pair, interval=interval, start_str=start_time)

        # Convert the klines to a DataFrame
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

        # Convert the time columns to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

        # Set the open time as the index
        df.set_index('open_time', inplace=True)
        print(df)
        data[pair] = df

    # Save the data to a pickle file
    with open('historical_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

async def update_historical_data(client, pairs, interval='5m'):
    # Load the historical data from the pickle file
    with open('historical_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Get the current time
    now = datetime.now()
    something_updated = False

    for pair in pairs:
        # Get the end time of the last kline in the data
        last_time = data[pair].iloc[-1]['close_time']

        # only update if older than 24 hours
        if now - last_time < timedelta(hours=24):
            continue
        else:
            something_updated = True

        print("Updating historical data for", pair, last_time)

        # If the last time is more than one interval ago, download the new klines
        if now - last_time > timedelta(hours=int(interval[:-1])):
            klines = await client.get_historical_klines(pair, interval, int(last_time.timestamp() * 1000))

            # Convert the klines to a DataFrame and append it to the existing data
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            data[pair] = pd.concat([data[pair], df])

    # Save the updated data to the pickle file
    if something_updated:
        with open('historical_data.pkl', 'wb') as f:
            pickle.dump(data, f)

    return data

def get_best_pair(data, hour, percentile=90, min_percent_change_24=0.01, min_percent_change_1=0.01):
    
    # Filter pairs based on min_percent_change_24 and min_percent_change_1
    pairs = {}
    for pair, df in data.items():
        if hour < len(df):
            row = df.iloc[hour]
            if row['priceChangePercent'] >= min_percent_change_24 and row['priceChangePercent1h'] >= min_percent_change_1:
                pairs[pair] = {'priceChangePercent': row['priceChangePercent'], 'priceChangePercent1h': row['priceChangePercent1h'], 'close': row['close']}

    if not pairs:
        return None
    print(str(datetime.now()), "done: Filter pairs based on min_percent_change_24 and min_percent_change_1")
    # Calculate the threshold for the top percentile of 24-hour price changes
    threshold = np.percentile([values['priceChangePercent'] for values in pairs.values()], percentile)
    print(str(datetime.now()), "done Calculate the threshold for the top percentile of 24-hour price changes")
    # Filter out the pairs that don't have a high 24-hour price change
    best_pairs = [(pair, values) for pair, values in pairs.items() if values['priceChangePercent'] >= threshold]
    print(str(datetime.now()), "done Filter out the pairs that don't have a high 24-hour price change")
    if not best_pairs:
        return None

    # Return the pair with the highest 1-hour price change
    return max(best_pairs, key=lambda pair: pair[1]['priceChangePercent1h'])

def get_best_pair_for_hour(args):
    data, hour, percentile, min_percent_change_24, min_percent_change_1 = args
    return get_best_pair(data, hour, percentile, min_percent_change_24, min_percent_change_1)

# Define a function that performs the backtesting for a given percentile and profit margin
def backtest(row, data, df_params):
    # Define the settings
    settings = {
        'percentile': row['percentile'],
        'profit_margin': row['profit_margin'],
        'min_percent_change_24': row['min_percent_change_24'],
        'min_percent_change_1': row['min_percent_change_1']
    }

    # Check if there's a row in df_params with the same settings
    matching_rows = df_params[
        (df_params['percentile'] == settings['percentile']) &
        (df_params['profit_margin'] == settings['profit_margin']) &
        (df_params['min_percent_change_24'] == settings['min_percent_change_24']) &
        (df_params['min_percent_change_1'] == settings['min_percent_change_1'])
    ]

    if not matching_rows.empty:
        # If such a row exists, check if it has all the results
        result_columns = ['overall_profit_24', 'overall_profit_72', 'total_hours_24', 'total_hours_72', 'over_24_hours', 'over_72_hours', 'open_positions', 'max_open_positions']
        if not matching_rows[result_columns].isnull().any().any():
            # If it has all the results, skip the current settings and return
            return # pd.DataFrame(np.nan, index=[0], columns=result_columns)
        
    percentile, profit_margin, min_percent_change_24, min_percent_change_1 = row['percentile'], row['profit_margin'], row['min_percent_change_24'], row['min_percent_change_1']

    # Get the maximum number of hours in the data
    first_df = next(iter(data.values()))
    max_hours = len(first_df) // 12

    # Subtract 7 days from max_hours
    max_hours -= 168
    over_24_hours = 0
    over_72_hours = 0
    open_positions = 0
    max_open_positions = 0
    overall_profit_24 = 0
    overall_profit_72 = 0
    total_hours_24 = 0
    total_hours_72 = 0

    print("\n" + str(datetime.now()), "testing", row['percentile'], row['profit_margin'], row['min_percent_change_24'], row['min_percent_change_1'], "for", max_hours, "hours")

    # max_hours = 100 # for testing

    with ProcessPoolExecutor() as executor:
        args = [(data, hour, percentile, min_percent_change_24, min_percent_change_1) for hour in range(max_hours)]
        best_pairs = list(executor.map(get_best_pair_for_hour, args))

    print(str(datetime.now()), "best_pairs calculated")

    # For each best_pair
    hour = 0
    symbol_bought_last = None
    for best_pair in best_pairs:
        hour += 12
        if best_pair is not None:
            best_pair_symbol = best_pair[0]
            df = data[best_pair_symbol]

            # Calculate the buy time
            try:
                buy_time = df.index[hour]
            except IndexError:
                continue

            # do not buy if already in wallet
            if symbol_bought_last == best_pair_symbol:
                # print("already bought", best_pair_symbol, "at", best_pair[1]['close'], "skipping")
                continue
            else:
                symbol_bought_last = best_pair_symbol

            bought_price = best_pair[1]['close']
            asked_sell_price = bought_price * profit_margin

            open_positions += 1

            # print(buy_time, best_pair_symbol, "bought at", bought_price, "for", asked_sell_price, "profit margin")

            # Calculate the number of shares that can be bought with an investment of 100
            shares = 100 / bought_price

            if open_positions > max_open_positions:
                max_open_positions = open_positions

            # calculate how many hours it would have taken to sell with profit_margin profit
            sell_time = (df.iloc[hour + 1:]['close'].ge(asked_sell_price)).idxmax()
            if df.iloc[hour + 1:]['close'].ge(asked_sell_price).any() and sell_time and (sell_time - buy_time).total_seconds() <= 24*60*60: # after 24 hours it's a loss
                # Calculate the hours between buy and sell times and add to total_hours
                hours = (sell_time - df.index[hour]) / np.timedelta64(1, 'h')
                total_hours_24 += hours

                # add profit to overall_profit
                overall_profit_24 += shares * (asked_sell_price - bought_price)
                over_24_hours += 1

                open_positions -= 1
                symbol_bought_last = None

                # print(sell_time, "sold at", asked_sell_price, "for", shares * (asked_sell_price - bought_price), "profit")
            elif df.iloc[hour + 1:]['close'].ge(asked_sell_price).any() and sell_time and (sell_time - buy_time).total_seconds() <= 72*60*60: # after 72 hours it's a loss
                # Calculate the hours between buy and sell times and add to total_hours
                hours = (sell_time - df.index[hour]) / np.timedelta64(1, 'h')
                total_hours_72 += hours
                
                # add profit to overall_profit
                overall_profit_72 += shares * (asked_sell_price - bought_price)
                over_72_hours += 1

                open_positions -= 1
                symbol_bought_last = None

                # print(sell_time, "sold at", asked_sell_price, "for", shares * (asked_sell_price - bought_price), "profit")
            elif df.iloc[hour + 1:]['close'].ge(asked_sell_price).any() and sell_time: # after 72 hours it's a loss
                hours = (sell_time - df.index[hour]) / np.timedelta64(1, 'h')
                total_hours_72 += hours
                open_positions -= 1
                over_72_hours += 1
                symbol_bought_last = None

            else:
                # print(best_pair_symbol, "for hour", hour, "no sell time found")
                # overall_profit -= 100 # penalty for not selling
                # total_hours += 16800 # penalty for not selling
                over_72_hours += 1

    # reduce overall_profit by invested capital
    # overall_profit -= 100 * max_hours

    # round overall profit to two decimals
    overall_profit_24 = round(overall_profit_24, 2)
    overall_profit_72 = round(overall_profit_72 + overall_profit_24, 2)

    # round total_hours to full int
    total_hours_24 = int(total_hours_24)
    total_hours_72 = int(total_hours_72 + total_hours_24)

    # Return overall_profit and hours
    print(str(datetime.now()), "result:", overall_profit_24, "profit(24)", overall_profit_72, "profit(72)", total_hours_24, "hours(24)", total_hours_72, "hours(72)", open_positions, "open positions at the end.", max_open_positions, "max open positions")

    # save values to correct df_params row
    df_params.loc[
        (df_params['percentile'] == settings['percentile']) &
        (df_params['profit_margin'] == settings['profit_margin']) &
        (df_params['min_percent_change_24'] == settings['min_percent_change_24']) &
        (df_params['min_percent_change_1'] == settings['min_percent_change_1']),
        ['overall_profit_24', 'overall_profit_72', 'total_hours_24', 'total_hours_72', 'over_24_hours', 'over_72_hours', 'open_positions', 'max_open_positions']
    ] = [overall_profit_24, overall_profit_72, total_hours_24, total_hours_72, over_24_hours, over_72_hours, open_positions, max_open_positions]
    
    df_params.to_pickle("df_params.pkl")

    # Find the row with the highest overall profit
    best_row = df_params.loc[df_params['overall_profit_24'].idxmax()]

    print(f"The best percentile is {best_row['percentile']} and the best profit margin is {best_row['profit_margin']} and the best min_percent_change_24 is {best_row['min_percent_change_24']}  and the best min_percent_change_1 is {best_row['min_percent_change_1']}  with an overall profit of {best_row['overall_profit_24']}", best_row)

    # return overall_profit_24, overall_profit_72, total_hours_24, total_hours_72, over_24_hours, over_72_hours, open_positions, max_open_positions

async def main():

    # Initialize the Binance client
    client = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET"))

    # Create a semaphore with a limit of 10 concurrent requests
    semaphore = asyncio.Semaphore(10)

    # first we get all USDT pairs from binance
    all_pairs = await client.get_exchange_info()

    usdt_pairs = {}
    for pair in all_pairs["symbols"]:
        if pair['status'] == 'TRADING' and pair['quoteAsset'] == 'USDT' and 'SPOT' in pair['permissions']:
            # remove unneeded stuff from pair
            del pair['permissions']
            del pair['allowedSelfTradePreventionModes']
            del pair['defaultSelfTradePreventionMode']
            usdt_pairs[pair['symbol']] = pair

    start_time = int(datetime(2023, 10, 1).timestamp() * 1000)
    
    # Download the historical data
    data = {}
    if os.path.exists('historical_data.pkl'):
        print("Updating historical data")
        data = await update_historical_data(client, usdt_pairs.keys())
    else:
        data = await download_historical_data(client, usdt_pairs.keys(), start_time)

    # calculate the 24h change for each pair
    print("Calculating 24h change for each pair")
    for pair in data:
        data[pair]["close"] = pd.to_numeric(data[pair]["close"])
        data[pair]["priceChangePercent"] = (data[pair]["close"].shift(12).pct_change(periods=288) * 100).round(2)
        data[pair]["priceChangePercent1h"] = (data[pair]["close"].pct_change(periods=12).round(3) * 100).round(2)
        data[pair] = data[pair][['close', 'priceChangePercent', 'priceChangePercent1h']]

    # drop all rows with NaN values
    for pair in data:
        data[pair].dropna(inplace=True)
    
    print("Running backtest")

    # Create a DataFrame with all combinations of percentiles and profit margins
    percentiles = [50, 70, 90]
    profit_margins = [1.01, 1.03, 1.06, 1.10, 1.15]
    min_percent_change_24 = [1, 3, 5, 10, 20, 30]
    min_percent_change_1 = [0.1, 0.5, 1, 1.5, 2, 5]

    # Check if df_params.pkl exists
    if os.path.exists('df_params.pkl'):
        # If it exists, load it
        df_params = pd.read_pickle('df_params.pkl')
    else:
        # If it doesn't exist, create df_params
        df_params = pd.DataFrame([(p, m, m24, m1) for p in percentiles for m in profit_margins for m24 in min_percent_change_24 for m1 in min_percent_change_1], columns=['percentile', 'profit_margin', 'min_percent_change_24', 'min_percent_change_1'])

        # Initialize the result columns with np.nan
        df_params['overall_profit_24'] = np.nan
        df_params['overall_profit_72'] = np.nan
        df_params['total_hours_24'] = np.nan
        df_params['total_hours_72'] = np.nan
        df_params['over_24_hours'] = np.nan
        df_params['over_72_hours'] = np.nan
        df_params['open_positions'] = np.nan
        df_params['max_open_positions'] = np.nan
    
    # print how many rows are left to calculate (all with NaN values)
    print("Calculating", df_params['overall_profit_24'].isna().sum(), "rows")

    '''print(data['AEURUSDT']['2023-12-05':'2023-12-09'])

    # Assuming 'close' is the column with closing prices and 'date' is the datetime column
    plt.figure(figsize=(10, 6))
    plt.plot(data['AEURUSDT'].index, data['AEURUSDT']['close'])
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Close Price Over Time')
    plt.savefig("AEURUSDT.png")
    exit()'''

    # Apply the backtest function to df_params
    df_params.apply(backtest, args=(data, df_params), axis=1, result_type='expand')

    # Find the row with the highest overall profit
    best_row = df_params.loc[df_params['overall_profit_24'].idxmax()]
            
    # Set the maximum number of rows displayed to a large number
    pd.set_option('display.max_rows', None)

    print(df_params)
    print(best_row)

    print(f"The best percentile is {best_row['percentile']} and the best profit margin is {best_row['profit_margin']} and the best min_percent_change_24 is {best_row['min_percent_change_24']}  and the best min_percent_change_1 is {best_row['min_percent_change_1']}  with an overall profit of {best_row['overall_profit_24']}")


    # Close the client session
    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())