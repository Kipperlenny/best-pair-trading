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

# Load the .env file
load_dotenv()

# Set a global default timeout for all requests
requests.adapters.DEFAULT_RETRIES = 5

# Define a function to apply the backtest function to a chunk of df_params
def apply_backtest_to_chunk(chunk, data, df_params):
    return chunk.apply(backtest, args=(data, df_params), axis=1, result_type='expand')

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

    for pair in pairs:
        # Get the end time of the last kline in the data
        last_time = data[pair].iloc[-1]['close_time']

        # only update if older than 24 hours
        if now - last_time < timedelta(hours=24):
            continue

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
    with open('historical_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

def get_best_pair(pairs, percentile=90, min_percent_change_24=0.01, min_percent_change_1=0.01):
    # remove pairs with less than min_percent_change 24h change
    pairs = {pair: values for pair, values in pairs.items() if float(values['priceChangePercent']) >= min_percent_change_24}

    if not pairs:
        return None

    # remove pairs with less than min_percent_change 1h change
    pairs = {pair: values for pair, values in pairs.items() if float(values['priceChangePercent1h']) >= min_percent_change_1}

    if not pairs:
        return None

    # Sort the pairs by the 1-hour price change in descending order
    sorted_pairs = sorted(pairs.items(), key=lambda pair: float(pair[1]['priceChangePercent1h']), reverse=True)

    # Calculate the threshold for the top percentile of 24-hour price changes
    threshold = np.percentile([float(pair[1]['priceChangePercent']) for pair in pairs.items()], percentile)

    # Filter out the pairs that don't have a high 24-hour price change
    best_pairs = [(pair[0], pair[1]) for pair in sorted_pairs if float(pair[1]['priceChangePercent']) >= threshold]

    # Return the pair with the highest 1-hour price change
    return best_pairs[0] if best_pairs else None

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
            return pd.DataFrame(np.nan, index=[0], columns=result_columns)
        
    percentile, profit_margin, min_percent_change_24, min_percent_change_1 = row['percentile'], row['profit_margin'], row['min_percent_change_24'], row['min_percent_change_1']

    overall_profit_24 = 0
    overall_profit_72 = 0
    
    # Initialize total_hours to 0
    total_hours_24 = 0
    total_hours_72 = 0

    # Get the maximum number of hours in the data
    max_hours = max(len(df) // 12 for df in data.values())

    # Subtract 7 days from max_hours
    max_hours -= 168

    over_24_hours = 0
    over_72_hours = 0

    open_positions = 0
    max_open_positions = 0

    print("\ntesting", row['percentile'], row['profit_margin'], row['min_percent_change_24'], row['min_percent_change_1'], "for", max_hours, "hours")

    # For each hour
    for hour in range(max_hours):
        # if too many open positions, this has failed
        # if open_positions > 20:
        #    print("too many open positions")
        #    break

        # Calculate the 1-hour and 24-hour price changes for each pair
        pairs = {pair: {'priceChangePercent1h': df.iloc[hour]['priceChangePercent1h'] if hour < len(df) else np.nan, 
                        'priceChangePercent': df.iloc[hour]['priceChangePercent'] if hour < len(df) else np.nan} 
                 for pair, df in data.items()}
        
        # Remove pairs with NaN values
        pairs = {pair: values for pair, values in pairs.items() if not any(np.isnan(val) for val in values.values())}

        # Find the best pair for this hour
        best_pair = get_best_pair(pairs, percentile, min_percent_change_24, min_percent_change_1)
        if best_pair is not None:
            best_pair_symbol = best_pair[0]
            df = data[best_pair_symbol]
            bought_price = df.iloc[hour]['close']
            asked_sell_price = bought_price * profit_margin

            open_positions += 1

            # Calculate the buy time
            buy_time = df.index[hour]

            # Calculate the number of shares that can be bought with an investment of 100
            shares = 100 / bought_price

            # calculate how many hours it would have taken to sell with profit_margin profit
            sell_time = (df.iloc[hour + 1:]['close'].ge(asked_sell_price)).idxmax()
            if df.iloc[hour + 1:]['close'].gt(bought_price).any() and sell_time and (sell_time - buy_time).total_seconds() <= 24*60*60: # after 24 hours it's a loss
                # Calculate the hours between buy and sell times and add to total_hours
                hours = (sell_time - df.index[hour]) / np.timedelta64(1, 'h')
                total_hours_24 += hours
                
                # add profit to overall_profit
                overall_profit_24 += shares * (asked_sell_price - bought_price)

                open_positions -= 1
            elif df.iloc[hour + 1:]['close'].gt(bought_price).any() and sell_time and (sell_time - buy_time).total_seconds() <= 72*60*60: # after 72 hours it's a loss
                # Calculate the hours between buy and sell times and add to total_hours
                hours = (sell_time - df.index[hour]) / np.timedelta64(1, 'h')
                total_hours_72 += hours
                
                # add profit to overall_profit
                overall_profit_72 += shares * (asked_sell_price - bought_price)
                over_72_hours += 1

                open_positions -= 1
            elif df.iloc[hour + 1:]['close'].gt(bought_price).any() and sell_time: # after 72 hours it's a loss
                hours = (sell_time - df.index[hour]) / np.timedelta64(1, 'h')
                total_hours_72 += hours
                open_positions -= 1
                over_72_hours += 1
            else:
                # print(best_pair_symbol, "for hour", hour, "no sell time found")
                # overall_profit -= 100 # penalty for not selling
                # total_hours += 16800 # penalty for not selling
                over_72_hours += 1

        if open_positions > max_open_positions:
            max_open_positions = open_positions

    # reduce overall_profit by invested capital
    # overall_profit -= 100 * max_hours

    # round overall profit to two decimals
    overall_profit_24 = round(overall_profit_24, 2)
    overall_profit_72 = round(overall_profit_72 + overall_profit_24, 2)

    # round total_hours to full int
    total_hours_24 = int(total_hours_24)
    total_hours_72 = int(total_hours_72 + total_hours_24)

    # Return overall_profit and hours
    print("result:", overall_profit_24, "profit(24)", overall_profit_72, "profit(72)", total_hours_24, "hours(24)", total_hours_72, "hours(72)", open_positions, "open positions at the end.", max_open_positions, "max open positions")
    
    return overall_profit_24, overall_profit_72, total_hours_24, total_hours_72, over_24_hours, over_72_hours, open_positions, max_open_positions

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
        data[pair]["priceChangePercent"] = data[pair]["close"].pct_change(periods=288)
        data[pair]["priceChangePercent1h"] = data[pair]["close"].pct_change(periods=12)

    # drop all rows with NaN values
    for pair in data:
        data[pair].dropna(inplace=True)

    '''
    # for every hour in ever DF in data, we select the best pair with get_best_pair() and check after how many hours we would have sold with 3% profit
    # we then run it again with a different percentile paramter to see how it changes
    percentiles = range(80, 100)  # replace with the range of percentiles you want to test
    profit_margins = [i / 100 for i in range(101, 111)]  # creates a list [1.01, 1.02, ..., 1.10]

    best_overall_profit = 0
    best_percentile = None
    best_profit_margin = None

    for percentile in percentiles:
        for profit_margin in profit_margins:
            overall_profit = 0
            for pair, df in data.items():
                df_dropped_na = df.dropna(subset=['priceChangePercent'])
                for hour, row in df_dropped_na.iterrows():
                    best_pair = get_best_pair(df, percentile)
                    if best_pair:
                        # calculate how many hours it would have taken to sell with profit_margin profit
                        sell_time = 0
                        for i in range(hour, len(df)):
                            if df.iloc[i]['close'] > df.iloc[hour]['close'] * profit_margin:
                                sell_time = i
                                # add profit to overall_profit
                                overall_profit += df.iloc[sell_time]['close'] - df.iloc[hour]['close']
                                print(f"Pair: {pair} bought at {df.iloc[hour]['close']} and sold at {df.iloc[sell_time]['close']} after {sell_time - hour} hours for a profit of {df.iloc[sell_time]['close'] - df.iloc[hour]['close']}")
                                break
            if overall_profit > best_overall_profit:
                best_overall_profit = overall_profit
                best_percentile = percentile
                best_profit_margin = profit_margin

    print(f"The best percentile is {best_percentile} and the best profit margin is {best_profit_margin} with an overall profit of {best_overall_profit}")
    '''

    print("Running backtest")

    '''
        percentile  profit_margin  overall_profit    total_hours
    0           95           1.01      165.205270  197102.750000
    1           95           1.02      268.707313  261372.500000
    2           95           1.03      362.496960  328234.750000
    3           95           1.04      450.505411  397228.666667
    4           96           1.01      166.990118  201003.000000
    5           96           1.02      259.229385  259271.500000
    6           96           1.03      353.967689  329152.416667
    7           96           1.04      441.847971  392900.416667
    8           97           1.01      185.446738  192885.166667
    9           97           1.02      282.229302  252212.916667
    10          97           1.03      391.439703  320912.916667
    11          97           1.04      505.325740  380357.666667
    12          98           1.01      208.396776  192630.333333
    13          98           1.02      308.653700  253066.333333
    14          98           1.03      420.844829  322382.333333
    15          98           1.04      560.576355  385820.916667
    16          99           1.01      219.003579  181407.500000
    17          99           1.02      315.194761  242095.666667
    18          99           1.03      413.590521  303419.416667
    19          99           1.04      539.747701  350310.750000
    percentile            98.000000
    profit_margin          1.040000
    overall_profit       560.576355
    total_hours       385820.916667
    Name: 15, dtype: float64
    The best percentile is 98.0 and the best profit margin is 1.04 with an overall profit of 560.5763550400001
    '''


    # Create a DataFrame with all combinations of percentiles and profit margins
    percentiles = [80, 90, 100]
    profit_margins = [i / 100 for i in range(101, 135)]
    min_percent_change_24 = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5]
    min_percent_change_1 = [0.001, 0.01, 0.03, 0.05, 0.1, 0.5]

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
        
    # Split df_params into 12 chunks
    chunks = np.array_split(df_params, 3)

    # Create a partial function with data pre-filled
    apply_backtest_to_chunk_partial = partial(apply_backtest_to_chunk, data=data, df_params=df_params)

    # Create a pool of worker processes
    with Pool() as p:
        # Apply the backtest function to each chunk using the pool
        results = p.map(apply_backtest_to_chunk_partial, chunks)

    # Filter out "empty" results
    results = [result for result in results if not result.isnull().all().all()]

    # Combine the results back into df_params
    df_params = pd.concat(results)

    # Find the row with the highest overall profit
    best_row = df_params.loc[df_params['overall_profit_24'].idxmax()]

    print(df_params)
    print(best_row)

    print(f"The best percentile is {best_row['percentile']} and the best profit margin is {best_row['profit_margin']} and the best min_percent_change_24 is {best_row['min_percent_change_24']}  and the best min_percent_change_1 is {best_row['min_percent_change_1']}  with an overall profit of {best_row['overall_profit_24']}")

    df_params.to_pickle("df_params.pkl")

    # Close the client session
    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())