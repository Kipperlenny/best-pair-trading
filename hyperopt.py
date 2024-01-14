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
from concurrent.futures import ProcessPoolExecutor

# Load the .env file
load_dotenv()

# Set a global default timeout for all requests
requests.adapters.DEFAULT_RETRIES = 5

train_test_split = 0.8
sell_period = 0.1

# Define a weight for open_positions
weight_open_positions = 5

# Define a weight for overall_profit_24
weight_overall_profit_24 = 0.15

# Define a weight for total_hours_24
weight_total_hours_24 = 0.03

# Define a weight for total_hours_24
weight_total_hours_72 = 1

# Initialize the global variable outside the function (do not change)
this_row_count = 0

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
        # check if pair is in data
        if pair not in data:
            continue

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

def get_best_pair(pair_data, hour, min_percent_change_24=0.01, min_percent_change_1=0.01, max_percent_change_24=0.01, max_percent_change_1=0.01):
    
   # Filter pairs based on min_percent_change_24 and min_percent_change_1
    pairs = {}
    for pair, df in pair_data.items():
        # Filter df to include only rows where the hour part of the index matches the desired hour
        df_hour = df[df.index.hour == hour]
        if not df_hour.empty:
            row = df_hour.iloc[0]
            if row['priceChangePercent'] >= min_percent_change_24 and row['priceChangePercent1h'] >= min_percent_change_1 and row['priceChangePercent'] <= max_percent_change_24 and row['priceChangePercent1h'] <= max_percent_change_1:
                pairs[pair] = {'priceChangePercent': row['priceChangePercent'], 'priceChangePercent1h': row['priceChangePercent1h'], 'close': row['close'], 'open_time': df_hour.index[0]}

    if not pairs:
        return None

    # Find the pair with the highest 1-hour price change
    best_pair = max(pairs, key=lambda pair: pairs[pair]['priceChangePercent1h'])

    # Return the best pair, its 'close' price, and the 'open_time'
    return {best_pair: {'close': pairs[best_pair]['close'], 'open_time': pairs[best_pair]['open_time']}}

def get_best_pair_for_hour(args):
    pair_data, hour, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1 = args
    return get_best_pair(pair_data, hour, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1)

def calculation(row, file_name, sum_rows, candle_data, pair_data, df_params, rerun = False):
    global this_row_count

    # Define the settings
    settings = {
        'profit_margin': row['profit_margin'],
        'min_percent_change_24': row['min_percent_change_24'],
        'min_percent_change_1': row['min_percent_change_1'],
        'max_percent_change_24': row['max_percent_change_24'],
        'max_percent_change_1': row['max_percent_change_1']
    }

    # Check if there's a row in df_params with the same settings
    matching_rows = df_params[
        (df_params['profit_margin'] == settings['profit_margin']) &
        (df_params['min_percent_change_24'] == settings['min_percent_change_24']) &
        (df_params['min_percent_change_1'] == settings['min_percent_change_1']) &
        (df_params['max_percent_change_24'] == settings['max_percent_change_24']) &
        (df_params['max_percent_change_1'] == settings['max_percent_change_1'])
    ]

    if not matching_rows.empty:
        # If such a row exists, check if it has all the results
        result_columns = ['overall_profit_24', 'overall_profit_72', 'total_hours_24', 'total_hours_72', 'over_24_hours', 'over_72_hours', 'open_positions', 'max_open_positions', 'trades']
        if not matching_rows[result_columns].isnull().any().any():
            # If it has all the results, skip the current settings and return
            return # pd.DataFrame(np.nan, index=[0], columns=result_columns)
        
    profit_margin, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1 = row['profit_margin'], row['min_percent_change_24'], row['min_percent_change_1'], row['max_percent_change_24'], row['max_percent_change_1']

    # Get the maximum number of hours in the data
    first_df = next(iter(candle_data.values()))
    max_hours = len(first_df) // 12

    # Subtract 7 days from max_hours
    max_hours -= int(max_hours * sell_period)
    over_24_hours = 0
    over_72_hours = 0
    open_positions = 0
    max_open_positions = 0
    overall_profit_24 = 0
    overall_profit_72 = 0
    total_hours_24 = 0
    total_hours_72 = 0
    trades = 0

    print("\n" + str(datetime.now()), "testing row", this_row_count, "/", sum_rows, "settings:", row['profit_margin'], row['min_percent_change_24'], row['min_percent_change_1'], row['max_percent_change_24'], row['max_percent_change_1'], "for", max_hours, "hours")

    # max_hours = 100 # for testing

    with ProcessPoolExecutor() as executor:
        # Concatenate all dataframes in pair_data
        all_data = pd.concat(pair_data.values())
        
        # Get unique hours
        hours = all_data.index.hour.unique()
        
        args = [(pair_data, hour, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1) for hour in hours]
        best_pairs = list(executor.map(get_best_pair_for_hour, args))

    # print(str(datetime.now()), "best_pairs calculated")
        
    # For each best_pair
    symbol_bought_last = None
    for best_pair in best_pairs:
        if best_pair is not None:
            best_pair_symbol, best_pair_info = list(best_pair.items())[0]
            bought_price = best_pair_info['close']
            open_time = best_pair_info['open_time']
            df = candle_data[best_pair_symbol]

            # do not buy if already in wallet
            if symbol_bought_last == best_pair_symbol:
                print("already bought", best_pair_symbol, "at", bought_price, "skipping")
                continue
            else:
                symbol_bought_last = best_pair_symbol

            asked_sell_price = bought_price * profit_margin

            open_positions += 1
            trades += 1

            if rerun:
                print(open_time, best_pair_symbol, "bought at", bought_price, "for", asked_sell_price, "profit margin")

            # Calculate the number of shares that can be bought with an investment of 100
            shares = 100 / bought_price

           # Get the location of open_time
            loc = df.index.get_loc(open_time)

            # Check if open_time is the last element in the index
            if loc + 1 < len(df.index):
                # If not, get the index of the next element
                next_index = df.index[loc + 1]
            else:
                # If so, handle this case as needed
                next_index = None

            # calculate how many hours it would have taken to sell with profit_margin profit
            sell_time = (df.loc[next_index:]['close'].ge(asked_sell_price)).idxmax()
            any_sell = (df.loc[next_index:]['close'].ge(asked_sell_price)).any()
            if any_sell and sell_time and (sell_time - open_time).total_seconds() <= 24*60*60: # after 24 hours it's a loss
                # Calculate the hours between buy and sell times and add to total_hours
                hours = (sell_time - open_time) / np.timedelta64(1, 'h')
                total_hours_24 += hours

                # add profit to overall_profit
                overall_profit_24 += shares * (asked_sell_price - bought_price)
                over_24_hours += 1

                open_positions -= 1
                symbol_bought_last = None

                if rerun:
                    print(sell_time, "sold at", asked_sell_price, "for", shares * (asked_sell_price - bought_price), "profit")
            elif any_sell and sell_time and (sell_time - open_time).total_seconds() <= 72*60*60: # after 72 hours it's a loss
                # Calculate the hours between buy and sell times and add to total_hours
                hours = (sell_time - open_time) / np.timedelta64(1, 'h')
                total_hours_72 += hours
                total_hours_24 += hours
                
                # add profit to overall_profit
                overall_profit_72 += shares * (asked_sell_price - bought_price)
                over_72_hours += 1

                open_positions -= 1
                symbol_bought_last = None

                if rerun:
                    print(sell_time, "sold at", asked_sell_price, "for", shares * (asked_sell_price - bought_price), "profit")
            elif any_sell and sell_time: # after 72 hours it's a loss
                hours = (sell_time - open_time) / np.timedelta64(1, 'h')
                total_hours_72 += hours
                total_hours_24 += hours
                open_positions -= 1
                over_72_hours += 1
                symbol_bought_last = None

                if rerun:
                    print(sell_time, "sold at", asked_sell_price, "for", shares * (asked_sell_price - bought_price), "profit, but too late")
            else:
                if rerun:
                    print(best_pair_symbol, "for", open_time, "no sell time found")
                overall_profit_24 -= 100 # penalty for not selling
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
    total_hours_72 = int(total_hours_72)

    # Return overall_profit and hours
    print(str(datetime.now()), "result:", overall_profit_24, "profit(24)", overall_profit_72, "profit(72)", total_hours_24, "hours(24)", total_hours_72, "hours(72)", open_positions, "open positions at the end.", max_open_positions, "max open positions", trades, "trades")

    this_row_count += 1  # Increment the row count

    # score = calculate_score(overall_profit_24, open_positions, total_hours_24, total_hours_72, max_hours * len(pair_data.keys()))

    # Save the score to df_params
    df_params.loc[
        (df_params['profit_margin'] == settings['profit_margin']) &
        (df_params['min_percent_change_24'] == settings['min_percent_change_24']) &
        (df_params['min_percent_change_1'] == settings['min_percent_change_1']) &
        (df_params['max_percent_change_24'] == settings['max_percent_change_24']) &
        (df_params['max_percent_change_1'] == settings['max_percent_change_1']),
        ['overall_profit_24', 'overall_profit_72', 'total_hours_24', 'total_hours_72', 'over_24_hours', 'over_72_hours', 'open_positions', 'max_open_positions', 'trades', 'score']
    ] = [overall_profit_24, overall_profit_72, total_hours_24, total_hours_72, over_24_hours, over_72_hours, open_positions, max_open_positions, trades, 0]

    # Sort df_params by the score
    # df_params = df_params.sort_values(by='score', ascending=False)

    # Find the row with the highest overall profit
    # best_row = df_params.iloc[0]

    if not rerun:
        df_params.to_pickle(file_name + ".pkl")

    # print(f"The best profit margin is {best_row['profit_margin']} and the best max_percent_change_24 is {best_row['max_percent_change_24']}  and the best max_percent_change_1 is {best_row['max_percent_change_1']} and the best min_percent_change_24 is {best_row['min_percent_change_24']}  and the best min_percent_change_1 is {best_row['min_percent_change_1']}  with an overall profit of {best_row['overall_profit_24']}")
    # print(best_row)

    # return overall_profit_24, overall_profit_72, total_hours_24, total_hours_72, over_24_hours, over_72_hours, open_positions, max_open_positions

def calculate_score(overall_profit_24, open_positions, total_hours_24, total_hours_72, max_total_hours):
    # Normalize total_hours_24
    normalized_total_hours_24 = (max_total_hours / total_hours_24 if total_hours_24 > 0 else 1) / 100

    # Normalize total_hours_72
    normalized_total_hours_72 = (max_total_hours / total_hours_72 if total_hours_72 > 0 else 1) / 100

    # Calculate the score
    score = (weight_overall_profit_24 * overall_profit_24 if open_positions == 0 else weight_overall_profit_24 * overall_profit_24 - (weight_open_positions * open_positions))

    # Subtract the penalty for high total_hours_24 values
    score += weight_total_hours_24 * normalized_total_hours_24

    # Subtract the penalty for high total_hours_72 values
    score += weight_total_hours_72 * normalized_total_hours_72

    return round(score*10)

def rescore_only(file_name, train_data_length = 0.8):
    # Load df_params from the pickle file
    df_params = pd.read_pickle(file_name + '.pkl')

    # Get the maximum number of hours in the data
    with open('historical_data.pkl', 'rb') as f:
        data = pickle.load(f)

    first_df = next(iter(data.values()))
    max_hours = int(len(first_df) * train_data_length) // 12

    # Subtract 7 days from max_hours
    max_hours -= int(max_hours * sell_period)

    max_hours *= len(data.keys())

    pd.set_option('display.max_columns', None)

    # Iterate over the rows of df_params
    for index, row in df_params.iterrows():
        # Recalculate the score
        if row['overall_profit_24'] <= 0:
            new_score = -1
        else:
            new_score = calculate_score(row['overall_profit_24'], row['open_positions'], row['total_hours_24'], row['total_hours_72'], max_hours)

        # Update the score in df_params
        df_params.loc[index, 'score'] = new_score

    df_params['score'] = df_params['score'].astype(int)

    # Sort df_params by the score
    df_params = df_params.sort_values(by='score', ascending=False)

    print(df_params)

    # Save df_params back to the pickle file
    df_params.to_pickle(file_name + '.pkl')

def show_only(index, file_name):
    # Load df_params from the pickle file
    df_params = pd.read_pickle(file_name + '.pkl')

    print(df_params.iloc[index])
    exit()

def filter_candle_and_pair_data(t_data, min_change_24, min_change_1, max_change_24, max_change_1):
    
    candle_data = {}
    pair_data = {}
    for pair in t_data:
        if t_data[pair].empty:
            continue

        candle_data[pair] = t_data[pair].copy()
        pair_data[pair] = t_data[pair].copy()
        
        candle_data[pair]["close"] = pd.to_numeric(candle_data[pair]["close"])
        pair_data[pair]["close"] = pd.to_numeric(pair_data[pair]["close"])
        pair_data[pair]["priceChangePercent"] = (pair_data[pair]["close"].shift(12).pct_change(periods=288) * 100).round(2)
        pair_data[pair]["priceChangePercent1h"] = (pair_data[pair]["close"].pct_change(periods=12).round(3) * 100).round(2)

        pair_data[pair] = pair_data[pair].dropna(subset=['priceChangePercent', 'priceChangePercent1h'])

        # Filter the data
        pair_data[pair] = pair_data[pair][(pair_data[pair]['priceChangePercent'] >= min_change_24) & (pair_data[pair]['priceChangePercent1h'] >= min_change_1)]
        pair_data[pair] = pair_data[pair][(pair_data[pair]['priceChangePercent'] <= max_change_24) & (pair_data[pair]['priceChangePercent1h'] <= max_change_1)]
        
        # Filter rows for 5 minutes after each hour
        mask = (pair_data[pair].index.minute == 5)
        pair_data[pair] = pair_data[pair][mask]

        pair_data[pair] = pair_data[pair][['close', 'priceChangePercent', 'priceChangePercent1h']]

    # drop all rows with NaN values and remove pairs that never meet the required minimums
    pairs_to_delete = []
    for pair in t_data:
        if pair not in candle_data:
            continue

        candle_data[pair].dropna(inplace=True)
        pair_data[pair].dropna(inplace=True)
        
        if pair_data[pair].empty:
            pairs_to_delete.append(pair)

    for pair in pairs_to_delete:
        del candle_data[pair]
        del pair_data[pair]

    return candle_data, pair_data

def create_or_get_params_file(file_name, profit_margins, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1, rerun = False):
    
    # Check if df_params.pkl exists
    if os.path.exists(file_name + '.pkl') and not rerun:
        # If it exists, load it
        df_params = pd.read_pickle(file_name + '.pkl')
    else:
        # If it doesn't exist, create df_params
        df_params = pd.DataFrame([(m, m24, m1, ma24, ma1) 
                                for m in profit_margins 
                                for m24 in min_percent_change_24 
                                for m1 in min_percent_change_1 
                                for ma24 in max_percent_change_24 
                                for ma1 in max_percent_change_1 
                                if m24 <= ma24 and m1 <= ma1 and m24 / ma24 < 0.75 and m1 / ma1 < 0.75], 
                                columns=['profit_margin', 'min_percent_change_24', 'min_percent_change_1', 'max_percent_change_24', 'max_percent_change_1'])
        
        # Initialize the result columns with np.nan
        df_params['overall_profit_24'] = np.nan
        df_params['overall_profit_72'] = np.nan
        df_params['total_hours_24'] = np.nan
        df_params['total_hours_72'] = np.nan
        df_params['over_24_hours'] = np.nan
        df_params['over_72_hours'] = np.nan
        df_params['open_positions'] = np.nan
        df_params['max_open_positions'] = np.nan
        df_params['trades'] = np.nan
        df_params['score'] = np.nan
    
    # print how many rows are left to calculate (all with NaN values)
    sum_rows = df_params['overall_profit_24'].isna().sum()
    print("Calculating", sum_rows, "rows")

    return df_params, sum_rows

def train(train_data, file_name, rerun = False, args = {}):

    if rerun:
        profit_margins = [args.profit_margin]
        min_percent_change_24 = [args.min_percent_change_24]
        min_percent_change_1 = [args.min_percent_change_1]
        max_percent_change_24 = [args.max_percent_change_24]
        max_percent_change_1 = [args.max_percent_change_1]
    else:
        profit_margins = [1.01, 1.02, 1.03, 1.05, 1.15]
        min_percent_change_24 = [5, 10, 20, 25, 30]
        min_percent_change_1 = [0.3, 0.75, 1.5, 5, 20]
        max_percent_change_24 = [10, 20, 40, 80]
        max_percent_change_1 = [1, 2, 10, 40]

    # Get the minimum values
    min_change_24 = min(min_percent_change_24)
    min_change_1 = min(min_percent_change_1)
    max_change_24 = max(max_percent_change_24)
    max_change_1 = max(max_percent_change_1)

    # Filter the candle and pair data
    print("Calculating 24h change for each pair")
    candle_data, pair_data = filter_candle_and_pair_data(train_data, min_change_24, min_change_1, max_change_24, max_change_1)

    print("Running hyperopt")

    df_params, sum_rows = create_or_get_params_file(file_name, profit_margins, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1, rerun)

    # Apply the hyperopt function to df_params
    df_params.apply(calculation, args=(file_name, sum_rows, candle_data, pair_data, df_params, rerun), axis=1, result_type='expand')

def test(test_data, train_file_name, file_name):

    # Check if df_params.pkl exists
    if os.path.exists(train_file_name + '.pkl'):
        # If it exists, load it
        df_params = pd.read_pickle(train_file_name + '.pkl')
    else:
        print(train_file_name + '.pkl does not exist, run the train first!')
        exit()

    # first remove duplicates on score and overall_profit_24
    df_params = df_params.drop_duplicates(subset=['score', 'overall_profit_24'])

    # Reset the index of df_params
    # df_params = df_params.reset_index(drop=True)

    # Round 'open_positions' and 'overall_profit_24' to 2 decimal places
    df_params['open_positions_rounded'] = df_params['open_positions'].round(2)
    df_params['overall_profit_24_rounded'] = df_params['overall_profit_24'].round(2)

    # Filter rows with positive 'overall_profit_24_rounded'
    df_params = df_params[df_params['overall_profit_24_rounded'] > 0]

    # Group by 'open_positions_rounded' and 'overall_profit_24_rounded', and keep only the row with the highest score in each group
    df_params = df_params.loc[df_params.groupby(['open_positions_rounded', 'overall_profit_24_rounded'])['score'].idxmax()]

    df_params = df_params.nlargest(20, 'score')
    df_params = df_params.sort_values(by='profit_margin', ascending=False)

    # Get the index of the best row
    best_row_index = df_params.index[0]

    df_params = df_params.iloc[0:1]

    profit_margins = [df_params.iat[0, df_params.columns.get_loc('profit_margin')]]
    min_percent_change_24 = [df_params.iat[0, df_params.columns.get_loc('min_percent_change_24')]]
    min_percent_change_1 = [df_params.iat[0, df_params.columns.get_loc('min_percent_change_1')]]
    max_percent_change_24 = [df_params.iat[0, df_params.columns.get_loc('max_percent_change_24')]]
    max_percent_change_1 = [df_params.iat[0, df_params.columns.get_loc('max_percent_change_1')]]
    
    print("Best row index is:", best_row_index, " - Testing with profit margin", profit_margins, "and min_percent_change_24", min_percent_change_24, "and min_percent_change_1", min_percent_change_1, "and max_percent_change_24", max_percent_change_24, "and max_percent_change_1", max_percent_change_1)

    # Filter the candle and pair data
    candle_data, pair_data = filter_candle_and_pair_data(test_data, min_percent_change_24[0], min_percent_change_1[0], max_percent_change_24[0], max_percent_change_1[0])

    if not candle_data or not pair_data:
        print("With these settings, no trades would be made in the test data")
        return

    test_params, sum_rows = create_or_get_params_file(file_name, profit_margins, min_percent_change_24, min_percent_change_1, max_percent_change_24, max_percent_change_1, False)

    test_params['score'] = df_params.iat[0, df_params.columns.get_loc('score')]

    # Now run this row on the test data and output the results
    test_params.apply(calculation, args=(file_name, sum_rows, candle_data, pair_data, test_params, False), axis=1, result_type='expand')

    best_row = test_params.iloc[0]

    print(f"The profit is {best_row['overall_profit_24']} on the test data")
    print(best_row)

async def main(rerun, rescore, show, args):

    if rescore:
        rescore_only(args, args.based_on)
        exit()
    elif show > -1:
        show_only(show, args.based_on)
        exit()

    # Initialize the Binance client
    client = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET"))

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

    if args.start_candle != "" and args.end_candle != "":
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
    for pair in data:
        train_size = int(len(data[pair]) * train_test_split)
        train_data[pair] = data[pair].iloc[:train_size]
        test_data[pair] = data[pair].iloc[train_size:]

    print("train_data", len(train_data), "from", list(train_data.values())[0].iloc[0]["close_time"], "to", list(train_data.values())[0].iloc[-1]["close_time"])
    print("test_data", len(test_data), "from", list(test_data.values())[0].iloc[0]["close_time"], "to", list(test_data.values())[0].iloc[-1]["close_time"])

    filename_friendly_from = list(data.values())[0].iloc[0]["close_time"].strftime("%Y-%m-%d_%H%M%S%f")
    filename_friendly_to = list(data.values())[0].iloc[-1]["close_time"].strftime("%Y-%m-%d_%H%M%S%f")
    file_name = "df_params_" + filename_friendly_from + "_" + filename_friendly_to

    print("working and saving on file:", file_name)

    # hyperopt on the train data
    print("Hyperopting on the train data")
    train(train_data, file_name + '_train', rerun, args)

    # we now need to score each result
    print("Score the results")
    rescore_only(file_name + '_train', train_test_split)

    # plot the results
    print("Plot the results")
    df_params = pd.read_pickle(file_name + "_train.pkl")
    create_plots(df_params, filename_friendly_from + "_" + filename_friendly_to)

    # score on the test data
    print("Test our best row on the test data")
    test(test_data, file_name + '_train', file_name + '_test')

    # Close the client session
    await client.close_connection()

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='hyperopting script')

    # Add the arguments
    parser.add_argument('--profit_margin', type=float, default=1.0, required=False)
    parser.add_argument('--min_percent_change_24', type=float, default=1.0, required=False)
    parser.add_argument('--min_percent_change_1', type=float, default=1.0, required=False)
    parser.add_argument('--max_percent_change_24', type=float, default=1.0, required=False)
    parser.add_argument('--max_percent_change_1', type=float, default=1.0, required=False)
    parser.add_argument('--rerun', type=bool, default=False, required=False, help='rerun the hyperopting with specific parameters to output debug data')
    parser.add_argument('--rescore', type=bool, default=False, required=False)
    parser.add_argument('--show', type=int, default=-1, required=False, help='show an index from the df_params')
    parser.add_argument('--based_on', type=str, default="", required=False, help='for showing, rescoreing and reruning, specify the file_name')
    parser.add_argument('--start_candle', type=str, default="", required=False, help='Start candle in the format "YYYY-MM-DD HH:MM" for the whole data')
    parser.add_argument('--end_candle', type=str, default="", required=False, help='End candle in the format "YYYY-MM-DD HH:MM" for the whole data')

    # Parse the arguments
    args = parser.parse_args()

    # Get the settings and action
    rerun = args.rerun
    rescore = args.rescore
    show = args.show

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

    asyncio.run(main(rerun, rescore, show, args))