
import gzip
import pickle
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from tqdm import tqdm

def load_month_data(pair, month, directory='historical_data', minimal = False):
    if minimal and os.path.exists(os.path.join(directory, f'{pair}_{month.strftime("%Y_%m")}_minimal.pkl.gz')):
        file = os.path.join(directory, f'{pair}_{month.strftime("%Y_%m")}_minimal.pkl.gz')
    else:
        file = os.path.join(directory, f'{pair}_{month.strftime("%Y_%m")}.pkl.gz')
    if os.path.exists(file):
        with gzip.open(file, 'rb') as f:
            p = pickle.load(f)

            if minimal:
                if 'quote_asset_volume' in p: del p['quote_asset_volume']
                if 'number_of_trades' in p: del p['number_of_trades']
                if 'taker_buy_base_asset_volume' in p: del p['taker_buy_base_asset_volume']
                if 'taker_buy_quote_asset_volume' in p: del p['taker_buy_quote_asset_volume']
                if 'ignore' in p: del p['ignore']
                
            return p
    else:
        return None

def is_data_up_to_date(pairs, start_time, end_time, directory='historical_channel_data', minimal = False):
    start_month = start_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    end_month = end_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    not_up_to_date_pairs = []
    while start_month <= end_month:
        for pair in pairs:
            file = os.path.join(directory, f'{pair}_{start_month.strftime("%Y_%m")}.pkl.gz')
            if not os.path.exists(file):
                print(f"File for {pair} in {file} is missing.")
                not_up_to_date_pairs.append(pair)
            else:
                data = load_month_data(pair, start_month, directory, minimal = minimal)
                if data is None:
                    print(f"Data for {pair} in {start_month.strftime('%Y-%m')} is missing.")
                    not_up_to_date_pairs.append(pair)
                if start_month == end_month:
                    if data.iloc[-1]['close_time'] < end_time:
                        not_up_to_date_pairs.append(pair)
                elif data.iloc[-1]['close_time'] < start_month + relativedelta(months=1) - timedelta(minutes=5):
                    print(f"Data for {pair} in {start_month.strftime('%Y-%m')} is not up to date.")
                    not_up_to_date_pairs.append(pair)
        start_month += relativedelta(months=1)
    return list(set(not_up_to_date_pairs))  # Remove duplicates

async def download_historical_data(client, pairs, start_time, end_time, interval='5m', directory='historical_data', retries=3, minimal = False):
    os.makedirs(directory, exist_ok=True)

    for pair in tqdm(pairs):
        current_start_time = start_time
        while current_start_time < end_time:
            current_end_time = current_start_time.replace(day=1) + relativedelta(months=1, minutes=-1)
            if current_end_time > end_time:
                current_end_time = end_time

            for attempt in range(retries):
                try:
                    print("Downloading historical data for", pair, current_start_time.strftime("%Y-%m"))
                    klines = await client.get_historical_klines(symbol=pair, interval=interval, start_str=current_start_time.strftime("%d %b %Y %H:%M:%S"), end_str=current_end_time.strftime("%d %b %Y %H:%M:%S"))

                    # Convert the klines to a DataFrame
                    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

                    # Convert the time columns to datetime
                    df['high'] = df['high'].astype('float32')
                    df['low'] = df['low'].astype('float32')
                    df['number_of_trades'] = df['number_of_trades'].astype(int)
                    df['volume'] = df['volume'].astype('float32')
                    df['quote_asset_volume'] = df['quote_asset_volume'].astype('float32')
                    df['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'].astype('float32')
                    df['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'].astype('float32')
                    df['open'] = df['open'].astype('float32')
                    df['close'] = df['close'].astype('float32')
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                    if minimal:
                        if 'quote_asset_volume' in df: del df['quote_asset_volume']
                        if 'number_of_trades' in df: del df['number_of_trades']
                        if 'taker_buy_base_asset_volume' in df: del df['taker_buy_base_asset_volume']
                        if 'taker_buy_quote_asset_volume' in df: del df['taker_buy_quote_asset_volume']
                        if 'ignore' in df: del df['ignore']

                    # Set the open time as the index
                    df.set_index('open_time', inplace=True)

                    # Save the data to a compressed pickle file
                    file = os.path.join(directory, f'{pair}_{current_start_time.strftime("%Y_%m")}{"_minimal" if minimal else ""}.pkl.gz')
                    with gzip.open(file, 'wb') as f:
                        pickle.dump(df, f)

                    break
                except Exception as e:
                    if attempt < retries - 1:  # i is zero indexed
                        continue
                    else:
                        raise

            current_start_time = current_end_time + relativedelta(minutes=1)

    return directory

async def update_historical_data(client, pairs, interval='5m', directory='historical_data', retries=3):
    # Get the current time
    now = datetime.now()
    something_updated = False

    for pair in pairs:
        # Load the historical data from the pickle file
        file = os.path.join(directory, f'{pair}_{now.strftime("%Y_%m")}.pkl.gz')
        if not os.path.exists(file):
            continue

        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)

        # Get the end time of the last kline in the data
        last_time = data.iloc[-1]['close_time']

        # only update if older than 24 hours
        if now - last_time < timedelta(hours=24):
            continue
        else:
            something_updated = True

        current_start_time = last_time
        while current_start_time < now:
            current_end_time = current_start_time.replace(day=1) + relativedelta(months=1, minutes=-1)
            if current_end_time > now:
                current_end_time = now

            print("Updating historical data for", pair, current_start_time.strftime("%Y-%m"))

            # If the last time is more than one interval ago, download the new klines
            for attempt in range(retries):
                try:
                    print("Downloading new data for", pair, current_start_time.strftime("%Y-%m"))
                    klines = await client.get_historical_klines(symbol=pair, interval=interval, start_str=current_start_time.strftime("%d %b %Y %H:%M:%S"), end_str=current_end_time.strftime("%d %b %Y %H:%M:%S"))

                    # Convert the klines to a DataFrame and append it to the existing data
                    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                    
                    df['high'] = df['high'].astype('float32')
                    df['low'] = df['low'].astype('float32')
                    df['number_of_trades'] = df['number_of_trades'].astype(int)
                    df['volume'] = df['volume'].astype('float32')
                    df['quote_asset_volume'] = df['quote_asset_volume'].astype('float32')
                    df['taker_buy_base_asset_volume'] = df['taker_buy_base_asset_volume'].astype('float32')
                    df['taker_buy_quote_asset_volume'] = df['taker_buy_quote_asset_volume'].astype('float32')
                    df['open'] = df['open'].astype('float32')
                    df['close'] = df['close'].astype('float32')
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    df.set_index('open_time', inplace=True)
                    data = pd.concat([data, df])

                    break
                except Exception as e:
                    if attempt < retries - 1:  # i is zero indexed
                        continue
                    else:
                        raise

            current_start_time = current_end_time + relativedelta(minutes=1)

        # Save the updated data to the pickle file
        if something_updated:
            with gzip.open(file, 'wb') as f:
                pickle.dump(data, f)

    return directory