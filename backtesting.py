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

    for pair in pairs:
        # Get the end time of the last kline in the data
        last_time = data[pair].iloc[-1]['close_time']

        # If the last time is more than one interval ago, download the new klines
        if now - last_time > timedelta(hours=int(interval[:-1])):
            klines = await client.get_historical_klines(pair, interval, int(last_time.timestamp() * 1000))

            # Convert the klines to a DataFrame and append it to the existing data
            df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df.set_index('open_time', inplace=True)
            print(df)
            data[pair] = pd.concat([data[pair], df])

    # Save the updated data to the pickle file
    with open('historical_data.pkl', 'wb') as f:
        pickle.dump(data, f)

    return data

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
        data = await update_historical_data(client, usdt_pairs.keys())
    else:
        data = await download_historical_data(client, usdt_pairs.keys(), start_time)

    # calculate the 24h change for each pair
    for pair in data:
        data[pair]["close"] = pd.to_numeric(data[pair]["close"])
        data[pair]["priceChangePercent"] = data[pair]["close"].pct_change(periods=288)
        data[pair]["priceChangePercent1h"] = data[pair]["close"].pct_change(periods=12)

    print(data)

    # Close the client session
    await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())