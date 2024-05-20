# hyperopting.py
import argparse
from matplotlib.dates import relativedelta
import pandas as pd
import pickle
from datetime import datetime, timedelta
from binance import AsyncClient
import asyncio
import os
from dotenv import load_dotenv
import requests
import numpy as np
from historic_data import is_data_up_to_date, load_month_data, download_historical_data
from plot import create_plots
from trading import get_best_channel_pair
from gridsearch import gridsearch
from cache import start_redis

train_test_split = 0.8

async def main(args):

    # Start the Redis server
    start_redis()

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

    # Parse the start and end candles into datetime objects
    start_candle = datetime.strptime(args.start_candle, "%Y-%m-%d %H:%M") if args.start_candle else datetime(2023, 10, 1)
    end_candle = datetime.strptime(args.end_candle, "%Y-%m-%d %H:%M") if args.end_candle else datetime.now()

    # Convert the start and end candles into timestamps
    start_time = int(start_candle.timestamp() * 1000) if start_candle else None
    end_time = int(end_candle.timestamp() * 1000) if end_candle else None

    # for testing, only use the first 5 pairs
    usdt_pairs = dict(list(usdt_pairs.items())[:1]) # TODO: remove this line

    # Download the historical data
    not_up_to_date_pairs = is_data_up_to_date(usdt_pairs.keys(), start_candle, end_candle, 'historical_channel_data', minimal=True)
    if not_up_to_date_pairs:
        print("Downloading historical data for not up to date pairs", not_up_to_date_pairs)
        await download_historical_data(client, not_up_to_date_pairs, start_candle, end_candle, '5m', 'historical_channel_data', minimal=True)
    else:
        print("All data is up to date")

    # Calculate the total number of days
    total_days = (end_candle - start_candle).days

    # Calculate the number of days for the training period
    train_days = int(total_days * train_test_split)

    # Adjust the end time for the training period
    train_end_candle = start_candle + timedelta(days=train_days)
    train_end_time = int(train_end_candle.timestamp() * 1000)

    # Convert the timestamp back to datetime for printing
    train_start_datetime = datetime.fromtimestamp(start_time / 1000)
    train_end_datetime = datetime.fromtimestamp(train_end_time / 1000)
    test_start_datetime = datetime.fromtimestamp(train_end_time / 1000) + timedelta(minutes=1)
    test_end_datetime = datetime.fromtimestamp(end_time / 1000)

    print(f"start_candle: {start_candle}, end_candle: {end_candle}")
    print(f"total_days: {total_days}, train_days: {train_days}")
    print(f"train_test_split: {train_test_split}")
    print(f"Train data: from {train_start_datetime} to {train_end_datetime}")
    print(f"Test data: from {test_start_datetime} to {test_end_datetime}")

    # hyperopt on the train data
    print("Hyperopting on the train data")
    best_params = gridsearch(usdt_pairs, train_start_datetime, train_end_datetime, 'historical_channel_data')

    # test params on test data
    # print("Testing params on test data")
    # test_result = calculate_strategy(test_data, *best_params)

    # end script with success message
    # print("Test result:", test_result)
    # print("Hyperopting script completed successfully")

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