import math
import statistics
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
import os
from dotenv import load_dotenv
import requests
import asyncio
from itertools import islice
import time
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from cache import get_redis_server, create_data_hash, create_cache_key
import xxhash

dry_run = True

min_percent_change_24 = -20
min_percent_change_1=-20
max_percent_change_24=500
max_percent_change_1=2000
needed_profit = 1.01
amount_for_order = 10

# Load the .env file
load_dotenv()

# Set a global default timeout for all requests
requests.adapters.DEFAULT_RETRIES = 5

async def process_pairs(client, pairs, semaphore, window_func, percent_key):
    # Split the dictionary of pairs into chunks of 20
    pair_chunks = list(chunks(pairs))

    # Create a list of tasks
    tasks = [window_func(client, chunk, semaphore) for chunk in pair_chunks]

    # Run the tasks concurrently
    results = await asyncio.gather(*tasks)

    # Combine the chunks back into a single dictionary
    pairs = {k: v for chunk in results for k, v in chunk.items()}

    # Remove all pairs not positive percent change
    pairs = {k: v for k, v in pairs.items() if float(v[percent_key]) > 0}

    return pairs

def chunks(data, SIZE=20):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}

async def get_1h_window(client, pairs, semaphore):
    # Acquire the semaphore
    async with semaphore:
        try:
            # Get the klines (candlestick data) for the pair
            symbols = '["' + '","'.join(list(pairs.keys())) + '"]'
            tickers = await client.get_ticker_window(symbols=symbols, windowSize=AsyncClient.KLINE_INTERVAL_1HOUR)

            # Iterate over the tickers and update pairs
            for ticker in tickers:
                symbol = ticker['symbol']

                # Check if the symbol is in pairs
                if symbol in pairs:
                    # Update the dictionary with the ticker data
                    pairs[symbol]["priceChangePercent1h"] = ticker['priceChangePercent']

            return pairs
        except IndexError as e:
            print(pairs[1], e)
        except KeyError as e:
            print(pairs[1], e)
        except BinanceAPIException as e:
            print(pairs[1], e)

async def get_24h_window(client, pairs, semaphore):
    # Acquire the semaphore
    async with semaphore:
        try:
            # Get the klines (candlestick data) for the pair
            symbols = '["' + '","'.join(list(pairs.keys())) + '"]'
            tickers = await client.get_ticker(symbols=symbols)

            # Iterate over the tickers and update pairs
            for ticker in tickers:
                symbol = ticker['symbol']

                # Check if the symbol is in pairs
                if symbol in pairs:
                    # Update the dictionary with the ticker data
                    pairs[symbol].update(ticker)

            return pairs
        except IndexError as e:
            print(pairs[1], e)
        except KeyError as e:
            print(pairs[1], e)
        except BinanceAPIException as e:
            print(pairs[1], e)

def get_best_pair(pair_data):

    # Filter pairs based on min_percent_change_24 and min_percent_change_1
    pairs = {}
    for pair, row in pair_data.items():
        if float(row['priceChangePercent']) >= min_percent_change_24 and float(row['priceChangePercent1h']) >= min_percent_change_1 and float(row['priceChangePercent']) <= max_percent_change_24 and float(row['priceChangePercent1h']) <= max_percent_change_1:
            pairs[pair] = {'priceChangePercent': row['priceChangePercent'], 'priceChangePercent1h': row['priceChangePercent1h']}

    if not pairs:
        return None

    # Find the pair with the highest 1-hour price change
    best_pair = max(pairs, key=lambda pair: pairs[pair]['priceChangePercent1h'])

    # Return the best pair, its 'close' price, and the 'open_time'
    return best_pair, pair_data[best_pair]

def get_candles(pair, candles_data):
    if pair in candles_data:
        if len(candles_data[pair]) >= 1000:
            return candles_data[pair][-1000:]
        else:
            print(f"Warning: Less than 1000 candles available for {pair}")
            return candles_data[pair]
    else:
        return requests.get(f'https://api.binance.com/api/v3/klines?symbol={pair}&interval=5m&limit=1000').json()

def calculate_bollinger_bands(df, rolling_window_number, std_for_BB, moving_average_type):
    if moving_average_type == 'SMA':
        df.loc[:, 'MA'] = df['close'].rolling(window=rolling_window_number).mean()
    elif moving_average_type == 'EMA':
        df.loc[:, 'MA'] = df['close'].ewm(span=rolling_window_number, adjust=False).mean()

    df.loc[:, 'STD'] = df['close'].rolling(window=rolling_window_number).std()
    df.loc[:, 'UpperBB'] = df['MA'] + std_for_BB * df['STD']
    df.loc[:, 'LowerBB'] = df['MA'] - std_for_BB * df['STD']

    return df

def calculate_transition_times(df):
    transition_times = []
    prev_state = None
    for row in df.itertuples():
        close_price = getattr(row, 'close')
        if close_price >= getattr(row, 'UpperBB'):
            if prev_state == 'low':
                transition_times.append(row.Index)
            prev_state = 'high'
        elif close_price <= getattr(row, 'LowerBB'):
            prev_state = 'low'
    return transition_times

# TODO this function is now only working for given candles, not for requesting new ones from binance!
def get_best_channel_pair(all_data, cache_key, pair_list, time, price_jump_threshold = 0.10, last_price_treshold = 0.50, rolling_window_number = 20, std_for_BB = 2, moving_average_type = 'SMA'):
    r = get_redis_server()
    
    # Check if the result is in the cache for the whole dataframe
    result = r.get(cache_key)
    if result is not None:
        result = result.decode('utf-8')
        if result == 'None':
            result = None
        return result
    
    # Convert time to datetime64[ns]
    # time = pd.to_datetime(time)
    # Create a subset of all_data that only includes rows where open_time <= time
    subset_data = all_data[all_data['open_time'] <= time]

    # Count the number of rows for each pair
    counts = subset_data['pair'].value_counts()

    # Get a list of pairs that have at least 1000 rows
    valid_pairs = counts[counts >= 1000].index.tolist()

    # Filter pair_list to only include valid pairs
    pair_list = [pair for pair in pair_list if pair in valid_pairs]

    # Reduce the subset to the last 1000 rows for each pair
    subset_data = subset_data[subset_data['pair'].isin(valid_pairs)].groupby('pair').tail(1000)
    
    # Check if the result is in the cache for this subset
    sub_cache_key = create_cache_key(xxhash.xxh64(str(subset_data)).hexdigest(), pair_list, time, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type)
    result = r.get(sub_cache_key)
    if result is not None:
        result = result.decode('utf-8')
        if result == 'None':
            result = None
        return result

    results = {pair: {'last_price': None, 'low_to_high': 0, 'std_dev': 0, 'mean_time': 0, 'price_to_lower_band': None} for pair in pair_list}
    highest_transitions = 0
    highest_sdt_dev = 0

    for pair in pair_list:

        df = subset_data[subset_data['pair'] == pair].copy()

        last_price = float(df.iloc[-1]['close'])
        price_jump = df['close'].max() - df['close'].min()
        threshold = price_jump_threshold * df['close'].min()

        # print(pair, price_jump, threshold, last_price, last_price_treshold, df['close'].min())

        if price_jump > threshold:
            del results[pair]
            continue

        # Shift the 'close' column by one row
        df.loc[:, 'prev_close'] = df['close'].shift(1)

        # Calculate the price jump for every two consecutive candles
        df.loc[:, 'price_jump'] = abs(df['open'] - df['prev_close'])

        # Check if any price jump is too big
        if any(df['price_jump'] > 0.25 * df['prev_close']):
            print("price_jump too big", pair, df['price_jump'].max())
            del results[pair]
            continue

        df = calculate_bollinger_bands(df, rolling_window_number, std_for_BB, moving_average_type)

        # Normalize the last price and the Bollinger Bands
        last_price_normalized = (last_price - df['LowerBB'].min()) / (df['UpperBB'].max() - df['LowerBB'].min())

        if last_price_normalized <= last_price_treshold:
            del results[pair]
            continue

        timeframe_days = (df['open_time'].max() - df['open_time'].min()).days
        transition_times = calculate_transition_times(df)

        # Filter the DataFrame to include only the rows that correspond to the transition times
        transition_df = df.loc[transition_times]

        # Convert the 'open_time' of these rows to ordinal values
        transition_days = [trade_date.toordinal() for trade_date in transition_df['open_time']]

        # Calculate the standard deviation of the trade days
        transition_days_std_dev = statistics.stdev(transition_days) if len(transition_days) > 1 else 0

        # Normalize the standard deviation by dividing it by the total number of days in the timeframe
        normalized_std_dev = transition_days_std_dev / timeframe_days

        # Normalize the thresholds based on the timeframe
        adjusted_low_to_high_threshold = timeframe_days * 0.25
        adjusted_std_dev_threshold = 0.05

        mean_time = statistics.mean(transition_times) if transition_times else 0

        # Store the standard deviation of the trade days in the results
        results[pair]['normalized_std_dev'] = normalized_std_dev

        results[pair]['last_price'] = last_price
        results[pair]['low_to_high'] = len(transition_times)
        results[pair]['mean_time'] = mean_time
        results[pair]['price_to_lower_band'] = abs(last_price - df['LowerBB'].iloc[-1])

        if highest_transitions < len(transition_times):
            highest_transitions = len(transition_times)
        if highest_sdt_dev < normalized_std_dev:
            highest_sdt_dev = normalized_std_dev

    sorted_pairs = sorted(results.items(), key=lambda x: (-x[1]['low_to_high'], x[1]['normalized_std_dev'], x[1]['price_to_lower_band'], -x[1]['mean_time']))

    if not sorted_pairs or sorted_pairs[0][1]['low_to_high'] < adjusted_low_to_high_threshold or sorted_pairs[0][1]['normalized_std_dev'] < adjusted_std_dev_threshold:
        result = 'None'
    else:
        result = sorted_pairs[0][0]
    
    r.set(sub_cache_key, result)
    r.set(cache_key, result)

    if result == 'None':
        result = None
    return result

async def place_order(client, pair, amount):
    if dry_run:
        return None

    # Place a market order
    order = await client.create_order(
        symbol=pair,
        side=client.SIDE_BUY,
        type=client.ORDER_TYPE_MARKET,
        quoteOrderQty=amount
    )

    return order

async def query_order(client, pair, order_id):
    # Query the order
    order = await client.get_order(
        symbol=pair,
        orderId=order_id
    )

    return order

async def wait_for_order_fill(client, symbol, order_id):
    while True:
        order_result = await query_order(client, symbol, order_id)
        if order_result['status'] == 'FILLED':
            break
        time.sleep(5)  # wait for 5 seconds before the next query
    return order_result

async def get_order_trades(client, symbol, order_id):
    trades = await client.get_my_trades(symbol=symbol, orderId=order_id)
    
    total_price = 0.0
    total_qty = 0.0
    total_commission = 0.0
    
    for trade in trades:
        total_price += float(trade['price']) * float(trade['qty'])
        total_qty += float(trade['qty'])
        if trade['commissionAsset'] == 'BNB':
            total_commission += float(trade['commission'])
    
    # Query the BNBUSDT price
    bnb_price = await client.get_symbol_ticker(symbol='BNBUSDT')
    total_commission_in_usdt = total_commission * float(bnb_price['price'])
    avg_price = (total_price + total_commission_in_usdt) / total_qty

    return avg_price, total_qty

async def place_sell_order(client, pair, symbol, quantity, avg_price):
    if dry_run:
        return None
    
    # Get the LOT_SIZE filter from the symbol info
    lot_size_filter = [filter for filter in pair['filters'] if filter['filterType'] == 'LOT_SIZE'][0]

    # Get the minQty, maxQty, and stepSize from the LOT_SIZE filter
    min_qty = float(lot_size_filter['minQty'])
    max_qty = float(lot_size_filter['maxQty'])
    step_size = float(lot_size_filter['stepSize'])

    # Ensure the quantity is within the minQty and maxQty
    quantity = max(min_qty, min(max_qty, quantity))

    # Calculate the number of decimal places for the stepSize
    step_size_decimals = int(round(-math.log(step_size, 10)))

    # Round the quantity to the correct number of decimal places
    quantity = round(quantity, step_size_decimals)
    
    # Calculate the sell price
    sell_price = avg_price * needed_profit

    # Get the baseAssetPrecision from the symbol info
    base_asset_precision = pair['baseAssetPrecision']
    
    # Get the PRICE_FILTER filter from the symbol info
    price_filter = [filter for filter in pair['filters'] if filter['filterType'] == 'PRICE_FILTER']
    if price_filter:

        # Get the minPrice, maxPrice, and tickSize from the PRICE_FILTER filter
        min_price = float(price_filter[0]['minPrice'])
        max_price = float(price_filter[0]['maxPrice'])
        tick_size = float(price_filter[0]['tickSize'])

        # Ensure the price is within the minPrice and maxPrice
        sell_price = max(min_price, min(max_price, sell_price))

        # Calculate the number of decimal places for the tickSize
        tick_size_decimals = int(round(-math.log(tick_size, 10)))

        # Round the price to the correct number of decimal places
        sell_price = round(sell_price, tick_size_decimals)

    # Check for PERCENT_PRICE filter
    percent_price_filters = [filter for filter in pair['filters'] if filter['filterType'] == 'PERCENT_PRICE']
    if percent_price_filters:
        percent_price_filter = float(percent_price_filters[0]['multiplierUp'])
        sell_price = min(sell_price, avg_price * percent_price_filter)
    
    # Check for PERCENT_PRICE_BY_SIDE filter
    percent_price_by_side_filters = [filter for filter in pair['filters'] if filter['filterType'] == 'PERCENT_PRICE_BY_SIDE']
    if percent_price_by_side_filters:
        ask_multiplier_up = float(percent_price_by_side_filters[0]['askMultiplierUp'])
        sell_price = min(sell_price, avg_price * ask_multiplier_up)

    order = await client.create_order(
        symbol=symbol,
        side='SELL',
        type='LIMIT',
        timeInForce='GTC',
        quantity=quantity,
        price='{:0.{}f}'.format(sell_price, base_asset_precision)  # format the price as a string with 8 decimal places
    )
    return order

async def main(channel_trading):

    # Initialize the Binance client
    client = await AsyncClient.create(os.getenv("BINANCE_API_KEY"), os.getenv("BINANCE_SECRET"))

    # Create a semaphore with a limit of 10 concurrent requests
    semaphore = asyncio.Semaphore(10)

    # first we get all USDT pairs from binance
    all_pairs = await client.get_exchange_info()

    usdt_pairs = {}
    for pair in all_pairs["symbols"]:
        if pair['status'] == 'TRADING' and pair['quoteAsset'] == 'USDT' and any('SPOT' in subarray for subarray in pair['permissionSets']):
            # remove unneeded stuff from pair
            del pair['permissions']
            del pair['permissionSets']
            del pair['allowedSelfTradePreventionModes']
            del pair['defaultSelfTradePreventionMode']
            usdt_pairs[pair['symbol']] = pair
        # else:
        #    print(pair['status'], pair['quoteAsset'], pair['permissionSets'])

    while True:

        # get the best pair
        best_pair = None
        if channel_trading:
            best_pair = get_best_channel_pair(usdt_pairs)
        else:
            # Process the pairs for the 24h window, checks if positive price in the last 24 hours
            usdt_pairs = await process_pairs(client, usdt_pairs, semaphore, get_24h_window, 'priceChangePercent')

            # only during testing
            # usdt_pairs = dict(list(usdt_pairs.items())[:15])

            # Process the pairs for the 1h window, checks if positive price in the last 1 hour
            usdt_pairs = await process_pairs(client, usdt_pairs, semaphore, get_1h_window, 'priceChangePercent1h')

            best_pair = get_best_pair(usdt_pairs)

        if best_pair:
            symbol = best_pair[0]
            pair = best_pair[1]
        else:
            print("No best pair found, waiting 60min to try again")
            time.sleep(60 * 60)  # Wait for 60 minutes
            continue

        # Print the results
        '''('FTTUSDT', {'symbol': 'FTTUSDT', 'status': 'TRADING', 'baseAsset': 'FTT', 'baseAssetPrecision': 8, 'quoteAsset': 'USDT', 'quotePrecision': 8, 'quoteAssetPrecision': 8, 'baseCommissionPrecision': 8, 'quoteCommissionPrecision': 8, 'orderTypes': ['LIMIT', 'LIMIT_MAKER', 'MARKET', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT_LIMIT'], 'icebergAllowed': True, 'ocoAllowed': True, 'quoteOrderQtyMarketAllowed': True, 'allowTrailingStop': True, 'cancelReplaceAllowed': True, 'isSpotTradingAllowed': True, 'isMarginTradingAllowed': True, 'priceChange': '0.43180000', 'priceChangePercent': '13.780', 'weightedAvgPrice': '3.37051857', 'prevClosePrice': '3.13270000', 'lastPrice': '3.56530000', 'lastQty': '0.93000000', 'bidPrice': '3.56530000', 'bidQty': '55.16000000', 'askPrice': '3.56630000', 'askQty': '5.39000000', 'openPrice': '3.13350000', 'highPrice': '3.61700000', 'lowPrice': '2.92660000', 'volume': '15472416.83000000', 'quoteVolume': '52150068.18244700', 'openTime': 1703852634537, 'closeTime': 1703939034537, 'firstId': 67466636, 'lastId': 67835478, 'count': 368843, 'priceChangePercent1h': '2.475'})'''
        print(symbol, pair)

        # Place an order for the best pair
        order = await place_order(client, symbol, amount_for_order)
        print(order)

        # query the order result
        buy_order_result = await wait_for_order_fill(client, symbol, order['orderId'])
        print(buy_order_result)

        avg_buy_price, total_qty = await get_order_trades(client, symbol, order['orderId'])
        total_buy_cost = avg_buy_price * total_qty
        print(avg_buy_price, total_qty, total_buy_cost)
        
        sell_order = await place_sell_order(client, pair, symbol, total_qty, avg_buy_price)
        print(sell_order)

        # Wait for the order to be filled
        sell_order_result = await wait_for_order_fill(client, symbol, sell_order['orderId'])
        print(sell_order_result)

        # Get the details of the sell order
        avg_sell_price, _ = await get_order_trades(client, symbol, sell_order['orderId'])
        total_sell_proceeds = avg_sell_price * total_qty
        print(avg_sell_price, total_qty, total_sell_proceeds)

        # Calculate the profit
        profit = total_sell_proceeds - total_buy_cost
        print("Profit:", profit)
    
    # Close the client session
    await client.close_connection()

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='hyperopting script')

    # Add the arguments
    parser.add_argument('--channel_trading', type=bool, default=False, required=False, help='wit this argument, the trading will be done on price channels instead of given parameters')

    # Parse the arguments
    args = parser.parse_args()

    asyncio.run(main(args.channel_trading))
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())