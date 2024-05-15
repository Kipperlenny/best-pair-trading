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

def get_best_channel_pair(pair_data, price_jump_threshold = 0.10, last_price_treshold = 0.50, rolling_window_number = 20, std_for_BB = 2, moving_average_type = 'SMA', low_to_high_threshold = 10, std_dev_threshold = 20, candles_data = None):

    candles = {}
    results = {pair: {'last_price': None, 'low_to_high': 0, 'std_dev': None, 'mean_time': None, 'price_to_lower_band': None} for pair in list(pair_data.keys())}
    transition_times = {}
    finished = False
    highest_transitions = 0
    highest_sdt_dev = 0
    for pair in list(pair_data.keys()):
        if finished:
            # delete from pair_data
            del pair_data[pair]
            continue

        # get the last 7 days of data
        if not pair in candles_data:
            # get the 1000 candles from candles_data
            if len(candles_data[pair]) >= 1000:
                candles[pair] = candles_data[pair][-1000:]  # get the last 1000 candles
            else:
                print(f"Warning: Less than 1000 candles available for {pair}")
                candles[pair] = candles_data[pair]
        else:
            candles[pair] = requests.get(f'https://api.binance.com/api/v3/klines?symbol={pair}&interval=5m&limit=1000').json()

        # get close price from last candle:
        last_price = float(candles[pair][-1][4])

        # Convert the candles to a DataFrame for easier calculations
        df = pd.DataFrame(candles[pair], columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Close'] = df['Close'].astype(float)

        # Calculate the price jump
        price_jump = df['Close'].max() - df['Close'].min()

        # Calculate the threshold as 10% of the minimum price
        threshold = price_jump_threshold * df['Close'].min()

        # Skip this pair if the price jump is too big
        if price_jump > threshold:
            # delete from pair_data
            del pair_data[pair]
            # print(f"Price jump too big for {pair}, skipping...", df['Close'].max(), df['Close'].min())
            continue

        # Calculate the moving average and standard deviation of the closing prices
        if moving_average_type == 'SMA':
            df['MA'] = df['Close'].rolling(window=rolling_window_number).mean()
        elif moving_average_type == 'EMA':
            df['MA'] = df['Close'].ewm(span=rolling_window_number, adjust=False).mean()

        df['STD'] = df['Close'].rolling(window=rolling_window_number).std()

        # Calculate the upper and lower Bollinger Bands
        df['UpperBB'] = df['MA'] + std_for_BB * df['STD']
        df['LowerBB'] = df['MA'] - std_for_BB * df['STD']

        # If the last price is near the upper Bollinger Band, delete the pair from pair_data
        if abs(last_price - df['UpperBB'].iloc[-1]) < last_price_treshold:
            del pair_data[pair]
            # print(f"Last price near upper Bollinger Band for {pair}, skipping...")
            continue

        # Initialize a list to store the times of each transition
        transition_times[pair] = []

        prev_state = None
        for i, row in df.iterrows():
            close_price = row['Close']
            if close_price >= row['UpperBB']:
                if prev_state == 'low':
                    # Record the time of this transition
                    transition_times[pair].append(i)
                prev_state = 'high'
            elif close_price <= row['LowerBB']:
                prev_state = 'low'

        # Now, 'transition_times' contains the times of each low-to-high transition

        # Calculate the standard deviation of the transition times
        mean_time = statistics.mean(transition_times[pair]) if transition_times[pair] else 0
        std_dev = statistics.stdev(transition_times[pair]) if len(transition_times[pair]) > 1 else 0

        # Save data to row
        results[pair]['last_price'] = last_price
        results[pair]['low_to_high'] = len(transition_times[pair])
        results[pair]['std_dev'] = std_dev
        results[pair]['mean_time'] = mean_time
        results[pair]['price_to_lower_band'] = abs(last_price - df['LowerBB'].iloc[-1])

        if highest_transitions < len(transition_times[pair]):
            highest_transitions = len(transition_times[pair])
        if highest_sdt_dev < std_dev:
            highest_sdt_dev = std_dev

        # print(pair, len(transition_times[pair]), std_dev)

        # if len(transition_times[pair]) > 5:
            # to avoid errors, we have to remove all remaining pairs from pair_data
            # finished = True

    # Calculate mean transition time and add it to results
    for pair in results:
        if pair in transition_times:
            results[pair]['mean_time'] = statistics.mean(transition_times[pair]) if transition_times[pair] else 0
        else:
            results[pair]['mean_time'] = 0

    # Sort the pairs based on the number of low-to-high transitions, the standard deviation, and the mean transition time
    sorted_pairs = sorted(results.items(), key=lambda x: (-x[1]['low_to_high'], x[1]['std_dev'], x[1]['price_to_lower_band'], -x[1]['mean_time']))

    # Check if sorted_pairs is empty
    if not sorted_pairs:
        # print("No pairs left after filtering.", highest_transitions, highest_sdt_dev, len(pair_data))
        return None

    # Check if best pair is above a needed transition threshold, std_dev threshold, and mean time threshold
    if sorted_pairs[0][1]['low_to_high'] < low_to_high_threshold or sorted_pairs[0][1]['std_dev'] < std_dev_threshold:
        # print("No pairs left after filtering for threshold.", highest_transitions, highest_sdt_dev, len(pair_data))
        return None

    # Get the pair with the highest number of low-to-high transitions and the lowest standard deviation, return the pair symbol only
    return sorted_pairs[0][0]

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