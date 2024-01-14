import math
from binance import AsyncClient
from binance.exceptions import BinanceAPIException
import os
from dotenv import load_dotenv
import requests
import asyncio
from itertools import islice
import time

dry_run = False

min_percent_change_24 = 20
min_percent_change_1=0.5
max_percent_change_24=50
max_percent_change_1=2
needed_profit = 1.03
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

    while True:
        # Process the pairs for the 24h window
        usdt_pairs = await process_pairs(client, usdt_pairs, semaphore, get_24h_window, 'priceChangePercent')

        # only during testing
        # usdt_pairs = dict(list(usdt_pairs.items())[:15])

        # Process the pairs for the 1h window
        usdt_pairs = await process_pairs(client, usdt_pairs, semaphore, get_1h_window, 'priceChangePercent1h')

        # get the best pair
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
    asyncio.run(main())
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(main())