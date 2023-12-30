import requests
import json
import os
import websocket
import sqlite3
import traceback
import pandas as pd
import prediction
from sklearn.preprocessing import MinMaxScaler
import numpy as np

last_candle = {
    'k': {
        't': 0,
        'o': 0,
        'h': 0,
        'l': 0,
        'c': 0,
        'v': 0
    }
}

def get_candles():
    global last_candle

    # Connect to Binance API
    symbol = 'BNBUSDT'
    interval = '1m'
    limit = 10
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)

    # Parse response
    candles = json.loads(response.text)

    # Unpack the tuple key into individual keys in the last_candle['k'] dictionary
    tc = candles.pop()
    last_candle['k'] = dict(zip(['t', 'o', 'h', 'l', 'c', 'v'], tc))

    # Save candles to SQLite database
    for candle in candles:
        open_time = candle[0]
        open = candle[1]
        high = candle[2]
        low = candle[3]
        close = candle[4]
        volume = candle[5]
        c.execute('INSERT OR IGNORE INTO candles (open_time, open, high, low, close, volume) VALUES (?, ?, ?, ?, ?, ?)', (open_time, open, high, low, close, volume))

    conn.commit()

    # Return candles
    return candles

# Connect to database
conn = sqlite3.connect('data/' + 'candles.db')
c = conn.cursor()

# Create table if it doesn't exist
c.execute('CREATE TABLE IF NOT EXISTS candles (open_time INTEGER PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)')

# Download last 10 candles
candles = get_candles()

# Do something with candles
print('import of starting candles done')

# Read candles from database
df = pd.read_sql_query("SELECT * FROM candles ORDER BY open_time ASC", conn)

print("DF from DB")
print(df.tail())

# Define models
models = []
for filename in os.listdir('models'):
    if filename.endswith('.py') and filename != 'base_model.py':
        model_name = filename[:-3]
        print('Importing model:', model_name)
        module = __import__('models.' + model_name, fromlist=[model_name])
        model_class = getattr(module, model_name.capitalize())
        model = model_class((6, 1), len(df), 'data/') # (devices, measeurements)
        models.append(model)

def on_message(ws, message):
    global df
    global last_candle

    # Parse message
    candle = json.loads(message)

    if "result" not in candle or candle["result"] is not None:

        # Do something with candle
        # print('got a candle', candle['k']['t'], [candle['k']['o'], candle['k']['h'], candle['k']['l'], candle['k']['c'], candle['k']['v']])

        # Check if candle already exists in database
        c.execute("SELECT * FROM candles WHERE open_time=?", (candle['k']['t'],))
        existing_candle = c.fetchone()

        if existing_candle:
            # print("we already know it. ignore...")
            pass
        elif not last_candle or last_candle['k']['t'] == candle['k']['t']:
            # print("temp save to last and ignore...")
            last_candle = candle
        elif last_candle and last_candle['k']['t'] != candle['k']['t']:
            # print("it's a new candle")

            # we do not add the new candle to DF yet, but we are going to add the last one now
            # because this new candle is not yet closed, it's the in progress candle
            # print('saving last candle to DB', last_candle['k']['t'], [last_candle['k']['o'], last_candle['k']['h'], last_candle['k']['l'], last_candle['k']['c'], last_candle['k']['v']])

            # Insert new candle into database
            c.execute("INSERT INTO candles VALUES (?, ?, ?, ?, ?, ?)",
                (last_candle['k']['t'], last_candle['k']['o'], last_candle['k']['h'], last_candle['k']['l'], last_candle['k']['c'], last_candle['k']['v']))
            conn.commit()

            # now we know the candle
            last_candle = candle

            # Create a DataFrame with the last new candle data
            new_candle_df = pd.read_sql_query("SELECT * FROM candles ORDER BY open_time DESC LIMIT 1", conn)
            # print(new_candle_df.tail())

            # Append new candle to DataFrame
            df = pd.concat([df, new_candle_df], ignore_index=True)
            
            # print('added the new candle')
            # print(df.tail())

            # Trigger prediction on new candles
            prediction.make_prediction(models, df)

def on_error(ws, error):
    tb = traceback.format_exc()
    print(f"Error in {__file__}: {tb}")
    print(error)

def on_close(ws, close_status_code, close_msg):
    print("Connection closed")

def on_open(ws):
    print("opened ws")
    # Subscribe to BNB/USDT 1-minute candles
    ws.send('{"method": "SUBSCRIBE", "params": ["bnbusdt@kline_1m"], "id": 1}')

if __name__ == "__main__":
    # Connect to Binance WebSocket API
    ws = websocket.WebSocketApp("wss://stream.binance.com:9443/ws",
                                on_message = on_message,
                                on_error = on_error,
                                on_close = on_close,
                                on_open = on_open
                                )
    ws.run_forever()