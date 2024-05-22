import joblib
import xxhash
import pandas as pd
import os
import redis
import subprocess
import time

redis_port = 50001
# memory = joblib.Memory("cache_directory", verbose=0)

def is_redis_running(host='localhost', port=redis_port):
    try:
        r = redis.Redis(host=host, port=port)
        r.ping()
        return True
    except redis.ConnectionError:
        return False
    
def start_redis():
    if not is_redis_running():
        conf_path = os.path.join(os.path.dirname(__file__), 'redis.conf')
        subprocess.Popen(['redis-server', conf_path])
        time.sleep(15)  # Wait for 5 seconds before retrying

def stop_redis():
    subprocess.run(['redis-cli', '-p', str(redis_port), 'shutdown'])

def get_redis_server():
    return redis.Redis(host='localhost', port=redis_port, db=0)

def create_data_hash(data):
    # Sum all close prices and multiply by the number of candles
    data_hash = data['close'].sum() * len(data)
    return data_hash

def create_cache_key(data_hash, pair_list, time, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type):
    
    # Create a hash of the subset dataframe and the parameters
    params_hash = xxhash.xxh64(str((pair_list, time, price_jump_threshold, last_price_treshold, rolling_window_number, std_for_BB, moving_average_type))).hexdigest()
    cache_key = data_hash + params_hash

    return cache_key

'''def dump(file='cache.pkl'):
    joblib.dump(memory, file)

def load(file='cache.pkl'):
    global memory

    if os.path.exists(file):
        memory = joblib.load(file)
    else:
        # Create a memory object
        memory = joblib.Memory("cache_directory", verbose=0)
'''