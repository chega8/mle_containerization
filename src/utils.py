import pickle
from typing import Any
from loguru import logger
import pandas as pd
import numpy as np
import os


INT8_MIN    = np.iinfo(np.int8).min
INT8_MAX    = np.iinfo(np.int8).max
INT16_MIN   = np.iinfo(np.int16).min
INT16_MAX   = np.iinfo(np.int16).max
INT32_MIN   = np.iinfo(np.int32).min
INT32_MAX   = np.iinfo(np.int32).max

FLOAT16_MIN = np.finfo(np.float16).min
FLOAT16_MAX = np.finfo(np.float16).max
FLOAT32_MIN = np.finfo(np.float32).min
FLOAT32_MAX = np.finfo(np.float32).max


def format_time(seconds):
    """
    Formates time in human readable form

    Args:
        seconds: seconds passed in a process
    Return:
        formatted string in form of MM:SS or HH:MM:SS
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    result = ''

    if h > 0:
        if h < 10:
            h = '0' + str(h)
        else:
            h = str(h)
        h += ' Hr'
        result += h
        result += ' '
    
    if m > 0:
        if m < 10:
            m = '0' + str(m)
        else:
            m = str(m)
        m += ' min'
        result += m
        result += ' '

    if s < 10:
        s = '0' + str(s)
    else:
        s = str(s)
    s += ' sec'
    result += s
    
    return result

def memory_usage(data):
    memory = data.memory_usage().sum() / (1024 * 1024)
    print("Memory usage : {0:.2f}MB".format(memory))
    return memory

def compress_dataset(data):
    memory_before_compress = memory_usage(data)
    print()

    # print('=' * length_interval)
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if col_dtype != 'object':
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == 'float64':
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                else:
                    pass

                # memory_after_compress = memory_usage(data)
                # print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))

            if col_dtype == 'int64':
                if (col_min > INT8_MIN/2) and (col_max < INT8_MAX/2):
                    data[col] = data[col].astype(np.int8)
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    data[col] = data[col].astype(np.int16)
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    data[col] = data[col].astype(np.int32)
                else:
                    pass
                
                # memory_after_compress = memory_usage(data)
                # print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))

    print()
    memory_after_compress = memory_usage(data)
    print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
    
    return data

@logger.catch
def load_data(pth: str) -> pd.DataFrame:
    if not os.path.isfile(pth):
        raise FileExistsError
        
    return pd.read_csv(pth)

def save_pickle(data: Any, path: str):
    with open(path, 'wb') as f:
        f.write(pickle.dumps(data))
        
def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.loads(f.read())
    
def save_model(model, name):
    save_pickle(model, f'data/models/{name}.pkl')

def load_model(name):
    return load_pickle(f'data/models/{name}.pkl')