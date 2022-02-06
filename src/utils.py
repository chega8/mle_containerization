import pandas as pd
import numpy as np
import os

from loguru import logger

from typing import Tuple, List

from sklearn import impute
from sklearn import ensemble
from sklearn import linear_model
from sklearn import decomposition
from sklearn import preprocessing

import pickle


@logger.catch
def load_data(pth: str) -> pd.DataFrame:
    if not os.path.isfile(pth):
        raise FileExistsError
        
    return pd.read_csv(pth)

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


def memory_usage(data):
    memory = data.memory_usage().sum() / (1024 * 1024)
    # print("Memory usage : {0:.2f}MB".format(memory))
    return memory


def compress_dataset(data):
    memory_before_compress = memory_usage(data)
    print()
    length_interval = 50
    length_float_decimal = 4

    # print('=' * length_interval)
    for col in data.columns:
        col_dtype = data[col][:100].dtype

        if col_dtype != 'object':
            # print("Name: {0:24s} Type: {1}".format(col, col_dtype))
            col_series = data[col]
            col_min = col_series.min()
            col_max = col_series.max()

            if col_dtype == 'float64':
                # print(" variable min: {0:15s} max: {1:15s}".format(str(np.round(col_min, length_float_decimal)), str(np.round(col_max, length_float_decimal))))
                if (col_min > FLOAT16_MIN) and (col_max < FLOAT16_MAX):
                    data[col] = data[col].astype(np.float16)
                    # print("  float16 min: {0:15s} max: {1:15s}".format(str(FLOAT16_MIN), str(FLOAT16_MAX)))
                    # print("compress float64 --> float16")
                elif (col_min > FLOAT32_MIN) and (col_max < FLOAT32_MAX):
                    data[col] = data[col].astype(np.float32)
                    # print("  float32 min: {0:15s} max: {1:15s}".format(str(FLOAT32_MIN), str(FLOAT32_MAX)))
                    # print("compress float64 --> float32")
                else:
                    pass

                memory_after_compress = memory_usage(data)
                # print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                # print('='*length_interval)

            if col_dtype == 'int64':
                # print(" variable min: {0:15s} max: {1:15s}".format(str(col_min), str(col_max)))
                type_flag = 64
                if (col_min > INT8_MIN/2) and (col_max < INT8_MAX/2):
                    type_flag = 8
                    data[col] = data[col].astype(np.int8)
                    # print("     int8 min: {0:15s} max: {1:15s}".format(str(INT8_MIN), str(INT8_MAX)))
                elif (col_min > INT16_MIN) and (col_max < INT16_MAX):
                    type_flag = 16
                    data[col] = data[col].astype(np.int16)
                    # print("    int16 min: {0:15s} max: {1:15s}".format(str(INT16_MIN), str(INT16_MAX)))
                elif (col_min > INT32_MIN) and (col_max < INT32_MAX):
                    type_flag = 32
                    data[col] = data[col].astype(np.int32)
                    # print("    int32 min: {0:15s} max: {1:15s}".format(str(INT32_MIN), str(INT32_MAX)))
                    type_flag = 1
                else:
                    pass
                memory_after_compress = memory_usage(data)
                # print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
                # if type_flag == 32:
                #     print("compress (int64) ==> (int32)")
                # elif type_flag == 16:
                #     print("compress (int64) ==> (int16)")
                # else:
                #     print("compress (int64) ==> (int8)")
                # print('='*length_interval)

    print()
    memory_after_compress = memory_usage(data)
    # print("Compress Rate: [{0:.2%}]".format((memory_before_compress-memory_after_compress) / memory_before_compress))
    
    return data


def scaler(df: pd.DataFrame, FEATURES: List[str], TARGET: List[str], scaler_obj=None) -> Tuple[np.ndarray, np.ndarray]:
    if scaler_obj is None:
        scaler_obj = preprocessing.StandardScaler()

    for col in FEATURES:
        df[col] = scaler_obj.fit_transform(df[col].to_numpy().reshape(-1,1))
        
    X = df[FEATURES].to_numpy().astype(np.float32)
    Y = df[TARGET].to_numpy().astype(np.float32)
    save_model(scaler_obj, 'scaler')

    return X, Y


def save_model(model, name):
    with open(f'volume/{name}.pkl', 'wb') as f:
        f.write(pickle.dumps(model))

def load_model(name):
    with open(f'volume/{name}.pkl', 'rb') as f:
        return pickle.loads(f.read())