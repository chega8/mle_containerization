"""Featurize step"""

import argparse
import os
import sys
from typing import Tuple
import yaml

from pandas import DataFrame

from loguru import logger
from sklearn.preprocessing import StandardScaler
import numpy as np


sys.path.append("./")

from src.utils import load_data, save_pickle


params = yaml.safe_load(open("params.yaml"))["featurize"]

MAX_FEATURES = params["max_features"]


def read_prepared(train_pth: str, test_pth: str) -> Tuple[DataFrame, DataFrame]:
    """Read prepared test and train data"""

    prepared_train = load_data(train_pth)
    prepared_test = load_data(test_pth)
    return prepared_train, prepared_test


def scale_prepared(
    data: DataFrame, scaler: StandardScaler, mode: str = "train"
) -> Tuple[np.ndarray, StandardScaler]:
    """Scale prepared data"""

    if mode == "train":
        data = scaler.fit_transform(data.to_numpy())
    elif mode == "test":
        data = scaler.transform(data.to_numpy())
    else:
        logger.error(f"Unsupported mode type: {mode}, select 'train' or 'test'")
        return data.to_numpy(), scaler

    features = data.astype(np.float32)
    return features, scaler


def fearurize(prepared_path: str, output_dir: str):
    """Make features from train and test parts and makes it ready for the training step

    Args:
        raw_data_path (str): path to raw csv file
        output_dir (str): path to output dir for the train and test parts
    """

    train_input = os.path.join(prepared_path, "train.csv")
    test_input = os.path.join(prepared_path, "test.csv")

    train_output = os.path.join(output_dir, "train.pkl")
    train_output_target = os.path.join(output_dir, "train_target.pkl")

    test_output = os.path.join(output_dir, "test.pkl")
    test_output_target = os.path.join(output_dir, "test_target.pkl")

    scaler_path = os.path.join(output_dir, "scaler.pkl")

    prepared_train, prepared_test = read_prepared(train_input, test_input)

    logger.info(f"Prepared data loaded from {prepared_path}")
    logger.info("Featurization...")

    target = "target"
    features = [col for col in prepared_train.columns if col not in ["id", target]][
        :MAX_FEATURES
    ]

    scaler = StandardScaler()

    # Transform train and test features
    features_train, scaler = scale_prepared(prepared_train[features], scaler)
    features_test, scaler = scale_prepared(prepared_test[features], scaler)

    train_target = prepared_train[target].to_numpy().astype(np.float32)
    test_target = prepared_test[target].to_numpy().astype(np.float32)

    # Save scaler
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Featurization completed")

    save_pickle(scaler, scaler_path)
    # Save features
    save_pickle(features_train, train_output)
    save_pickle(train_target, train_output_target)

    save_pickle(features_test, test_output)
    save_pickle(test_target, test_output_target)

    logger.info(f"features saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_path", type=str, help="path to prepared data")
    parser.add_argument("output", type=str, help="path to output dir")
    args = parser.parse_args()

    prepared_path = args.prepared_path
    output_dir = args.output
    fearurize(prepared_path, output_dir)
