"""Featurize step"""

import argparse
import os
import sys
import yaml

from loguru import logger
from sklearn.preprocessing import StandardScaler
import numpy as np


sys.path.append("./")

from src.utils import load_data, save_pickle


def fearurize(prepared_train: str, prepared_test: str, max_features: int):
    """Make features from train and test parts and makes it ready for the training step

    Args:
        raw_data_path (str): path to raw csv file
        output_dir (str): path to output dir for the train and test parts
    """

    target = "target"
    features = [col for col in prepared_train.columns if col not in ["id", target]][
        :max_features
    ]

    # Transform train and test features
    scaler = StandardScaler()
    features_train = scaler.fit_transform(prepared_train[features].to_numpy())
    features_test = scaler.transform(prepared_test[features].to_numpy())

    train_target = prepared_train[target].to_numpy().astype(np.float32)
    test_target = prepared_test[target].to_numpy().astype(np.float32)
    logger.info("Featurization completed")
    return features_train, features_test, train_target, test_target, scaler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_path", type=str, help="path to prepared data")
    parser.add_argument("output", type=str, help="path to output dir")
    args = parser.parse_args()

    params = yaml.safe_load(open("params.yaml"))["featurize"]
    max_features = params["max_features"]

    prepared_path = args.prepared_path
    output_dir = args.output

    train_input = os.path.join(prepared_path, "train.csv")
    test_input = os.path.join(prepared_path, "test.csv")
    train_output = os.path.join(output_dir, "train.pkl")
    train_output_target = os.path.join(output_dir, "train_target.pkl")
    test_output = os.path.join(output_dir, "test.pkl")
    test_output_target = os.path.join(output_dir, "test_target.pkl")
    scaler_path = os.path.join(output_dir, "scaler.pkl")

    prepared_train = load_data(train_input)
    prepared_test = load_data(test_input)
    logger.info(f"Prepared data loaded from {prepared_path}")
    logger.info("Featurization...")

    # Featurize data
    train, test, train_target, test_target, scaler = fearurize(
        prepared_train, prepared_test, max_features
    )

    # Save scaler
    os.makedirs(output_dir, exist_ok=True)

    save_pickle(scaler, scaler_path)

    # Save features
    save_pickle(train, train_output)
    save_pickle(train_target, train_output_target)

    save_pickle(test, test_output)
    save_pickle(test_target, test_output_target)

    logger.info(f"features saved in {output_dir}")
