from loguru import logger
from sklearn import preprocessing
import numpy as np
import argparse
import os
import sys
import yaml

sys.path.append("./")

from src.utils import save_model, load_data, save_pickle

"""Feturize step of the pipeline"""

params = yaml.safe_load(open("params.yaml"))["featurize"]

MAX_FEATURES = params["max_features"]


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

    prepared_train = load_data(train_input)
    prepared_test = load_data(test_input)

    logger.info(f"Prepared data loaded from {prepared_path}")
    logger.info(f"Featurization...")

    scaler = preprocessing.StandardScaler()

    TARGET = "target"
    FEATURES = [col for col in prepared_train.columns if col not in ["id", TARGET]][
        :MAX_FEATURES
    ]

    # Transform train and test features
    for col in FEATURES:
        prepared_train[col] = scaler.fit_transform(
            prepared_train[col].to_numpy().reshape(-1, 1)
        )
        prepared_test[col] = scaler.transform(
            prepared_test[col].to_numpy().reshape(-1, 1)
        )

    features_train = prepared_train[FEATURES].to_numpy().astype(np.float32)
    train_target = prepared_train[TARGET].to_numpy().astype(np.float32)

    features_test = prepared_test[FEATURES].to_numpy().astype(np.float32)
    test_target = prepared_test[TARGET].to_numpy().astype(np.float32)

    # Save scaler
    os.makedirs(output_dir, exist_ok=True)

    save_pickle(scaler, scaler_path)
    logger.info("Featurization completed")

    # Save features
    save_pickle(features_train, train_output)
    save_pickle(train_target, train_output_target)

    save_pickle(features_test, test_output)
    save_pickle(test_target, test_output_target)

    logger.info(f"Features saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prepared_path", type=str, help="path to prepared data")
    parser.add_argument("output", type=str, help="path to output dir")
    args = parser.parse_args()

    prepared_path = args.prepared_path
    output_dir = args.output
    fearurize(prepared_path, output_dir)
