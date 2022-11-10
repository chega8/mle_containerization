"""Prepare step"""

import sys
import os
import random
import argparse
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.append("./")
from src.utils import load_data, compress_dataset

params = yaml.safe_load(open("params.yaml"))["prepare"]

split = params["split"]
random.seed(params["seed"])


def prepare(raw_data_path: str, output_dir: str):
    """Prepare raw csv data for featurization

    Args:
        raw_data_path (str): path to raw csv file
        output_dir (str): path to output dir for the train and test parts
    """

    output_train = os.path.join(output_dir, "train.csv")
    output_test = os.path.join(output_dir, "test.csv")

    raw_df = load_data(raw_data_path)
    logger.info(f"Raw data loaded from {raw_data_path}")
    logger.info("Preparing raw data...")

    target = "target"
    features = [col for col in raw_df.columns if col not in ["id", target]]

    raw_df["mean"] = raw_df[features].mean(axis=1)
    raw_df["std"] = raw_df[features].std(axis=1)
    raw_df["min"] = raw_df[features].min(axis=1)
    raw_df["max"] = raw_df[features].max(axis=1)

    features.extend(["mean", "max", "min", "max"])

    raw_df = compress_dataset(raw_df)

    train_df, test_df, train_target, test_target = train_test_split(
        raw_df[features], raw_df[target], test_size=split, random_state=params["seed"]
    )
    logger.info("Preparing completed")

    train_df["target"] = train_target
    test_df["target"] = test_target

    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

    logger.info(f"Prepared data saved in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data", type=str, help="path to raw data")
    parser.add_argument("output", type=str, help="path to output dir")
    args = parser.parse_args()

    raw_data_path = args.raw_data
    output_dir = args.output
    prepare(raw_data_path, output_dir)
