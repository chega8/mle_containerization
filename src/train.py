"""Train model"""

import argparse
import os
import time
import random
import sys
import yaml

import numpy as np

from loguru import logger
from sklearn import metrics, model_selection

from sklearn.linear_model import LogisticRegression

sys.path.append("./")

from src.utils import format_time, load_pickle, save_model


params = yaml.safe_load(open("params.yaml"))["train"]

l2 = params["l2"]
l1 = params["l1"]
random.seed(params["seed"])

N_FOLDS = 5


def crossval_train(features: np.ndarray, target: np.ndarray):
    """Train model and estimate cross-val scores"""
    models = [[LogisticRegression, {}, "logreg"]]
    skfolds = model_selection.StratifiedKFold(n_splits=N_FOLDS, shuffle=False)

    for fold, (t, v) in enumerate(skfolds.split(features, target)):
        x_train, x_val = features[t], features[v]
        y_train, y_val = target[t], target[v]

        for class_name, class_params, name in models:
            tic = time.time()

            clf = class_name(**class_params)
            clf = clf.fit(x_train, y_train)
            preds = clf.predict(x_val).tolist()

            score = metrics.roc_auc_score(y_val, preds)
            logger.info(
                f"MODEL: {name}\tSCORE: {score}\tTIME: {format_time(time.time()-tic)}\tFOLD: {fold}"
            )

        del x_train, x_val, y_train, y_val

    return clf


def train(features_path: str):
    """Train the model on features

    Args:
        features_path (str): path to features
    """
    features = load_pickle(os.path.join(features_path, "train.pkl"))
    target = load_pickle(os.path.join(features_path, "train_target.pkl"))

    logger.info(f"Features loaded from {features_path}")

    start = time.time()
    model = crossval_train(features, target)
    logger.info(f"TOTAL TIME: {format_time(time.time() - start)}")

    os.makedirs("data/models", exist_ok=True)
    save_model(model, "logreg")
    logger.info("Model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_path", type=str, help="path to features")
    args = parser.parse_args()

    train(args.features_path)
