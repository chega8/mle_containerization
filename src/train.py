"""Train model"""

import argparse
import os
import time
import random
import sys
import yaml

import numpy as np
import pandas as pd

from loguru import logger
from sklearn import metrics, model_selection

from sklearn.linear_model import LogisticRegression

import mlflow

sys.path.append("./")

from src.utils import format_time, load_pickle, save_model
from src.models.models import Logreg, SVM, GB


params = yaml.safe_load(open("params.yaml"))["train"]

model_type = params["model"]

random.seed(params["seed"])

N_FOLDS = 5


def gridsearch_train(features: np.ndarray, target: np.ndarray):
    """Grid search"""

    models = {
        "logreg": Logreg(),
        "svm": SVM(),
        "gb": GB(),
    }

    model = models[model_type].train(features, target)
    return model


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
    features = load_pickle(os.path.join(features_path, "train.pkl"))[:1000]
    target = load_pickle(os.path.join(features_path, "train_target.pkl"))[:1000]
    logger.info(f"Features loaded from {features_path}")

    model = gridsearch_train(features, target)

    os.makedirs("data/models", exist_ok=True)
    save_model(model, "model")
    logger.info("Model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_path", type=str, help="path to features")
    args = parser.parse_args()

    train(args.features_path)
