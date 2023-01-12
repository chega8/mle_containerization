"""Evaluation"""
import argparse
import os
import sys
import yaml
import random

import pandas as pd

from sklearn import metrics
import mlflow

import matplotlib.pyplot as plt

from dvclive import Live


sys.path.append("./")

from src.utils import load_pickle


live = Live("evaluation")

# def log_search_params(search_results):
#     params = search_results.cv_results_['params']
#     scores = search_results.cv_results_['mean_test_score']


def evaluate(model_path: str, features_path: str):
    """Evaluate model

    Args:
        model_path (str): path to model
        features_path (str): path to test featuers
    """

    features = load_pickle(os.path.join(features_path, "test.pkl"))
    target = load_pickle(os.path.join(features_path, "test_target.pkl"))

    model = load_pickle(model_path)

    predictions_by_class = model.predict_proba(features)
    predictions = predictions_by_class[:, 1]

    avg_prec = metrics.average_precision_score(target, predictions)
    auc = metrics.roc_auc_score(target, predictions)

    live.log_sklearn_plot("roc", target.tolist(), predictions.tolist())
    live.log_sklearn_plot("precision_recall", target.tolist(), predictions.tolist())
    live.log_metric("avg_prec", avg_prec)
    live.log_metric("roc_auc", auc)

    with mlflow.start_run():
        params = {param: v for param, v in model.get_params().items() if v is not None}
        mlflow.log_params(params)

        mlflow.log_metric("average precision", avg_prec)
        mlflow.log_metric("roc auc", auc)

        mlflow.sklearn.log_model(model, "model")

    live.log_sklearn_plot(
        "confusion_matrix", target.squeeze(), predictions_by_class.argmax(-1)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="path to model")
    parser.add_argument("features_path", type=str, help="path to features")
    args = parser.parse_args()

    evaluate(args.model_path, args.features_path)
