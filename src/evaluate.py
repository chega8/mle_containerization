"""Evaluation"""
import argparse
import os
import sys
import json
import math

import pandas as pd

from sklearn import metrics

import matplotlib.pyplot as plt

from dvclive import Live


sys.path.append("./")

from src.utils import load_pickle

live = Live("evaluation")


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

    live.log_sklearn_plot("roc", target, predictions)
    live.log_sklearn_plot("precision_recall", target, predictions)
    live.log_metric("avg_prec", metrics.average_precision_score(target, predictions))
    live.log_metric("roc_auc", metrics.roc_auc_score(target, predictions))

    # ... confusion matrix plot
    live.log_sklearn_plot(
        "confusion_matrix", target.squeeze(), predictions_by_class.argmax(-1)
    )

    # ... and finally, we can dump an image, it's also supported:
    fig, axes = plt.subplots(dpi=100)
    fig.subplots_adjust(bottom=0.2, top=0.95)
    importances = model.coef_[0]
    forest_importances = pd.Series(importances).nlargest(n=30)
    axes.set_ylabel("Mean decrease in impurity")
    forest_importances.plot.bar(ax=axes)
    fig.savefig(os.path.join("evaluation", "importance.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="path to model")
    parser.add_argument("features_path", type=str, help="path to features")
    args = parser.parse_args()

    evaluate(args.model_path, args.features_path)
