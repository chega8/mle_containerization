from loguru import logger
import argparse
import os
import sys
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import pandas as pd
from dvclive import Live
import json

sys.path.append("./")

from src.utils import load_pickle

live = Live("evaluation")


def evaluate(model_path: str, features_path: str):
    """Evaluate model

    Args:
        model_path (str): path to model
        features_path (str): path to test featuers
    """

    x = load_pickle(os.path.join(features_path, "test.pkl"))
    y = load_pickle(os.path.join(features_path, "test_target.pkl"))

    model = load_pickle(model_path)

    predictions_by_class = model.predict_proba(x)
    predictions = predictions_by_class[:, 1]

    live.log_plot("roc", y, predictions)
    live.log("avg_prec", metrics.average_precision_score(y, predictions))
    live.log("roc_auc", metrics.roc_auc_score(y, predictions))

    precision, recall, prc_thresholds = metrics.precision_recall_curve(y, predictions)
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]
    prc_file = os.path.join("evaluation", "plots", "precision_recall.json")
    os.makedirs(os.path.join("evaluation", "plots"), exist_ok=True)
    with open(prc_file, "w") as fd:
        json.dump(
            {
                "prc": [
                    {"precision": p, "recall": r, "threshold": t}
                    for p, r, t in prc_points
                ]
            },
            fd,
            indent=4,
        )

    # ... confusion matrix plot
    live.log_plot("confusion_matrix", y.squeeze(), predictions_by_class.argmax(-1))

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
