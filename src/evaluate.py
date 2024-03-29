"""Evaluation"""
import argparse
import os
import sys

from sklearn import metrics
from loguru import logger
import mlflow

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
    logger.info("Start evaluation")

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
        mlflow.log_artifact(model_path)

    live.log_sklearn_plot(
        "confusion_matrix", target.squeeze(), predictions_by_class.argmax(-1)
    )
    logger.info("Metrics saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="path to model")
    parser.add_argument("features_path", type=str, help="path to features")
    args = parser.parse_args()

    evaluate(args.model_path, args.features_path)
