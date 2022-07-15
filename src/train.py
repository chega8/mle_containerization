from loguru import logger
from sklearn import metrics, model_selection
import argparse
import os
import time
import random
import sys
from sklearn.linear_model import LogisticRegression
import yaml

sys.path.append('./')

from src.utils import format_time, load_pickle, save_model, load_data, save_pickle, load_features


params = yaml.safe_load(open("params.yaml"))["train"]

l2 = params["l2"]
l1 = params["l1"]
random.seed(params["seed"])

def train(features_path: str):
    """Train the model on features

    Args:
        features_path (str): path to features
    """
    x = load_pickle(os.path.join(features_path, "train.pkl"))
    y = load_pickle(os.path.join(features_path, "train_target.pkl"))
    
    logger.info(f"Features loaded from {features_path}")

    models = [
        [LogisticRegression, {}, 'logreg']
    ]

    N_FOLDS = 5
    start = time.time()

    skfolds = model_selection.StratifiedKFold(n_splits=N_FOLDS, shuffle=False)

    for fold, (t, v) in enumerate(skfolds.split(x, y)):
        x_train, x_val = x[t], x[v]
        y_train, y_val = y[t], y[v]
        
        for class_name, class_params, name in models:
            tic = time.time()

            clf = class_name(**class_params)
            clf = clf.fit(x_train, y_train)
            preds = clf.predict(x_val).tolist()
            
            score = metrics.roc_auc_score(y_val, preds)
            logger.info(f"MODEL: {name}\tSCORE: {score}\tTIME: {format_time(time.time()-tic)}\tFOLD: {fold}")
            
        del x_train, x_val, y_train, y_val
    
    logger.info(f'TOTAL TIME: {format_time(time.time() - start)}')

    os.makedirs('data/models', exist_ok=True)
    save_model(clf, 'logreg')
    logger.info(f"Model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('features_path', type=str, help='path to features')
    args = parser.parse_args()
    
    train(args.features_path)
