from cmath import log
import imp
import sys
from collections import defaultdict
import numpy as np

import time

from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics

sys.path.append('./')

import click

from src.utils import compress_dataset, scaler, format_time, load_data, save_model

@click.command()
@click.argument('train_path')
def train(train_path: str):
    train_df = load_data(train_path)
    logger.info(f"train data loaded from {train_path}")

    TARGET = 'target'
    FEATURES = [col for col in train_df.columns if col not in ['id', TARGET]]

    train_df["mean"] = train_df[FEATURES].mean(axis=1)
    train_df["std"] = train_df[FEATURES].std(axis=1)
    train_df["min"] = train_df[FEATURES].min(axis=1)
    train_df["max"] = train_df[FEATURES].max(axis=1)

    FEATURES.extend(['mean', 'max', 'min', 'max'])

    train_df = compress_dataset(train_df)

    X, Y = scaler(train_df, FEATURES, TARGET)
    logger.info("Data preprocessing completed")

    models = [
        [LogisticRegression, {}, 'logreg']
    ]

    N_FOLDS = 5
    start = time.time()

    skfolds = model_selection.StratifiedKFold(n_splits=N_FOLDS, shuffle=False)

    for fold, (t, v) in enumerate(skfolds.split(X, Y)):
        x_train, x_val = X[t], X[v]
        y_train, y_val = Y[t], Y[v]
        
        for class_name, class_params, name in models:
            tic = time.time()

            clf = class_name(**class_params)
            clf = clf.fit(x_train, y_train)
            preds = clf.predict(x_val).tolist()
            
            score = metrics.roc_auc_score(y_val, preds)
            logger.info(f"MODEL: {name}\tSCORE: {score}\tTIME: {format_time(time.time()-tic)}\tFOLD: {fold}")
            
        del x_train, x_val, y_train, y_val
            
    logger.info(f'TOTAL TIME: {format_time(time.time() - start)}')

    save_model(clf, 'logreg')
    logger.info(f"Model saved")


if __name__ == "__main__":
    train()