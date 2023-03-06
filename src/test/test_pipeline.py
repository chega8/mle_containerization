"""Test pipeline"""

import sys
import pytest
import numpy as np
import pandas as pd


sys.path.append(".")

from src.prepare import prepare
from src.featurize import fearurize


def test_prepare():
    """Test prepare"""

    sample = pd.read_csv("src/test/data/test_sample.csv")
    train, test = prepare(sample, 0.2, 4)
    assert train.shape == (80, 286)
    assert test.shape == (20, 286)

    train, test = prepare(sample, 0.3, 4)
    assert train.shape == (70, 286)
    assert test.shape == (30, 286)


def test_featurize():
    """Test prepare"""
    sample = pd.read_csv("src/test/data/test_sample.csv")
    train, test = sample.loc[:50], sample.loc[50:]

    max_features = 100
    train_f, test_f, train_target, test_target, _ = fearurize(train, test, max_features)
    assert len(train_f) == len(train_target)
    assert len(test_f) == len(test_target)

    assert train_f.shape[1] == max_features
    assert test_f.shape[1] == max_features

    max_features = 60
    train_f, test_f, train_target, test_target, _ = fearurize(train, test, max_features)
    assert train_f.shape[1] == max_features
    assert test_f.shape[1] == max_features


def test_evaluate():
    """Test evaluate"""
    assert True


def test_train():
    """Test train"""
    assert True
