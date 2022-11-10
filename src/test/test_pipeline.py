"""Test pipeline"""

import sys
import pytest
from sklearn.preprocessing import StandardScaler
import numpy as np


sys.path.append("./")

from src.featurize import read_prepared, scale_prepared


@pytest.mark.skip
def test_featurization():
    """Test featurization"""

    train_pth = "data/prepared/train.csv"
    test_pth = "data/prepared/train.csv"
    prepared_train, prepared_test = read_prepared(train_pth, test_pth)

    features = ["f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14"]

    scaler = StandardScaler()
    features, scaler = scale_prepared(prepared_train, scaler)
    assert np.linalg.norm(features) == np.float32(4813.489)

    features, scaler = scale_prepared(prepared_test, scaler, mode="test")
    assert np.linalg.norm(features) == np.float32(4813.489)

    features, scaler = scale_prepared(prepared_test, scaler, "none")
    assert np.linalg.norm(features) == 282891213.48606616


def test_prepare():
    """Test prepare"""
    assert True


def test_evaluate():
    """Test evaluate"""
    assert True
