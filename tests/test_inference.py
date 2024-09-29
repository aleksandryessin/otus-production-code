"""
Module: test_inference.py
Description: This module is used to test the inference module.
"""

import os

from sklearn.base import BaseEstimator
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import load_model, predict


def test_load_model():
    model_path = os.path.join("models", "model_in_docker.joblib")
    assert isinstance(load_model(model_path), BaseEstimator)


def test_predict():
    model_path = os.path.join("models", "model_in_docker.joblib")
    model = load_model(model_path)
    df = pd.DataFrame(
        {
            "sepal_length": [5.1],
            "sepal_width": [3.3],
            "petal_length": [1.7],
            "petal_width": [0.5],
        }
    )
    assert predict(model, df) == [0]
