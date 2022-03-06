import pandas as pd
import pytest

from model.ml.data import eda, process_data
from model.ml.model import inference, train_model


def test_eda():
    """check null values"""
    df = eda(pd.read_csv("cleaned_data.csv"))
    assert df.shape == df.dropna().shape
    assert "?" not in df.values
    assert " " not in df.values


def test_process():
    """check that X and Y data have same length"""
    df = pd.read_csv("cleaned_data.csv")
    X_train, y_train, _, _ = process_data(
        df,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )
    assert X_train.shape[0] == y_train.shape[0]


def test_inference():
    df = pd.read_csv("cleaned_data.csv")
    X_train, y_train, _, _ = process_data(
        df,
        categorical_features=[
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        label="salary",
        training=True,
    )
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert len(preds) == len(y_train)
