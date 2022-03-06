# Script to train machine learning model.

import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from model.ml.data import process_data
from model.ml.model import compute_model_metrics, inference, train_model


def train():
    # Add the necessary imports for the starter code.
    data = pd.read_csv("cleaned_data.csv")
    # Add code to load in the data.

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    model = train_model(X_train, y_train)
    model.save_model("model.json")
    with open("encoder.p", "wb") as pickle_file:
        pickle.dump(encoder, pickle_file)
    with open("lb.p", "wb") as pickle_file:
        pickle.dump(lb, pickle_file)
    # Train and save a model.

    # Validation

    X_test, y_test, encoder, lb = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    y_pred = inference(model, X_train)
    precision, recall, fbeta = compute_model_metrics(y=y_train, preds=y_pred)
    print(f"Train data->  precision: {precision} \n recall: {recall} \n fbeta: {fbeta}")
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y=y_test, preds=y_pred)
    print(f"Test data->  precision: {precision} \n recall: {recall} \n fbeta: {fbeta}")
    return model


def slice_score():
    """
    Execute score checking
    """
    df = pd.read_csv("cleaned_data.csv")
    _, test = train_test_split(df, test_size=0.20)
    trained_model = XGBClassifier()
    trained_model.load_model("model.json")
    with open("encoder.p", "rb") as pickle_file:
        encoder = pickle.load(pickle_file)
    with open("lb.p", "rb") as pickle_file:
        lb = pickle.load(pickle_file)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    output = []
    for cat in cat_features:
        output.append(f"Category {cat}:")
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                label="salary",
                encoder=encoder,
                lb=lb,
                training=False,
            )

            y_preds = inference(trained_model, X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = f"{cls} has {prc} Precision, {rcl} Recall, {fb} FBeta"

            output.append(line)

    with open("slice_score.txt", "w") as out:
        for line in output:
            out.write(line + "\n")


if __name__ == "__main__":
    train()
    slice_score()
