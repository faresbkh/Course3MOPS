# Script to train machine learning model.

import pandas as pd
from sklearn.model_selection import train_test_split

from model.ml.data import process_data
from model.ml.model import compute_model_metrics, inference, train_model


def main():
    # Add the necessary imports for the starter code.
    data = pd.read_csv("census.csv")
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


if __name__ == "__main__":
    main()
