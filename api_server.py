import os
import pickle
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from model.ml.data import process_data
from model.ml.model import inference

# from xgboost import XGBClassifier
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")
app = FastAPI()


with open("encoder.p", "rb") as pickle_file:
    encoder = pickle.load(pickle_file)
with open("lb.p", "rb") as pickle_file:
    lb = pickle.load(pickle_file)
with open("model.p", "rb") as pickle_file:
    trained_model = pickle.load(pickle_file)
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
columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-Week",
    "native-country",
]


class Humain(BaseModel):

    age: int
    workclass: Literal[
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ]
    fnlwgt: int

    education: Literal[
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ]
    education_num: int
    maritalStatus: Literal[
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ]
    occupation: Literal[
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ]
    relationship: Literal[
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
    ]
    race: Literal["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hoursPerWeek: int
    nativeCountry: Literal[
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ]


@app.get("/")
async def get_items():
    return {"message": "Greetings!"}


@app.post("/")
async def infer(data: Humain):

    dataframe = pd.DataFrame(data.dict(), index=[0])
    dataframe.rename(
        columns={
            "nativeCountry": "native-country",
            "hoursPerWeek": "hours-per-Week",
            "capital_gain": "capital-gain",
            "capital_loss": "capital-loss",
            "education_num": "education-num",
            "maritalStatus": "marital-status",
        },
        inplace=True,
    )
    print(dataframe.head())
    X, _, _, _ = process_data(
        dataframe, categorical_features=cat_features, encoder=encoder, lb=lb, training=False,
    )
    pred = inference(trained_model, X)
    y = lb.inverse_transform(pred)
    return {"salary_category": y[0]}
