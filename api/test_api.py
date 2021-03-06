import json

from fastapi.testclient import TestClient

from api_server import Humain, app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Greetings!"}


def test_get_path_query_low():

    data = Humain(
        age=16,
        workclass="State-gov",
        fnlwgt=77516,
        education="Bachelors",
        education_num=13,
        maritalStatus="Never-married",
        occupation="Adm-clerical",
        relationship="Not-in-family",
        race="White",
        sex="Male",
        capital_gain=2174,
        capital_loss=0,
        hoursPerWeek=40,
        nativeCountry="United-States",
    )
    r = client.post("/", data.json())
    assert r.status_code == 200
    assert r.json() == {"salary_category": "<=50K"}


def test_get_path_query_high():
    data = Humain(
        age=16,
        workclass="State-gov",
        fnlwgt=77516,
        education="Doctorate",
        education_num=16,
        maritalStatus="Married-civ-spouse",
        occupation="Adm-clerical",
        relationship="Husband",
        race="White",
        sex="Male",
        capital_gain=100000,
        capital_loss=0,
        hoursPerWeek=40,
        nativeCountry="United-States",
    )
    r = client.post("/", data.json())
    assert r.status_code == 200
    assert r.json() == {"salary_category": ">50K"}

