import requests

from api_server import Humain

data: Humain = Humain(
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
r = requests.post("https://mlopscourse.herokuapp.com/", data.json())

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
