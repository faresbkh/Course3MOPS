import pickle

from fastapi import FastAPI
from pydantic import BaseModel
from xgboost import XGBClassifier

app = FastAPI()

# loading component only once
trained_model = XGBClassifier()
trained_model.load_model("model.json")
with open("encoder.p", "rb") as pickle_file:
    encoder = pickle.load(pickle_file)
with open("lb.p", "rb") as pickle_file:
    lb = pickle.load(pickle_file)


class Value(BaseModel):
    value: int


@app.get("/")
async def get_items():
    return {"message": "Greetings!"}

