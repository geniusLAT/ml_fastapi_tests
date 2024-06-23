from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline('sentiment-analysis', model=model_name)



@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
