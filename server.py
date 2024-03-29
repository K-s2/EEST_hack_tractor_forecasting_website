import joblib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from pathlib import Path

# App creation and model loading
app = FastAPI()
model = joblib.load("./model.joblib")

class IrisSpecies(BaseModel):
    """
    Input features validation for the ML model
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Set up templates directory
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory="templates")

@app.get("/items", response_class=HTMLResponse)
async def read_items(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict(iris: IrisSpecies):
    """
    :param iris: input data from the post request
    :return: predicted iris type
    """
    features = [[
        iris.sepal_length,
        iris.sepal_width,   
        iris.petal_length,
        iris.petal_width
    ]]
    prediction = model.predict(features).tolist()[0]
    return {
        "prediction": prediction
    }
