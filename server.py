import joblib
import uvicorn
import csv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
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

# Путь к папке со статическими файлами
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/items", response_class=HTMLResponse)
async def read_items(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/prediction_result", response_class=HTMLResponse)
async def show_prediction(request: Request, prediction: str):
    return templates.TemplateResponse("prediction.html", {"request": request, "prediction": prediction})

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
