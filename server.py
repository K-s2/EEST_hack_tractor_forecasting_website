import joblib
import uvicorn
import csv
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import io
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
async def show_prediction(request: Request, predictions: str = None):
    return templates.TemplateResponse("prediction.html", {"request": request, "predictions": predictions})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    :param file: input CSV file from the post request
    :return: predicted iris type for each row
    """
    # Read the file 
    df = pd.read_csv(file.file, nrows=1)
    
    # Make sure it has the right columns
    assert set(df.columns) == {"sepal_length", "sepal_width", "petal_length", "petal_width"}, f"Incorrect columns: {df.columns}"
    
    # Ensure columns are in the right order
    df = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

    logger.debug(f"File received: {file.filename}")
    
    predictions = model.predict(df.values).tolist()[0]

    logger.debug(f"Predictions done: {predictions}")

    # Return the predictions
    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
