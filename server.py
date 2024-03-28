import joblib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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

@app.get("/items/", response_class=HTMLResponse)
async def read_items():
    return """
    <html>
    <head>
        <title>Flower Species Prediction</title>
    </head>
    <body>
        <h1>Enter Flower Measurements</h1>
        <form action="/predict" method="post">
            Sepal Length: <input type="text" name="sepal_length"><br>
            Sepal Width: <input type="text" name="sepal_width"><br>
            Petal Length: <input type="text" name="petal_length"><br>
            Petal Width: <input type="text" name="petal_width"><br>
            <input type="submit" value="Predict">
        </form>
    </body>
</html>
    """

@app.post('/predict')
def predict(iris: IrisSpecies):
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

if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
