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
            <form id="predictForm" onsubmit="event.preventDefault(); submitData()">
                Sepal Length: <input type="text" name="sepal_length" id="sepal_length"><br>
                Sepal Width: <input type="text" name="sepal_width" id="sepal_width"><br>
                Petal Length: <input type="text" name="petal_length" id="petal_length"><br>
                Petal Width: <input type="text" name="petal_width" id="petal_width"><br>
                <input type="submit" value="Predict">
            </form>

            <script>
                async function submitData() {
                    const formData = {
                        sepal_length: parseFloat(document.getElementById('sepal_length').value),
                        sepal_width: parseFloat(document.getElementById('sepal_width').value),
                        petal_length: parseFloat(document.getElementById('petal_length').value),
                        petal_width: parseFloat(document.getElementById('petal_width').value)
                    };

                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(formData)
                    });

                    const data = await response.json();
                    alert(data.prediction);
                }
            </script>
        </body>
    </html>

    """

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

if name == 'main':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)

а я хочу чтобы весь штмл код был в отдельном html файлике
