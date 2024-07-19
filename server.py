from fastapi import FastAPI, UploadFile, File
import json
from model import build_model2
from PIL import Image
from io import BytesIO
import numpy as np

app = FastAPI()
IMG_shape = (224,224)
NUM_CLASSES = 6
model = build_model2(IMG_shape, NUM_CLASSES, drop_out=0.25)

@app.get("/")
def greeter():
    return {
        "response": "Face Expression"
    }

def image_pipeline(image):
    # step 1: Load the image
    image = Image.open(BytesIO(image))

    # step 2: Resize the image
    image = image.resize((224, 224))
    
    # Step 4: Add a Dimension at end stating it as grayscale channel
    image = np.expand_dims(image, axis=-1)

    return image

def get_model_prediction(image):
    image = image_pipeline(image)
    image = np.expand_dims(image, axis=0)

    # Get Model Prediction
    pred = model.predict(image)

    return pred[0].tolist()

# API to predict the face expression
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image = await image.read()

    prediction = get_model_prediction(image)

    return {
        "prediction": json.dumps(prediction)
    }
    