import streamlit as st
import pandas as pd
import requests
from io import BytesIO
import json
import numpy as np

st.header('Face expression')

class_labels = {
    0: 'Surprise',
    1: 'Sad',
    2: 'Ahegao',
    3: 'Happy',
    4: 'Neutral',
    5: 'Angry'
}

#Button to upload Xrays Image
uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])

#Check if a file has been uploaded
if uploaded_file is not None:
    #Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully");
else:
    st.write("Please upload an image file");

def predict():
    if uploaded_file is not None:
        image = uploaded_file.getvalue()

        response = requests.post(
            "https://monica22-faceexpression.hf.space/predict",
            files = {
                "image": BytesIO(image)
            }
        )

        response = json.loads(response._content.decode('utf-8'))
        predict = json.loads(response["prediction"])
        label = np.argmax(predict)

        # st.write(prediction)
        # st.write(label)
        st.markdown("#### Prediction - blue:[{class_labels[label]}]")

st.button(
    label="Predict",
    on_click=predict,
    type="primary"
)