#!/usr/bin/env python
# coding: utf-8

import tensorflow.lite as tflite
from PIL import Image
import numpy as np
import requests
from io import BytesIO

def preprocess_input(img):
    img = img.resize((224, 224), Image.NEAREST)   
    x = np.array(img, dtype='float32')
    x = np.array([x])
    x = x * (1./255)
    return x

interpreter = tflite.Interpreter(model_path='tflite_model.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

classes = [
    'glioma',
    'meningioma',
    'notumor',
    'pituitary'
]

# url = 'file:///D:/my_github/capstone2_project_ml_zoomcamp/brain-tumor-mri-dataset/versions/1/Testing/notumor/Te-no_0010.jpg'
# url = 'https://github.com/maminederkaoui/capstone2_project_ml_zoomcamp/blob/8d7c67e228bdb813923168403829debb75815716/brain-tumor-mri-dataset/versions/1/Testing/notumor/Te-noTr_0000.jpg'

def predict(url):
    image = None
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
    else:
        print(f"Failed to load image. Status code: {response.status_code}")
        return None

    X = preprocess_input(image)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()

    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result