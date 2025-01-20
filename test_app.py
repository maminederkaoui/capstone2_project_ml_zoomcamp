import requests

url = "http://localhost:9696/predict"

img_url = 'https://raw.githubusercontent.com/maminederkaoui/capstone2_project_ml_zoomcamp/main/brain-tumor-mri-dataset/versions/1/Testing/notumor/Te-noTr_0002.jpg'

result = requests.post(url, json={"url": img_url}).json()

print(result)