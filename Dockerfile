FROM python:3.8-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "./"]

RUN pipenv install --system --deploy

COPY ["tflite_model.tflite","predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]