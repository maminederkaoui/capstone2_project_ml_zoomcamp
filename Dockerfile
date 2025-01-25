# Use Python 3.12 slim image as the base image
FROM python:3.12-slim

# Install pipenv to manage dependencies
RUN pip install pipenv
RUN pipenv run pip install tflite_runtime-2.13.0-cp311-cp311-win_amd64.whl

# Set the working directory in the container
WORKDIR /app

# Copy Pipfile and Pipfile.lock into the container
COPY Pipfile Pipfile.lock /app/

# Install the dependencies using pipenv (ignoring the Pipfile if lock is present)
RUN pipenv install --deploy --ignore-pipfile

# Copy the application files into the container
COPY ["saved_models/tflite_model.tflite", "predict.py", "./"]

# Expose the port for the app
EXPOSE 9696

# Run the app using waitress-serve inside the pipenv environment
ENTRYPOINT ["pipenv", "run", "waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
