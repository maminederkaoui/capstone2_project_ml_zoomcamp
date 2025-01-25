# CAPSTONE 2 : Classification of Brain Tumor MRI's images

## Subject of the project and problem to solve

A brain tumor is an abnormal mass of cells in the brain that can be benign or malignant. Its growth increases pressure in the skull, potentially causing brain damage and life-threatening complications.
<br><br>Early detection and classification of brain tumors are crucial in medical imaging, aiding in choosing the most effective treatment to save patients' lives.
<br><br>Deep learning approaches are revolutionizing health diagnosis with impactful solutions. According to WHO, proper brain tumor diagnosis requires detection, location identification, and classification based on malignancy, grade, and type. This study leverages a CNN-based multi-task model to detect and classify brain tumors using MRI, streamlining tasks like tumor segmentation and location identification.
<br><br>My goal through this project is to develop a comprehensive multi-classification solution aimed at classifying MRI brain scans with high accuracy and efficiency. By leveraging advanced deep learning techniques, the solution will assist in identifying and categorizing brain tumors based on factors such as malignancy, grade, and type.

## The dataset

In Kaggle, I have found an interesting [dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) for MRI brain scans which originates from other sources.

The dataset is composed of two main folders : Training and Testing. Each of the folders has 4 subfolders, labeled by MRI brain scan labels : glioma, meningioma, notumor, pituitary.

## Repository content
- download_dataset.ipynb : Script used to download the dataset locally.
- brain-tumor-mri-dataset : it contains the dataset downloaded from Kaggle. 
- notebook.ipynb : Jypyter notebook that includes Exploratory Data Analysis (EDA), training the models and fine-tuning them, testing various models at the end and saving the final model (and some specific models).
- train.py : notebook converted to scrpit to train the final model.
- best_epochs_v1 : folder contains the best epochs while training the final model using train.py on Saturn Cloud GPU. The best model have a size of 144 MB which is not easy to load it to my repo on Github. This keras file is not loaded to the repo ! You'll find a screenshot on the best epochs saved in the notebook.
![alt text](<pics/best epochs train.png>)
- convert-model.py : a script to convert the the final model from keras to tflite for inference purpose.
- saved_models : contains the final model converted to tflite format and 3 other models (augmented_model_v1, augmented_model_v2 and base_model) that are mentionned in the notebook.
- test_keras_model.py : testing the best epoch (=final model in keras format) on testing dataset. 
- test_tflite_model.ipynb : testing the final model in format tflite on testing dataset.
- predict.py : The flask API that returns classification score on given MRI brain scans.
- test_app.py : A script that sends an image to the flask serving application (=predict.py). 
- Pipfile and Pipfile.lock : the necessary dependencies to install in a virtual environnement.
- Dockerfile : used to create a docker image for the serving application.
- tflite_runtime-2.13.0-cp311-cp311-win_amd64.whl : The wheel of tflite runtime compatible with windows x64 and python 3.11.x. You need to download it manually from this [link](https://github.com/NexelOfficial/tflite-runtime-win/tree/main/win_amd64).


## Instructions on how to run the project

### Saturn Cloud for executing the notebook and the train script
As we know, training a tensorflow model need a powerful GPU. For my case, I used Saturn Cloud to have a workspace working on GPU and I installed specific dependencies in the workspace for that (using terminal or while configuring the workspace under pip block). Under pip block, I defined necessary dependencies with specific versions, which are :
- tensorflow==2.17.1
- protobuf==3.20.3

![alt text](<pics/saturn cloud pip.png>)

### Installing the dependecies in a virtual environment
With the Pipfile, you can install the necessary dependencies in a virtual environnement for the project following these steps :
- step 1 : install pipenv with bash command if you don't have it yet : 
  - pip install pipenv
- step 2 : Navigate to the Project Directory and exactly where the Pipfile is located
- step 3 : Run the following command to install dependencies specified in the Pipfile and create the virtual environment
  <br>*command to execute in bash or terminal* : 
  - pipenv install

### Containerization using Dockerfile
With the DockerFile, you can launch the application in docker container with its dependencies defined in the Pipefile. 
<br>The steps to follow are :
- step 1 : launch Docker Desktop application
- step 2 : Navigate to the Project Directory and exactly where the Dockerfile is located
- step 3 : Run the following command to create the image using the Dockerfile : 
  - docker build -t brain_scan_image .
- step 4 : Run the following command to create and run a container using the created docker image : 
  - docker run -it --rm -p 9696:9696 brain_scan_image