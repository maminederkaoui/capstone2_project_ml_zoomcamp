# I used Saturn Cloud to train my models. I add this two dependencies in the configuration file of my Saturn Cloud workspace (under pip) because I faced compatibility issues before.

#!pip install tensorflow==2.17.1
#!pip install protobuf==3.20.3

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings("ignore")

# Parameters
input_size = (224, 224)

# test dataset generator with resizing
test_gen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
)

# Test generator
test_ds = test_gen.flow_from_directory(
    'brain-tumor-mri-dataset/versions/1/Testing',              # Path to dataset directory
    target_size=input_size,  # Resize images
    batch_size=32,           # Number of images per batch
    class_mode='categorical',# For multi-class classification
    shuffle=False             # Shuffle data to ensure randomness
)

# Loading the trained model
model = keras.models.load_model('best_epochs_v1/model_v1_20_0.808.keras') # here, add the name of keras file

# Evaluate the model using the best model (according to best epoch's accuracy score)
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.2f}")
    
print(test_ds.class_indices)