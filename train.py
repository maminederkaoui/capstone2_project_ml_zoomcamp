# I used Saturn Cloud to train my models. I add this two dependencies in the configuration file of my Saturn Cloud workspace (under pip) because I faced compatibility issues before.

#!pip install tensorflow==2.17.1
#!pip install protobuf==3.20.3

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Parameters
input_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 32
nb_epochs = 7

# Define ImageDataGenerator for loading and splitting to train et validation datasets
data_gen = ImageDataGenerator(validation_split=0.25)

# Load training data (original, without augmentation)
train_loader = data_gen.flow_from_directory(
    'brain-tumor-mri-dataset/versions/1/Training',
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Load validation data
val_loader = data_gen.flow_from_directory(
    'brain-tumor-mri-dataset/versions/1/Training',
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save all data to variables
train_images, train_labels = zip(*(train_loader[i] for i in range(len(train_loader))))
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

val_images, val_labels = zip(*(val_loader[i] for i in range(len(val_loader))))
val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)


# %%
def aug_generator(train_images,
                train_labels,
                rotation_range = 0, 
                height_shift_range=0, 
                zoom_range=0, 
                brightness_range=None):

    # Preparing generator for training dataset
    aug_gen = ImageDataGenerator(
        rescale=1.0 / 255, 
        rotation_range=rotation_range,       
        height_shift_range=height_shift_range,  
        zoom_range=zoom_range,         
        brightness_range=brightness_range
    )

    # Generating training dataset with data augmentation
    train_ds = aug_gen.flow(train_images, train_labels, batch_size=batch_size, shuffle=True)

    return train_ds

def aug_generator_val_test(val_images, val_labels):
    aug_gen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    output_ds = aug_gen.flow(val_images, val_labels, batch_size=batch_size, shuffle=False)

    return output_ds

#### Training the best model and saving it

# Data after data augmentation
augmented_train = aug_generator(train_images, train_labels, height_shift_range=0.05)
augmented_val = aug_generator_val_test(val_images, val_labels)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(4, activation='softmax')  # 4 classes
])

learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

# To save the best model reached at each epoch
checkpoint = keras.callbacks.ModelCheckpoint(
    'model_v1_{epoch:02d}_{val_accuracy:.3f}.keras',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

# Compile the model
model.compile(optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

# Train the model
history = model.fit(augmented_train,
    validation_data=augmented_val,
    epochs=15, batch_size=32,
    callbacks=[checkpoint])
