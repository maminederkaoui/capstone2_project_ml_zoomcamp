import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('best_epochs_v1/model_v1_13_0.764.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('model-v1.tflite', 'wb') as f_out:
    f_out.write(tflite_model)