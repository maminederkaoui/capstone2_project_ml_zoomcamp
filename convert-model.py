import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('best_epochs_v1/model_v1_20_0.808.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('saved_models/tflite_model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)