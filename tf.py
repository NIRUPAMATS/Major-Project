import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or '2' to also display warnings
# import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()


# Save the converted model to a .tflite file
with open("quantized_model.tflite", "wb") as f:
    f.write(tflite_model)
