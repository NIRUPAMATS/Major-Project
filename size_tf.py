import os

# Get the size of the converted model file
tflite_model_size = os.path.getsize("quantized_model.tflite")

# Convert size to megabytes (MB) for better readability
tflite_model_size_mb = tflite_model_size / (1024 * 1024)

print(f"Size of the converted TensorFlow Lite model: {tflite_model_size} bytes ({tflite_model_size_mb:.2f} MB)")
