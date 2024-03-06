!pip install keras-squeezenet
import cv2
import os
import numpy as np
from keras.applications import MobileNetV2
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

from google.colab import drive
drive.mount('/content/drive')

zip_file_path = '/content/drive/MyDrive/archive.zip'

import zipfile

extracted_folder_path = '/content/extracted_contents'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_folder_path)

print(os.listdir(extracted_folder_path))
import os

print(os.getcwd())

DATA_PATH = "/content/extracted_contents/asl_alphabet_train/asl_alphabet_train"
print(os.listdir(DATA_PATH))

def load_data(data_path):
    print("Data path:", data_path)
    dataset = []
    for directory in os.listdir(data_path):
        path = os.path.join(data_path, directory)
        if not os.path.isdir(path):
            continue
        for item in os.listdir(path):
            img = cv2.imread(os.path.join(path, item))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (50, 50))
            dataset.append([img, directory])

    data, labels = zip(*dataset)
    return np.array(data), labels

CODES = {"nothing": 0}

def make_labels():
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(1, 27):
        CODES[alpha[i - 1]] = i

    CODES["del"] = 27
    CODES["space"] = 28
    return CODES

def code_conv(label, codes):
    return codes[label]

def load_data(data_path):
    dataset = []
    for directory in os.listdir(data_path):
        path = os.path.join(data_path, directory)
        if not os.path.isdir(path):
            print(f"Skipping non-directory: {path}")
            continue

        for item in os.listdir(path):
            img_path = os.path.join(path, item)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image: {img_path}")
                    continue
                if len(img.shape) == 2:
                    # Image is grayscale, convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (50, 50))
                dataset.append([img, directory])
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

    if not dataset:
        raise ValueError("No images loaded from the provided data path.")

    data, labels = zip(*dataset)
    return np.array(data), labels


def preprocess_labels(labels, codes):
    labels = list(map(lambda x: code_conv(x, codes), labels))
    return to_categorical(labels, num_classes=len(codes))


def normalize_data(data):
    return data.astype('float32') / 255.0

def build_model(input_shape, num_classes):
    model = Sequential()

    # Use MobileNetV2 as base model
    base_model = MobileNetV2(input_shape=input_shape, include_top=False)
    model.add(base_model)

    model.add(Dropout(0.4))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax'))
    return model

def train_model(model, data, labels, optimizer, batch_size, epochs, validation_split=0.2):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)
    return history

def main():
    data, labels = load_data(DATA_PATH)
    codes = make_labels()
    labels = preprocess_labels(labels, codes)
    data = normalize_data(data)

    input_shape = (50, 50, 3)
    num_classes = len(codes)

    model = build_model(input_shape, num_classes)
    optimizer = Adam(learning_rate=0.0001)

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Train the model
    history = train_model(model, X_train, y_train, optimizer, batch_size=16, epochs=10, validation_split=0.2)

    # Save the model
    model.save("model.h5")

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Assuming X and y are your original dataset features and labels
# Replace these with your actual dataset
data, labels = load_data(DATA_PATH)
codes = make_labels()
labels = preprocess_labels(labels, codes)
data = normalize_data(data)

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import load_model
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

import numpy as np
from tensorflow.keras.models import load_model

# Replace '//content//model.h5' with the actual path to your model file
model_path = '//content//model.h5'
model = load_model(model_path)

# Assuming X_val is your validation data
# Make predictions on the validation data
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)


# Generate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_val and y_pred are 1D arrays of integer-encoded class labels
# conf_matrix = confusion_matrix(y_val, y_pred)

# Plot the confusion matrix

# Display classification report
# print(classification_report(y_val, y_pred, target_names=class_names))
plt.figure(figsize=(29, 29))
sns.heatmap(confusion_matrix(y_val.argmax(axis=1), y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# Display classification report
print(classification_report(y_val.argmax(axis=1), y_pred))

import os

model_file_path = '//content//model.tflite'


# Check if the file exists
if os.path.exists(model_file_path):
    # Get the size of the model file in bytes
    model_size_bytes = os.path.getsize(model_file_path)

    # Convert the size to kilobytes (KB) or megabytes (MB) for better readability
    model_size_kb = model_size_bytes / 1024
    model_size_mb = model_size_kb / 1024

    print(f"Model size: {model_size_bytes} bytes, {model_size_kb:.2f} KB, {model_size_mb:.4f} MB")
else:
    print(f"The file '{model_file_path}' does not exist.")

