import os

model_file_path = 'C:\\Users\\Aiswarya\\OneDrive\\Desktop\\S7\\project\\model.h5'


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
