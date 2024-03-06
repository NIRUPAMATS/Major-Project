import tensorflow as tf
import numpy as np
import os
import cv2

def preprocess_input(img):
    img = cv2.resize(img, (50, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    # Setting default cam to webcam and necessary variables
    img = cv2.VideoCapture(0)
    exit = False

    # labelling data
    CODES = {
        0: "nothing"
    }
    alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(1, 27):
        CODES[i] = alpha[i - 1]

    CODES[27] = "del"
    CODES[28] = "space"

    model_path = os.path.join(r"C:\Users\Aiswarya\OneDrive\Desktop\S7\project", "quantized_model.tflite")
    print("Model Path:", os.path.abspath(model_path))
    
    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        ret, frame = img.read()
        frame = cv2.flip(frame, 1)

        # defining frame to be used
        frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)
        frame2 = frame[100:350, 60:310]
        image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        image = preprocess_input(image)  # Add this line to preprocess the image

        # Set the input tensor for the TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], image)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        pred = interpreter.get_tensor(output_details[0]['index'])

        # predicting the letter
        move_code = CODES[np.argmax(pred[0])]
        window_width = 1200
        window_height = 820
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Frame', window_width, window_height)

        ret, frame = img.read()

        frame = cv2.flip(frame, 1)
        frame = cv2.rectangle(frame, (60, 100), (310, 350), (0, 0, 255), 3)

        # displaying our prediction
        cv2.putText(frame, "Letter : {}".format(move_code), (63, 320),
                    font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        if cv2.waitKey(1) & 0xff == ord('q'):
            exit = True
            break
        if exit:
            break
        cv2.imshow('Frame', frame)

    img.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
