import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

class_names = ['10', '100', '20', '200', '2000', '50', '500']

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='NewModel.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Configure camera settings
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to preprocess the frame
def preprocess_frame(frame, size):
    frame_resized = cv2.resize(frame, size)  # Adjust size to match model's expected input
    frame_preprocessed = tf.keras.applications.resnet.preprocess_input(frame_resized)
    return frame_preprocessed

# Prediction loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Preprocess the image to fit the model input
        processed_frame = preprocess_frame(frame, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_frame.astype(np.float32))

        # Run inference
        interpreter.invoke()

        # Extract the output and display the prediction
        output_data = interpreter.get_tensor(output_details[0]['index'])
        predictions = tf.nn.softmax(output_data[0])  # Apply softmax to logits
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Max probability as confidence

        # Display predicted class and confidence on the frame
        if confidence < 19.5:
            label = "No currency detected"
        else:
            label = f"Note: {predicted_class} Rupees   Confidence: {confidence:.2f}%"

        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Frame', frame)

        # Announce the prediction when spacebar is pressedq
        if cv2.waitKey(1) == 32:  # ASCII value for spacebar is 32
            if confidence >= 19.5:
                engine.say(f"{predicted_class} Rupees")
            else:
                engine.say("No currency detected")    
            engine.runAndWait()
        if cv2.waitKey(1) == ord('q'):
            break

finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()