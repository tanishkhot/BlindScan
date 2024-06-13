import cv2
import numpy as np
from tensorflow.keras.models import load_model
#source myenv/bin/activate
import os

model = load_model('/Users/tanishkhot/Coding/BlindScan/my_model.h5')  

# Replace 'path_to_training_data' with the path to your training data directory
path_to_training_data = '/Users/tanishkhot/Coding/BlindScan/SampleDS/Currency Notes v2/'
class_names = sorted(os.listdir(path_to_training_data))

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_resized = cv2.resize(frame, (256, 256)) 
    frame_normalized = frame_resized / 255.0  
    frame_expanded = np.expand_dims(frame_normalized, axis=0)  

    
    predictions = model.predict(frame_expanded)
    predicted_class = class_names[np.argmax(predictions, axis=1)[0]]
    

    
    cv2.putText(frame, f'Possible note: {predicted_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (256, 0, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Live Video', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
