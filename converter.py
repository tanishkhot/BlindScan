import tensorflow as tf

# Ensure using TensorFlow's Keras version, not standalone Keras
from tensorflow.keras.models import model_from_json

# Clear any existing session
tf.keras.backend.clear_session()

# Model reconstruction from JSON file.
with open('model.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights from h5 file.
model.load_weights("Code/my_inceptionv4_model.h5")

# Print summary to check the model
model.summary()

# Save the model and weights in a single h5 file, then load again using tf.keras
model.save('model_full.h5')
model = tf.keras.models.load_model('model_full.h5', compile=False)

# Converting a tf.Keras model to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TF Lite model is saved.")
