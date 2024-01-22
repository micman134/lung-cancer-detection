import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('cancer_detection_model.h5')

# Define the class labels
class_labels = ['Adenocarcinoma', 'Largecellcarcinoma', 'Squamouscellcarcinoma', 'Normal']

# Streamlit app
st.title("Cancer Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the test image
    test_image = image.load_img(uploaded_file, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize

    # Perform inference
    predictions = model.predict(test_image)
    predicted_class = np.argmax(predictions)

    # Display the uploaded image
    st.image(test_image[0], caption="Uploaded Image", use_column_width=True)

    # Print the classification label
    st.success(f'Predicted Class: {class_labels[predicted_class]}')
