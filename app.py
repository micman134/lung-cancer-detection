import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
try:
    model = tf.keras.models.load_model('cancer_detection_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define the class labels
class_labels = ['Adenocarcinoma', 'Largecellcarcinoma', 'Squamouscellcarcinoma', 'Normal']

# Streamlit app
st.title("Lung Cancer Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Check file type
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension not in ["jpg", "jpeg", "png"]:
        st.error("Invalid file type. Please upload a valid image file.")
    else:
        # Load and preprocess the test image
        test_image = image.load_img(uploaded_file, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Perform inference
        predictions = model.predict(test_image)
        predicted_class = np.argmax(predictions)

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Display probability scores for each class
        st.write("Class Probabilities:")
        for i, label in enumerate(class_labels):
            st.write(f"{label}: {predictions[0][i]:.4f}")

        # Print the classification label
        st.success(f'Predicted Class: {class_labels[predicted_class]}')
