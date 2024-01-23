import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
try:
    model = tf.keras.models.load_model('cancer_detection_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define the relatable class labels
class_labels = ['Adenocarcinoma', 'Large Cell Carcinoma', 'Flat Cell Lung Cancer', 'No Cancer']

# Streamlit app
st.title("Lung Cancer Detection App")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Charts"])

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

if page == "Prediction":
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
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            predicted_class_probability = predictions[0][predicted_class_index] * 100

            # Display the uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Display probability scores for each class
            st.write("Class Probabilities:")
            for i, label in enumerate(class_labels):
                probability_percentage = predictions[0][i] * 100
                st.write(f"{label}: {probability_percentage:.2f}%")

            # Print the classification label with probability
            st.success(f'Predicted Class: {predicted_class_label} with {predicted_class_probability:.2f}% probability')

elif page == "Charts":
    # Display a horizontal bar chart of model percentages
    if uploaded_file is not None:
        # Load and preprocess the test image
        test_image = image.load_img(uploaded_file, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Perform inference
        predictions = model.predict(test_image)

        # Create a horizontal bar chart
        chart_data = {'Model Percentage': predictions[0] * 100, 'Class Labels': class_labels}
        st.bar_chart(chart_data, use_container_width=True, height=400)
