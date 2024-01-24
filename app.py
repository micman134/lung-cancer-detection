import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
try:
    model = tf.keras.models.load_model('cancer_detection_model.h5')
except Exception as e:
    st.error(f"Error loading the model: {str(e)}")

# Define the relatable class labels
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma', 'normal']

# Streamlit app
st.title("Lung Cancer Detection App")

# Sidebar navigation
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Performance Analysis"])

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose a test image...", type=["jpg", "jpeg", "png"])

# Model performance analysis
true_classes = []
predicted_classes = []

if page == "Prediction":
    if uploaded_file is not None:
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

elif page == "Performance Analysis":
    uploaded_files = st.file_uploader("Choose test images (or finish to analyze):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    for uploaded_file in uploaded_files:
        # Load and preprocess the test image
        test_image = image.load_img(uploaded_file, target_size=(150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize

        # Perform inference
        predictions = model.predict(test_image)

        # Get predicted class index and probability after inference
        predicted_class_index = np.argmax(predictions)
        predicted_class_probability = predictions[0][predicted_class_index] * 100

        true_class = st.radio("Select the true class of the uploaded image:", class_labels)
        true_class_index = class_labels.index(true_class)

        true_classes.append(true_class_index)
        predicted_classes.append(predicted_class_index)

    # After processing all images
    if true_classes and predicted_classes:
        unique_true_classes = np.unique(true_classes)
        unique_predicted_classes = np.unique(predicted_classes)

        if len(unique_true_classes) != len(class_labels) or len(unique_predicted_classes) != len(class_labels):
            st.warning("Number of unique classes does not match the expected number.")
        else:
            cm = confusion_matrix(true_classes, predicted_classes)
            st.write("Confusion Matrix:")
            st.write(cm)

            cr = classification_report(true_classes, predicted_classes, target_names=class_labels)
            st.write("Classification Report:")
            st.text_area(" ", cr)

            # Display a heatmap of the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            st.pyplot(plt)
