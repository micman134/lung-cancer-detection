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
if uploaded_file is not None:
    # Load and preprocess the test image
    test_image = image.load_img(uploaded_file, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image = test_image / 255.0  # Normalize

    if page == "Prediction":
        # Perform inference for prediction
        predictions = model.predict(test_image)
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_class_probability = predictions[0][predicted_class_index] * 100

        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Display probability scores for each class
        st.write("Class Probabilities:")
        for label, probability in zip(class_labels, predictions[0] * 100):
            st.write(f"{label}: {probability:.2f}%")

        # Print the classification label with probability
        st.success(f'Predicted Class: {predicted_class_label} with {predicted_class_probability:.2f}% probability')

    elif page == "Performance Analysis":
        # Perform inference for performance analysis
        predictions = model.predict(test_image)

        # Get predicted class index and probability after inference
        predicted_class_index = np.argmax(predictions)
        predicted_class_probability = predictions[0][predicted_class_index] * 100

        true_class = st.radio("Select the true class of the uploaded image:", class_labels)
        true_class_index = class_labels.index(true_class)

        y_true = [true_class_index]
        y_pred = [predicted_class_index]

        # Sub-navigation for performance analysis
        analysis_option = st.radio("Select analysis option", ["Confusion Matrix", "Classification Report"])

        if analysis_option == "Confusion Matrix":
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            st.write("Confusion Matrix:")
            st.write(cm)

            # Display a heatmap of the confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            st.pyplot(plt)

        elif analysis_option == "Classification Report":
            # Classification Report
            cr = classification_report(y_true, y_pred, labels=np.unique(y_true + y_pred), target_names=class_labels)
            st.write("Classification Report:")
            st.text_area(" ", cr)
