import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Get the working directory of the current script
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, 'trained_model', 'trained_fashion_mnist_model.h5')

# Load the pre-trained model
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define class labels for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    try:
        img = Image.open(image)
        img = img.resize((28, 28))
        img = img.convert('L')  # Convert to grayscale
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape((1, 28, 28, 1))
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            resized_img = image.resize((100, 100))
            st.image(resized_img, caption='Uploaded Image')

        with col2:
            if st.button('Classify'):
                # Preprocess the uploaded image
                img_array = preprocess_image(uploaded_image)
                
                if img_array is not None:
                    # Make a prediction using the pre-trained model
                    result = model.predict(img_array)
                    st.write("Prediction raw output:", result)  # Debugging statement

                    predicted_class = np.argmax(result)
                    prediction = class_names[predicted_class]

                    st.success(f'Prediction: {prediction}')
    except Exception as e:
        st.error(f"Error with uploaded image: {e}")
