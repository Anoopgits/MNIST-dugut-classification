import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("mnist_cnn_model.h5")
# C:\Users\dell\OneDrive\Desktop\mnistdigitclification\mnist_cnn_model.h5
st.title("🖊️ Handwritten Digit Recognition (MNIST)")

st.write("Upload a handwritten digit image (0–9) and the model will predict the digit.")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to grayscale
    image = np.array(image.convert("L"))

    # Resize to 28x28
    image_resized = cv2.resize(image, (28, 28))

    # Normalize
    image_resized = image_resized / 255.0

    # Reshape for model
    image_reshaped = np.reshape(image_resized, (1, 28, 28, 1))

    # Prediction
    prediction = model.predict(image_reshaped)
    pred_label = np.argmax(prediction)

    st.success(f"✅ The model predicts this digit is: **{pred_label}**")
    # feed=st.text_area("give feedback of my project")
    # st.button("submit")
    # import streamlit as st

# Initialize session state
if 'feedback' not in st.session_state:
    st.session_state.feedback = ""

def submit_feedback():
    # Do something with the feedback
    st.write("Feedback submitted:", st.session_state.feedback)
    # Reset the feedback
    st.session_state.feedback = ""

# Text area bound to session state
st.text_area("Give feedback of my project", key='feedback')

# Submit button
st.button("Submit", on_click=submit_feedback)

