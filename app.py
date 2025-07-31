import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model("cat_dog_model.keras")

# Title
st.title("Cat vs Dog Classifier 🐱🐶")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        st.write("### It's a Dog 🐶")
    else:
        st.write("### It's a Cat 🐱")
