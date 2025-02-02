import streamlit as st

import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('real_vs_ai_model.h5')

class_name = ["real", "fake"]

st.title("AI vs REal ART classifier")

st.write("Upload an image to classify")

uploaded_file = st.file_uploader("choose an image", type = ["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,caption = "uploaded image", use_column_width=True)
    if st.button("Classify"):
        img = image.resize((224,224))
        img_arr = np.array(img)/255.0
        img_array = np.expand_dims(img_arr,axis=0)

        pred = model.predict(img_array)[0][0]

        result = class_name[0] if pred < 0.5 else class_name[1]
        st.subheader(f"Prediction : {result}")