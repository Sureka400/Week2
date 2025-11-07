import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('waste_model.h5')

st.title("♻ Waste Classification for Smart Recycling")
st.write("Upload an image of waste to classify it as Organic or Recyclable.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((224, 224))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]

    if pred > 0.5:
        st.success("♻ Predicted: Recyclable Waste")
    else:
        st.warning("🍎 Predicted: Organic Waste"
