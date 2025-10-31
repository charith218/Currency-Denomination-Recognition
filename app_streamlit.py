import os
import sys

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Ensure the local package directory is importable both locally and on Render
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from scanner.classifier import Predictor

st.set_page_config(page_title="Currency Denomination Scanner", page_icon="💵", layout="centered")

st.title("💵 Currency Denomination Recognition")
st.caption("Starter app: upload a banknote image to get a predicted denomination (baseline ORB matcher).")

@st.cache_resource
def get_predictor():
    return Predictor()

pred = get_predictor()

uploaded = st.file_uploader("Upload a banknote photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    img_np = np.array(image)[:, :, ::-1]  # RGB->BGR for OpenCV
    results = pred.predict(img_np, topk=3)

    st.image(image, caption="Input", use_column_width=True)
    st.subheader("Predictions")
    for label, conf in results:
        st.write(f"**{label}** — confidence: {conf*100:.1f}%")

st.markdown("---")
st.markdown("**Tip:** Replace images in `data/references/` with real banknotes to improve accuracy, or plug-in your own CNN in `scanner/classifier.py`.")
