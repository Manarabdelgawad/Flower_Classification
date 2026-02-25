import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES   = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE  = (224, 224)
MODEL_PATH = "models/best_model.h5"

CLASS_EMOJI = {
    'daisy':     '🌼',
    'dandelion': '🌻',
    'rose':      '🌹',
    'sunflower': '🌻',
    'tulip':     '🌷',
}

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_flower_model():
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at **{MODEL_PATH}**.\n\n"
            "Make sure `best_model.h5` is inside a `models/` folder "
            "in the same directory as `app.py`."
        )
        return None
    return load_model(MODEL_PATH)

# ── Prediction helper ─────────────────────────────────────────────────────────
def predict(model, pil_image: Image.Image):
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx   = np.argmax(probs)
    return CLASSES[idx], float(probs[idx]), probs

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌸 Flower Classification")
st.markdown("Upload a flower photo and the model will identify it!")

model = load_flower_model()

uploaded = st.file_uploader(
    "Choose an image (jpg / jpeg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        if model is not None:
            with st.spinner("Classifying..."):
                label, confidence, all_probs = predict(model, image)

            emoji = CLASS_EMOJI.get(label, "🌸")
            st.markdown(f"### {emoji} Prediction: **{label.capitalize()}**")
            st.metric("Confidence", f"{confidence * 100:.1f}%")

            st.markdown("#### All class probabilities")
            for cls, prob in sorted(zip(CLASSES, all_probs), key=lambda x: -x[1]):
                bar_emoji = CLASS_EMOJI.get(cls, "🌸")
                st.progress(float(prob), text=f"{bar_emoji} {cls.capitalize()}: {prob*100:.1f}%")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: ResNet50 fine-tuned on Oxford 102 Flowers subset (5 classes)")