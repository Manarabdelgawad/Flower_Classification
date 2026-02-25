import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from Flower_Classification.fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io

# ─────────────────────────── Config ───────────────────────────
MODEL_PATH = "best_model.h5"
IMG_SIZE   = (224, 224)

# Class names — must match the order used during training.
# If you saved class_indices from training, load them here instead.
CLASS_NAMES = [
    "daisy", "dandelion", "rose", "sunflower", "tulip"   # ← عدّل حسب dataset-ك
]

# ─────────────────────────── Load model ───────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# ─────────────────────────── App ───────────────────────────
app = FastAPI(
    title="Flower Classification API",
    description="Upload a flower image and get the predicted class.",
    version="1.0.0"
)


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Convert raw bytes → model-ready tensor."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = keras_image.img_to_array(img) / 255.0      # rescale same as training
    arr = np.expand_dims(arr, axis=0)                 # (1, 224, 224, 3)
    return arr


@app.get("/")
def root():
    return {"message": "Flower Classification API is running 🌸"}


@app.get("/classes")
def get_classes():
    """Return all available classes."""
    return {"classes": CLASS_NAMES}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file (jpg / png) and receive:
    - predicted class
    - confidence score
    - full probability distribution
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )

    img_bytes = await file.read()

    try:
        arr = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    preds = model.predict(arr)[0]                     # shape: (num_classes,)

    predicted_index = int(np.argmax(preds))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence      = float(preds[predicted_index])

    all_scores = {CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(len(CLASS_NAMES))}

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence":      round(confidence, 4),
        "all_scores":      all_scores,
        "filename":        file.filename
    })


# ─────────────────────────── Run ───────────────────────────
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
