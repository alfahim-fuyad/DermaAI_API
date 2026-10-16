
import os
import threading
from functools import lru_cache
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


app = FastAPI(title="DermaCheckerBD AI API", version="1.0.0")

REPO_ID = "fuyad07/dermacheckerbd-resnet50"
MODEL_FILENAME = "resnet50_finetuned.keras"
IMAGE_SIZE = 224

# Must match the class order used when training the model.
CLASS_ORDER = os.getenv(
    "CLASS_ORDER",
    "akiec,bcc,bkl,df,mel,nv,vasc",
).split(",")

_model = None
_model_lock = threading.Lock()


def get_model():
    """Download and load the model once per process."""
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=MODEL_FILENAME,
        )

        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

        input_shape = model.input_shape
        output_shape = model.output_shape

        if isinstance(input_shape, list) or isinstance(output_shape, list):
            raise RuntimeError("Only single-input, single-output models are supported.")

        if tuple(input_shape[1:]) != (IMAGE_SIZE, IMAGE_SIZE, 3):
            raise RuntimeError(f"Unexpected model input shape: {input_shape}")

        if output_shape[-1] != len(CLASS_ORDER):
            raise RuntimeError(f"Unexpected model output shape: {output_shape}")

        _model = model
        return _model


@app.get("/")
def home():
    return {
        "service": "DermaCheckerBD AI API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image.")

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Maximum image size is 10 MB.")

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        if image.width < 16 or image.height < 16:
            raise HTTPException(
                status_code=400,
                detail="Image dimensions must be at least 16x16 pixels.",
            )

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    try:
        pixels = np.asarray(image, dtype=np.float32)
        batch = tf.keras.applications.resnet50.preprocess_input(
            np.expand_dims(pixels, axis=0)
        )

        model = get_model()
        raw = model(batch, training=False)
        probabilities = np.asarray(raw, dtype=np.float32)

        if probabilities.shape != (1, len(CLASS_ORDER)):
            raise RuntimeError("Model returned an unexpected output shape.")

        probabilities = probabilities[0]

        if (
            not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
        ):
            raise RuntimeError("Model returned invalid probabilities.")

        total = float(probabilities.sum())

        if total <= 0:
            raise RuntimeError("Model returned an invalid probability total.")

        probabilities = probabilities / total

        results = {
            label: round(float(prob), 6)
            for label, prob in zip(CLASS_ORDER, probabilities)
        }

        predicted_class = max(results, key=results.get)

        return {
            "success": True,
            "backend": "keras",
            "predicted_class": predicted_class,
            "confidence": results[predicted_class],
            "probabilities": results,
            "input_size": IMAGE_SIZE,
        }

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="The AI prediction service is temporarily unavailable.",
        )
