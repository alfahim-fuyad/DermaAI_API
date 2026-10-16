
import os

# Configure TensorFlow before importing it.
# This API will use CPU instead of attempting GPU initialization.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import threading
from io import BytesIO

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


app = FastAPI(
    title="DermaCheckerBD AI API",
    version="1.0.0",
)

REPO_ID = "fuyad07/dermacheckerbd-resnet50"
MODEL_FILENAME = "resnet50_finetuned.keras"
IMAGE_SIZE = 224
MAX_IMAGE_SIZE = 10 * 1024 * 1024

# IMPORTANT:
# This order must exactly match the class-index order
# used when the model was trained.
CLASS_ORDER = os.getenv(
    "CLASS_ORDER",
    "akiec,bcc,bkl,df,mel,nv,vasc",
).split(",")

_model = None
_model_lock = threading.Lock()


def get_model():
    """Download and load the model only once per process."""
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

        if isinstance(input_shape, list):
            raise RuntimeError(
                "Only single-input models are supported."
            )

        if isinstance(output_shape, list):
            raise RuntimeError(
                "Only single-output models are supported."
            )

        if tuple(input_shape[1:]) != (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ):
            raise RuntimeError(
                f"Unexpected model input shape: {input_shape}"
            )

        if output_shape[-1] != len(CLASS_ORDER):
            raise RuntimeError(
                f"Unexpected model output shape: {output_shape}"
            )

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
        "service": "DermaCheckerBD AI API",
        "status": "ok",
        "model_loaded": _model is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded skin-lesion image."""

    # Validate content type when provided by the client.
    if (
        file.content_type
        and not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum image size is 10 MB.",
        )

    # Decode and prepare the image.
    try:
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

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

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file.",
        )

    # Run actual model inference.
    try:
        pixels = np.asarray(
            image,
            dtype=np.float32,
        )

        batch = tf.keras.applications.resnet50.preprocess_input(
            np.expand_dims(pixels, axis=0)
        )

        model = get_model()

        raw = model(
            batch,
            training=False,
        )

        probabilities = np.asarray(
            raw,
            dtype=np.float32,
        )

        if probabilities.shape != (
            1,
            len(CLASS_ORDER),
        ):
            raise RuntimeError(
                "Model returned an unexpected output shape."
            )

        probabilities = probabilities[0]

        # Check that the model output contains valid values.
        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Model returned non-finite values."
            )

        # This endpoint expects a seven-class probability output.
        if np.any(probabilities < 0):
            raise RuntimeError(
                "Model returned negative probability values."
            )

        total = float(probabilities.sum())

        if total <= 0:
            raise RuntimeError(
                "Model returned an invalid probability total."
            )

        # Normalize the output to sum to 1.
        probabilities = probabilities / total

        results = {
            label: round(float(probability), 6)
            for label, probability in zip(
                CLASS_ORDER,
                probabilities,
            )
        }

        predicted_class = max(
            results,
            key=results.get,
        )

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
        # Avoid exposing internal server details to public clients.
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI prediction service is temporarily "
                "unavailable. Please try again later."
            ),
        )
