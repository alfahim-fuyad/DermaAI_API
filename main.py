
import os

# Configure TensorFlow before importing it.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import threading
import logging
from io import BytesIO

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, File, HTTPException, UploadFile
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dermachecker")


# --------------------------------------------------
# TENSORFLOW CPU CONFIGURATION
# --------------------------------------------------

try:
    tf.config.set_visible_devices([], "GPU")
except RuntimeError:
    logger.warning("GPU configuration was already initialized.")

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    logger.warning("TensorFlow threading was already initialized.")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="DermaCheckerBD AI API",
    version="1.0.0",
    description="ResNet50 skin lesion image classification API",
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

REPO_ID = "fuyad07/dermacheckerbd-resnet50"
MODEL_FILENAME = "resnet50_finetuned.keras"

IMAGE_SIZE = 224
MAX_IMAGE_SIZE = 10 * 1024 * 1024

# IMPORTANT:
# This order MUST match the original model training order.
CLASS_ORDER = [
    label.strip()
    for label in os.getenv(
        "CLASS_ORDER",
        "akiec,bcc,bkl,df,mel,nv,vasc",
    ).split(",")
]

if len(CLASS_ORDER) != 7 or len(set(CLASS_ORDER)) != 7:
    raise RuntimeError("CLASS_ORDER must contain 7 unique classes.")

_model = None
_model_lock = threading.Lock()


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def get_model():
    """Download and load the model once per process."""

    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        logger.info("STEP 1: Downloading model from Hugging Face.")

        try:
            model_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MODEL_FILENAME,
            )
        except Exception:
            logger.exception("STEP 1 FAILED: Model download failed.")
            raise

        logger.info("STEP 2: Model downloaded successfully.")
        logger.info("Model file path: %s", model_path)

        logger.info("STEP 3: BEFORE MODEL LOAD")

        try:
            model = tf.keras.models.load_model(
                model_path,
                compile=False,
            )
        except Exception:
            logger.exception("STEP 3 FAILED: Keras model loading failed.")
            raise

        logger.info("STEP 4: AFTER MODEL LOAD")

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

        logger.info(
            "STEP 5: Model validated successfully. "
            "Input=%s, Output=%s",
            input_shape,
            output_shape,
        )

        return _model


# --------------------------------------------------
# HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "service": "DermaCheckerBD AI API",
        "status": "running",
        "model_loaded": _model is not None,
        "docs": "/docs",
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "service": "DermaCheckerBD AI API",
        "status": "ok",
        "model_loaded": _model is not None,
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict the class of an uploaded skin lesion image."""

    logger.info("Prediction request received.")

    # Validate content type when supplied.
    if (
        file.content_type
        and not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload an image file.",
        )

    # Read the upload.
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

    # Decode and resize image.
    try:
        with Image.open(BytesIO(image_bytes)) as uploaded_image:
            if uploaded_image.width < 16 or uploaded_image.height < 16:
                raise HTTPException(
                    status_code=400,
                    detail="Image dimensions must be at least 16x16 pixels.",
                )

            image = uploaded_image.convert("RGB")

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

    # Run inference.
    try:
        logger.info("STEP A: Preparing image.")

        pixels = np.asarray(image, dtype=np.float32)

        batch = tf.keras.applications.resnet50.preprocess_input(
            np.expand_dims(pixels, axis=0)
        )

        logger.info("STEP B: Getting model.")

        model = get_model()

        logger.info("STEP C: Running model inference.")

        raw = model(batch, training=False)

        probabilities = np.asarray(raw, dtype=np.float32)

        logger.info("STEP D: Model inference completed.")

        expected_shape = (1, len(CLASS_ORDER))

        if probabilities.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected model output shape: {probabilities.shape}"
            )

        probabilities = probabilities[0]

        if not np.isfinite(probabilities).all():
            raise RuntimeError(
                "Model returned non-finite values."
            )

        # This implementation expects probabilities,
        # not raw logits.
        if np.any(probabilities < 0):
            raise RuntimeError(
                "Model returned negative values; "
                "verify whether the model outputs logits."
            )

        total = float(probabilities.sum())

        if not np.isclose(total, 1.0, atol=0.02):
            raise RuntimeError(
                "Model output does not appear to be a probability "
                f"distribution. Sum={total:.6f}. "
                "Check the final activation and model training code."
            )

        # Small floating-point differences are normalized.
        probabilities = probabilities / total

        results = {
            label: round(float(probability), 6)
            for label, probability in zip(
                CLASS_ORDER,
                probabilities,
            )
        }

        predicted_class = max(results, key=results.get)

        logger.info(
            "Prediction completed. Class=%s, Confidence=%.6f",
            predicted_class,
            results[predicted_class],
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
        logger.exception(
            "Prediction failed due to an internal error."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The AI prediction service is temporarily "
                "unavailable. Please try again later."
            ),
        )

    finally:
        await file.close()
