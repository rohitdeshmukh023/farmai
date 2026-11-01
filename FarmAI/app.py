"""Streamlit web app for the FarmAI plant identifier."""
from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from plant_info import PLANT_INFO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_model.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (224, 224)

# Reject predictions below this score.
MIN_CONFIDENCE = 0.70

# Reject uncertain predictions when the top two scores are too similar.
MIN_CONFIDENCE_GAP = 0.15

st.set_page_config(
    page_title="FarmAI | Plant Scanner",
    page_icon="🌱",
    layout="centered",
)


@st.cache_resource
def load_model_and_labels():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Model files are missing. Complete the training steps first: "
            "python prepare_dataset.py, then python train_model.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    if model.output_shape[-1] != len(labels):
        raise ValueError("The saved model and class labels do not match.")

    return model, labels


def identify_plant(image: Image.Image, model, labels):
    image = (
        ImageOps.exif_transpose(image)
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    pixels = np.asarray(image, dtype=np.float32)[None, ...]

    # Normalization is built into the model.
    scores = model.predict(pixels, verbose=0)[0]

    index = int(np.argmax(scores))
    confidence = float(scores[index])

    # Find the highest and second-highest prediction scores.
    sorted_indices = np.argsort(scores)[::-1]
    second_confidence = (
        float(scores[sorted_indices[1]])
        if len(sorted_indices) > 1
        else 0.0
    )

    confidence_gap = confidence - second_confidence

    # Reject low-confidence or ambiguous predictions.
    if (
        confidence < MIN_CONFIDENCE
        or confidence_gap < MIN_CONFIDENCE_GAP
    ):
        return {
            "valid": False,
            "confidence": confidence,
            "reason": (
                "The image does not match a supported plant category "
                "confidently enough."
            ),
        }

    return {
        "valid": True,
        "name": labels[index],
        "confidence": confidence,
        "top": [
            {
                "name": labels[int(i)],
                "score": float(scores[int(i)]),
            }
            for i in sorted_indices[:min(3, len(scores))]
        ],
    }


st.title("🌱 FarmAI — AI Plant Scanner")

st.write(
    "Upload a plant photo or use your camera to identify a supported "
    "plant category."
)

st.info(
    "For best results, photograph one plant clearly in good lighting. "
    "This prototype identifies crop categories; it does not diagnose disease."
)

source = st.radio(
    "Choose how to add a photo",
    ["Upload image", "Use camera"],
    horizontal=True,
)

if source == "Upload image":
    chosen_file = st.file_uploader(
        "Upload a plant image",
        type=["jpg", "jpeg", "png", "webp"],
    )
else:
    chosen_file = st.camera_input("Take a photo of the plant")


if chosen_file is not None:
    try:
        with Image.open(chosen_file) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")

        st.image(
            image,
            caption="Selected image",
            use_container_width=True,
        )

        if st.button(
            "🔍 Identify plant",
            type="primary",
            use_container_width=True,
        ):
            # Clear the previous result before processing a new image.
            st.session_state.pop("plant_result", None)

            try:
                model, labels = load_model_and_labels()

                with st.spinner("Checking the image..."):
                    result = identify_plant(image, model, labels)

                st.session_state["plant_result"] = result

            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

            except Exception:
                st.error(
                    "The model could not analyze this image. "
                    "Check the model files and try again."
                )

    except (UnidentifiedImageError, OSError):
        st.error("This file could not be opened as a valid image.")


result = st.session_state.get("plant_result")

if result:
    st.divider()

    if not result["valid"]:
        st.error("❌ Invalid image or uncertain prediction")

        st.write(
            "Please upload a clear image of a supported plant: "
            "cotton, maize, rice, soybean, or tomato."
        )

        st.caption(
            "Try a closer photo with good lighting and one plant clearly visible. "
            "This check is a preliminary filter, not a guaranteed plant detector."
        )

    else:
        st.header("Identification result")

        name = str(result["name"])
        normalized_name = name.strip().lower().replace(" ", "_")
        confidence = float(result["confidence"])

        st.success("✅ Supported plant detected")

        st.subheader(name.replace("_", " ").title())
        st.metric("Model score", f"{confidence:.1%}")

        info = PLANT_INFO.get(normalized_name)

        if info:
            st.write(
                f"**Scientific name:** {info['scientific_name']}"
            )
            st.write(f"**Plant type:** {info['plant_type']}")
            st.write(f"**About this plant:** {info['description']}")
        else:
            st.write(
                "Extra information for this category has not been added yet."
            )

        with st.expander("See the top 3 model predictions"):
            for item in result["top"]:
                st.write(
                    f"{item['name']}: {item['score']:.1%}"
                )

        st.caption(
            "Model scores are not guarantees of correctness. "
            "Verify identification before making important farming decisions."
        )



