"""Run a prediction for one image using the trained FarmAI model."""
from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "plant_model.keras"
LABELS_PATH = BASE_DIR / "models" / "class_names.json"
IMAGE_SIZE = (224, 224)


def predict(image_path: Path) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError("Train the model first using `python train_model.py`.")

    model = keras.models.load_model(MODEL_PATH)
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB").resize(IMAGE_SIZE)
        array = np.asarray(image, dtype=np.float32)[None, ...]

    # Normalization is included inside the saved model, matching training.
    scores = model.predict(array, verbose=0)[0]
    if len(scores) != len(labels):
        raise ValueError("Model output and label list do not match.")

    top_indices = np.argsort(scores)[::-1][: min(3, len(scores))]
    print("\nFarmAI prediction (scores are not guarantees of correctness)")
    print("-" * 55)
    for index in top_indices:
        print(f"{labels[int(index)]:20s} {float(scores[index]) * 100:6.2f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "path/to/plant_photo.jpg"')
        raise SystemExit(2)
    predict(Path(sys.argv[1]))
