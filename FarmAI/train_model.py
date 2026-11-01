"""Train a plant image classifier using transfer learning (MobileNetV2)."""
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parent
TRAIN_DIR = BASE_DIR / "plant_dataset" / "train"
TEST_DIR = BASE_DIR / "plant_dataset" / "test"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42


def main() -> None:
    if not TRAIN_DIR.exists() or not TEST_DIR.exists():
        raise FileNotFoundError(
            "Dataset split is missing. Put labeled photos in plant_dataset/raw/<class>/, "
            "then run `python prepare_dataset.py`."
        )

    print("Loading training and validation images...")
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.20,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    if len(class_names) < 2:
        raise ValueError("You need at least two plant classes.")

    print("Loading independent test images...")
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )
    if test_ds.class_names != class_names:
        raise ValueError(
            "Class folders in train and test must match. Re-run prepare_dataset.py."
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    print("Classes:", class_names)

    # Augmentation is applied only during training. Input image pixels are 0..255.
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
        ],
        name="data_augmentation",
    )

    # Pretrained MobileNetV2 expects pixels scaled to [-1, 1].
    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_normalization")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(class_names), activation="softmax", name="plant_class")(x)
    model = keras.Model(inputs, outputs, name="farmai_plant_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_model_path = MODEL_DIR / "best_model.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    print("Training started. First run downloads MobileNetV2 weights.")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    best_model = keras.models.load_model(best_model_path)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.2%}")
    print("Remember: test accuracy is only meaningful if the test images are truly independent.")

    final_model_path = MODEL_DIR / "plant_model.keras"
    best_model.save(final_model_path)
    labels_path = MODEL_DIR / "class_names.json"
    labels_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Model saved to: {final_model_path}")
    print(f"Labels saved to: {labels_path}")
    print("Next: test with `python predict.py path/to/plant_photo.jpg`.")


if __name__ == "__main__":
    main()
