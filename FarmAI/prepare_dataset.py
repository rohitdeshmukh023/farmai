"""Split labeled images from plant_dataset/raw/<class>/ into train/test folders."""
from pathlib import Path
import random
import shutil

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "plant_dataset" / "raw"
OUTPUT_DIR = BASE_DIR / "plant_dataset"
TEST_FRACTION = 0.20
SEED = 42
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw image folder not found: {RAW_DIR}")

    class_dirs = sorted(
        p for p in RAW_DIR.iterdir()
        if p.is_dir() and any(
            f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
            for f in p.iterdir()
        )
    )
    if len(class_dirs) < 2:
        raise ValueError(
            "Add images for at least 2 classes in plant_dataset/raw/<class-name>/."
        )

    # Re-create only the generated train/test folders.
    for split in ("train", "test"):
        target = OUTPUT_DIR / split
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    overall_counts = {"train": 0, "test": 0}

    for class_dir in class_dirs:
        images = sorted(
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        )
        if len(images) < 5:
            raise ValueError(
                f"Class '{class_dir.name}' has {len(images)} image(s). "
                "Add at least 5 per class (hundreds are recommended for a real model)."
            )

        rng.shuffle(images)
        test_count = max(1, round(len(images) * TEST_FRACTION))
        test_count = min(test_count, len(images) - 1)
        test_images = images[:test_count]
        train_images = images[test_count:]

        for split_name, split_images in (("train", train_images), ("test", test_images)):
            destination = OUTPUT_DIR / split_name / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            for source in split_images:
                shutil.copy2(source, destination / source.name)
            overall_counts[split_name] += len(split_images)

        print(
            f"{class_dir.name}: {len(train_images)} train, "
            f"{len(test_images)} test"
        )

    print("\nDataset split complete.")
    print(f"Total training images: {overall_counts['train']}")
    print(f"Total test images: {overall_counts['test']}")
    print("Next: run `python train_model.py`.")
    print(
        "Note: Keep duplicate/near-duplicate images and photos from the same "
        "plant/session in the same split when possible; random splitting can overstate accuracy."
    )


if __name__ == "__main__":
    main()
