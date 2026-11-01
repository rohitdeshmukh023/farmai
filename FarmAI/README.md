# FarmAI — AI Plant Identification starter project

This Python starter project trains a plant-category image classifier using TensorFlow/Keras and MobileNetV2, tests a single image, and exposes the model in a Streamlit app with camera capture and image upload.

## Important limitations

- The starter model only predicts the classes represented in your labeled training folders. It is not a universal plant identifier.
- It does not detect plant diseases.
- Softmax scores are not guarantees of correctness. The app's low-score warning is only a basic warning, not a reliable unknown-plant detector.
- Do not use AI predictions alone for pesticide, fertilizer, or other consequential crop decisions.
- No plant photos or trained model are bundled in this source archive. You must provide a suitable labeled dataset and train the model.

## 1. Prerequisites

Recommended for a beginner: Windows, Python 3.11 or 3.12 compatible with the TensorFlow release you install, and VS Code. First confirm Python is installed by running `python --version`.

## 2. Install dependencies

In a terminal inside this folder:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If using Windows Command Prompt, activate with `venv\Scripts\activate.bat`. On macOS/Linux, use `source venv/bin/activate`.

If TensorFlow installation fails, consult https://www.tensorflow.org/install and use a supported Python/OS combination.

## 3. Add labeled images

Copy real image files into these raw class folders. Keep each image in the correct class folder:

```text
plant_dataset/raw/cotton/
plant_dataset/raw/maize/
plant_dataset/raw/rice/
plant_dataset/raw/soybean/
plant_dataset/raw/tomato/
```

Use JPG, JPEG, PNG, BMP, or WEBP images. The folder name is the class label. You can add or remove classes, but use at least two classes and ensure each has at least five images for the script to run. For meaningful performance, aim for hundreds of varied, correctly labeled images per class and collect more data as evaluation indicates. Make sure you have permission to use the images and check the dataset license.

Dataset starting points and learning resources:

- TensorFlow image classification tutorial: https://www.tensorflow.org/tutorials/images/classification
- PlantVillage dataset record: https://data.mendeley.com/datasets/tywbtsjrjv/1

Note: PlantVillage is principally a plant health/disease image dataset. Inspect its labels carefully; it is not automatically a pure species-identification dataset. Do not place images into a crop-species folder unless they truly show that species. Include natural field photos for the intended real-world use.

## 4. Split the raw images

```powershell
python prepare_dataset.py
```

This creates `plant_dataset/train/` and `plant_dataset/test/` using an 80/20 random split. It overwrites those generated folders each time it is run. Avoid duplicates and near-duplicates across train/test. If you have multiple photos of the same plant or photo session, keep them in one split; for serious evaluation, split by plant, farm, or photo session rather than randomly.

## 5. Train the model

```powershell
python train_model.py
```

The first run downloads MobileNetV2 pretrained weights, so you need internet access. Training can take several minutes or longer depending on hardware and dataset size. It creates:

```text
models/plant_model.keras
models/best_model.keras
models/class_names.json
```

Check the printed test accuracy, but do not treat it as proof that the model works on every real-world farm image.

## 6. Test one image directly

```powershell
python predict.py "C:\path\to\your\plant_photo.jpg"
```

Replace the example path with the full path of a real image. This prints the top three predicted categories.

## 7. Run the app

```powershell
streamlit run app.py
```

Open the local URL printed in the terminal, usually http://localhost:8501. Use Upload image or Use camera. Allow camera permission in your browser. For mobile use or deployment, serve the app over HTTPS for browser camera access.

## 8. Add a new plant class

1. Create `plant_dataset/raw/new_class/` and add correctly labeled images.
2. Re-run `python prepare_dataset.py`.
3. Re-run `python train_model.py`.
4. Add a corresponding record to `PLANT_INFO` in `plant_info.py` if you want a description on the result screen.

## Files

- `prepare_dataset.py` — splits labeled raw images into training and test folders.
- `train_model.py` — trains and evaluates the classifier.
- `predict.py` — predicts categories for one image.
- `app.py` — Streamlit camera/upload app.
- `plant_info.py` — example plant information displayed by the app.
