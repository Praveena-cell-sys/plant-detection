# LeafLens: plant leaf disease detection

Upload a leaf photo, get the disease name, confidence and treatment steps.
Stack: Python, NumPy, Pillow, TensorFlow/Keras (trained on PlantVillage in Colab), Flask, HTML/CSS.

## Structure
```
app.py                  Flask backend (/, /predict, /health)
disease_info.json       Descriptions and prevention steps for all 38 classes
model_training.ipynb    Colab training notebook
model/                  Put plant_disease_model.h5 and class_names.json here
templates/index.html    Frontend page (+ JavaScript)
static/style.css        "Deep Purple Glow" theme
```

## 1. Train the model (Google Colab)
1. Open `model_training.ipynb` in Colab and set Runtime > Change runtime type > GPU.
2. Create a Kaggle API token (Kaggle > Settings > Create New Token) to get `kaggle.json`. Upload it when the notebook asks.
3. Run all cells (about 20-30 minutes). It fine-tunes MobileNetV2 on the PlantVillage dataset.
4. Two files download at the end: `plant_disease_model.h5` and `class_names.json`.
5. Copy both into this project's `model/` folder.

## 2. Run locally (VS Code)
```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000. Use Python 3.10-3.12 for TensorFlow. If loading fails with a version error, install the same TensorFlow version the notebook printed.

## API
`POST /predict` with form field `image` returns:
```json
{"disease_name": "Tomato: Late blight", "confidence": 97.4, "description": "...",
 "prevention_steps": ["..."], "is_healthy": false, "low_confidence": false, "alternatives": []}
```
Errors return `{"error": "..."}` with status 400 (bad file), 413 (over 8 MB), 500 (prediction failed) or 503 (model not loaded).

## Notes
- Preprocessing must match training: RGB, 224x224, pixel values scaled to 0-1. Both `app.py` and the notebook do this.
- PlantVillage photos are lab-style (one leaf, plain background). Accuracy on field photos is lower, so a low-confidence warning is shown under 60%.
- Run `/health` to check whether the model loaded.
