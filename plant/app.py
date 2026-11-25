"""Flask backend: loads the trained Keras model and serves /predict."""
import json
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "plant_disease_model.h5")
CLASSES_PATH = os.path.join(BASE, "model", "class_names.json")
INFO_PATH = os.path.join(BASE, "disease_info.json")
IMG_SIZE = 224            # must match the size used in training
ALLOWED = {"png", "jpg", "jpeg", "webp"}
LOW_CONFIDENCE = 0.60

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

model, class_names, load_error = None, [], None
with open(INFO_PATH, encoding="utf-8") as f:
    DISEASE_INFO = json.load(f)


def load_artifacts():
    global model, class_names, load_error
    try:
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH, compile=False)
        with open(CLASSES_PATH, encoding="utf-8") as f:
            class_names = json.load(f)
    except Exception as exc:  # missing files, version mismatch, etc.
        load_error = f"Model could not be loaded: {exc}"
        print(load_error)


def preprocess(stream):
    """Pillow: load, fix rotation, RGB, resize. NumPy: scale to 0-1, add batch axis."""
    img = Image.open(stream)
    img.load()
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def describe(class_name):
    info = DISEASE_INFO.get(class_name)
    if info:
        return info
    plant, _, cond = class_name.partition("___")
    return {
        "plant": plant.replace("_", " ").strip(),
        "disease": cond.replace("_", " ").strip() or class_name,
        "healthy": cond.lower() == "healthy",
        "description": "No detailed notes are stored for this class yet.",
        "prevention_steps": ["Ask a local agricultural extension office to confirm the diagnosis."],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(model_loaded=model is not None, error=load_error)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify(error=load_error or "Model is not loaded."), 503
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="No image was uploaded."), 400
    if file.filename.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return jsonify(error="Unsupported file type. Use JPG, PNG or WebP."), 400
    try:
        x = preprocess(file.stream)
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not a readable image."), 400
    try:
        probs = model.predict(x, verbose=0)[0]
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify(error="Prediction failed. Please try another photo."), 500

    top = np.argsort(probs)[::-1][:3]
    best = describe(class_names[top[0]])
    confidence = float(probs[top[0]])
    return jsonify(
        disease_name=f"{best['plant']}: {best['disease']}",
        confidence=round(confidence * 100, 2),
        description=best["description"],
        prevention_steps=best["prevention_steps"],
        is_healthy=bool(best.get("healthy")),
        low_confidence=confidence < LOW_CONFIDENCE,
        alternatives=[
            {"name": f"{describe(class_names[i])['plant']}: {describe(class_names[i])['disease']}",
             "confidence": round(float(probs[i]) * 100, 2)}
            for i in top[1:]
        ],
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is larger than 8 MB."), 413


load_artifacts()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
