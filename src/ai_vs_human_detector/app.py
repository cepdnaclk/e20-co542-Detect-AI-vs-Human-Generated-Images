from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "uploads"
MODELS_DIR = PROJECT_ROOT / "models"
TEMPLATES_DIR = PACKAGE_DIR / "templates"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "gif"}
IMG_SIZE: Tuple[int, int] = (224, 224)
MODEL_CANDIDATES = (
    "resnet50_ai_vs_real_final.h5",
    "resnet50_ai_vs_real_best.h5",
    "resnet50.h5",
)


def _resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        candidate_path = MODELS_DIR / candidate
        if candidate_path.exists():
            return candidate_path
    expected = ", ".join(str(MODELS_DIR / name) for name in MODEL_CANDIDATES)
    raise FileNotFoundError(f"Model file not found. Expected one of: {expected}")


MODEL_PATH = _resolve_model_path()
MODEL = load_model(MODEL_PATH)


def create_app() -> Flask:
    """Application factory used by Flask entry points."""

    app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
    app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

    # Ensure the upload directory exists before handling requests.
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    @app.route("/", methods=["GET"])
    def index() -> str:
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        if not _allowed_file(file.filename):
            return jsonify({"error": "Unsupported file type."}), 400

        filename = secure_filename(file.filename)
        upload_dir = Path(app.config["UPLOAD_FOLDER"])
        saved_path = upload_dir / filename
        file.save(saved_path)

        try:
            img_tensor = _prepare_image(saved_path)
            probability = float(MODEL.predict(img_tensor, verbose=0)[0][0])
            label = "AI-generated" if probability > 0.5 else "Real"
        finally:
            # Delete the uploaded file to avoid unbounded disk usage.
            saved_path.unlink(missing_ok=True)

        return jsonify({"label": label, "probability": probability})

    return app


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _prepare_image(filepath: Path) -> np.ndarray:
    img = image.load_img(filepath, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
