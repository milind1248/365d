"""Trains the mulberry leaf-condition classifier used by ai/classifier.py.

Honest disclosure: there is no labeled mulberry photo dataset bundled with
this project (same situation the Flutter app was in - see its
MockModelService docstring). Rather than faking predictions with a random
number generator like the mobile app's mock model does, this script trains a
*real* scikit-learn RandomForestClassifier on feature distributions defined
from the agronomic symptom descriptions in data/diseases_knowledge_base.json
(e.g. "white powdery coating" -> high white_gray_ratio, "orange-brown
pustules" -> high orange_ratio). Each class is a Gaussian cloud around a
domain-defined centroid in the same 9-dimensional feature space that
ai/features.py extracts from real uploaded photos, so the model performs a
genuine trained classification (fit/predict_proba, held-out accuracy) rather
than returning a hand-coded label.

Swap-in path for a real trained model: once you have labeled mulberry leaf
photos, replace `_synthetic_dataset()` with a loader that runs
`ai.features.extract_features()` over the labeled images and re-run this
script - `ai/classifier.py` does not need to change.

Run: python ai/train_model.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ai.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = ROOT / "data" / "labels.txt"
MODEL_PATH = ROOT / "data" / "mulberry_disease_model.joblib"
METADATA_PATH = ROOT / "data" / "model_metadata.json"

# feature order: green, yellow, brown, white_gray, orange, black_dark, saturation, texture_std, edge_density
_BASE = dict(zip(FEATURE_NAMES, [0.85, 0.03, 0.02, 0.02, 0.01, 0.02, 0.55, 0.08, 0.03]))

# Overrides per label, derived from the symptom text in diseases_knowledge_base.json.
_CENTROID_OVERRIDES: dict[str, dict[str, float]] = {
    "healthy_leaf": {},
    "powdery_mildew": {"white_gray_ratio": 0.35, "green_ratio": 0.45, "mean_saturation": 0.35, "texture_std": 0.12},
    "leaf_spot": {"brown_ratio": 0.20, "yellow_ratio": 0.08, "green_ratio": 0.55, "edge_density": 0.10, "texture_std": 0.14},
    "leaf_rust": {"orange_ratio": 0.22, "green_ratio": 0.50, "texture_std": 0.11, "edge_density": 0.09},
    "bacterial_leaf_spot": {"brown_ratio": 0.18, "black_dark_ratio": 0.12, "green_ratio": 0.45, "edge_density": 0.14, "texture_std": 0.15},
    "cercospora_leaf_spot": {"white_gray_ratio": 0.12, "brown_ratio": 0.18, "green_ratio": 0.50, "edge_density": 0.11},
    "myrothecium_leaf_spot": {"brown_ratio": 0.22, "black_dark_ratio": 0.10, "green_ratio": 0.45, "edge_density": 0.12, "texture_std": 0.14},
    "leaf_blight": {"brown_ratio": 0.35, "black_dark_ratio": 0.08, "green_ratio": 0.30, "edge_density": 0.10, "texture_std": 0.13},
    "root_rot": {"brown_ratio": 0.28, "yellow_ratio": 0.15, "green_ratio": 0.35, "texture_std": 0.10},
    "stem_canker": {"brown_ratio": 0.30, "black_dark_ratio": 0.15, "green_ratio": 0.30, "edge_density": 0.13},
    "root_knot_nematode": {"yellow_ratio": 0.25, "green_ratio": 0.45, "brown_ratio": 0.08, "texture_std": 0.08},
    "mealybug": {"white_gray_ratio": 0.28, "green_ratio": 0.50, "edge_density": 0.15, "texture_std": 0.16},
    "whitefly": {"white_gray_ratio": 0.15, "black_dark_ratio": 0.10, "green_ratio": 0.55, "edge_density": 0.12},
    "thrips": {"white_gray_ratio": 0.18, "green_ratio": 0.50, "texture_std": 0.18, "edge_density": 0.14},
    "scale_insects": {"brown_ratio": 0.18, "green_ratio": 0.55, "edge_density": 0.10, "texture_std": 0.10},
    "mites": {"yellow_ratio": 0.15, "brown_ratio": 0.10, "green_ratio": 0.50, "texture_std": 0.20, "edge_density": 0.09},
    "leaf_roller_caterpillar": {"green_ratio": 0.45, "edge_density": 0.22, "brown_ratio": 0.10, "texture_std": 0.15},
    "nitrogen_deficiency": {"yellow_ratio": 0.35, "green_ratio": 0.40, "texture_std": 0.05, "edge_density": 0.02},
    "potassium_deficiency": {"brown_ratio": 0.20, "yellow_ratio": 0.18, "green_ratio": 0.45, "edge_density": 0.08, "texture_std": 0.09},
    "magnesium_deficiency": {"yellow_ratio": 0.30, "green_ratio": 0.42, "texture_std": 0.16, "edge_density": 0.08},
    "iron_deficiency": {"yellow_ratio": 0.40, "white_gray_ratio": 0.10, "green_ratio": 0.30, "texture_std": 0.06, "edge_density": 0.02},
    "zinc_deficiency": {"yellow_ratio": 0.22, "green_ratio": 0.48, "texture_std": 0.14, "edge_density": 0.07},
    "water_stress": {"brown_ratio": 0.18, "green_ratio": 0.50, "mean_saturation": 0.35, "texture_std": 0.09},
    "sun_scorch": {"white_gray_ratio": 0.20, "brown_ratio": 0.22, "green_ratio": 0.35, "edge_density": 0.10},
    "chemical_spray_burn": {"brown_ratio": 0.28, "yellow_ratio": 0.15, "black_dark_ratio": 0.05, "green_ratio": 0.35, "edge_density": 0.16, "texture_std": 0.15},
}


def _load_label_keys() -> list[str]:
    keys = []
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        keys.append(line.split("|")[0])
    return keys


def _centroid_vector(label_key: str) -> np.ndarray:
    values = dict(_BASE)
    values.update(_CENTROID_OVERRIDES.get(label_key, {}))
    return np.array([values[name] for name in FEATURE_NAMES], dtype=np.float32)


def _synthetic_dataset(samples_per_class: int = 180, noise_std: float = 0.045, seed: int = 42):
    rng = np.random.default_rng(seed)
    label_keys = _load_label_keys()
    X, y = [], []
    for key in label_keys:
        centroid = _centroid_vector(key)
        samples = rng.normal(loc=centroid, scale=noise_std, size=(samples_per_class, len(FEATURE_NAMES)))
        samples = np.clip(samples, 0.0, 1.0)
        X.append(samples)
        y.extend([key] * samples_per_class)
    return np.vstack(X), np.array(y), label_keys


def train() -> dict:
    X, y, label_keys = _synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    val_accuracy = accuracy_score(y_test, model.predict(X_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "version": "mulberry-rf-1.0",
        "algorithm": "RandomForestClassifier (scikit-learn)",
        "class_count": len(label_keys),
        "classes": label_keys,
        "feature_names": FEATURE_NAMES,
        "validation_accuracy": round(float(val_accuracy), 4),
        "trained_on": "Synthetic feature distributions derived from documented disease/pest/deficiency "
                       "symptom descriptions (data/diseases_knowledge_base.json), not photographed leaves.",
        "is_photo_trained": False,
        "notes": "This is a real, trained scikit-learn model operating on real pixel-derived features "
                 "extracted from the uploaded photo (ai/features.py) - it is not a random mock. "
                 "Its accuracy is measured on held-out synthetic data, so treat predictions as advisory "
                 "pattern-matching, not clinical diagnosis; retrain on real labeled photos for production "
                 "-grade accuracy by replacing _synthetic_dataset() in this script.",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = train()
    print(f"Trained RandomForestClassifier on {meta['class_count']} classes.")
    print(f"Held-out synthetic validation accuracy: {meta['validation_accuracy'] * 100:.1f}%")
    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metadata -> {METADATA_PATH}")
