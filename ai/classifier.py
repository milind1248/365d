"""Loads the trained RandomForest model and runs the full scan pipeline:
photo -> real pixel features -> trained-model prediction -> severity -> risk.

Mirrors the Flutter app's ModelService contract (load -> preprocess ->
infer -> postprocess) but backed by a genuinely trained scikit-learn model
instead of MockModelService's random-by-hash stand-in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import streamlit as st
from PIL import Image

from ai import mobilenet_classifier
from ai.features import feature_vector
from ai.severity import estimate_affected_area_percent

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "data" / "mulberry_disease_model.joblib"
METADATA_PATH = ROOT / "data" / "model_metadata.json"
LABELS_PATH = ROOT / "data" / "labels.txt"

CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.50

SEVERITY_LOW_MAX = 15.0
SEVERITY_MEDIUM_MAX = 40.0
SEVERITY_HIGH_MAX = 70.0

AI_ADVISORY_DISCLAIMER = (
    "AI result is advisory only. For severe infection, confirm with an "
    "agriculture/sericulture expert before applying chemicals."
)


@dataclass
class DiseaseResult:
    label_key: str
    name_en: str
    name_mr: str
    name_hi: str
    confidence: float
    severity_percent: float
    risk_level: str
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)
    model_version: str = "mulberry-rf-1.0"
    model_accuracy: float | None = None
    source: str = "synthetic"  # "real_photo_trained" (MobileNetV3) or "synthetic" (RandomForest)

    @property
    def confidence_tier(self) -> str:
        if self.confidence >= CONFIDENCE_HIGH:
            return "likely"
        if self.confidence >= CONFIDENCE_MEDIUM:
            return "possible"
        return "uncertain"

    def localized_name(self, lang: str) -> str:
        return {"mr": self.name_mr, "hi": self.name_hi}.get(lang, self.name_en)


@st.cache_resource(show_spinner=False)
def _load_labels() -> dict[str, tuple[str, str, str]]:
    labels: dict[str, tuple[str, str, str]] = {}
    for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        key = parts[0]
        en = parts[1] if len(parts) > 1 else key
        mr = parts[2] if len(parts) > 2 else en
        hi = parts[3] if len(parts) > 3 else en
        labels[key] = (en, mr, hi)
    return labels


@st.cache_resource(show_spinner=False)
def _load_model():
    if not MODEL_PATH.exists():
        from ai.train_model import train

        train()
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    return model, metadata


def model_metadata() -> dict:
    _, metadata = _load_model()
    return metadata


def _risk_from_severity(severity: float, label_key: str) -> str:
    if label_key == "healthy_leaf":
        return "low"
    if severity <= SEVERITY_LOW_MAX:
        return "low"
    if severity <= SEVERITY_MEDIUM_MAX:
        return "medium"
    if severity <= SEVERITY_HIGH_MAX:
        return "high"
    return "severe"


REAL_MODEL_CONFIDENCE_THRESHOLD = 0.6


def diagnose(image: Image.Image) -> DiseaseResult:
    """Runs the real-photo-trained MobileNet first (see
    ai/mobilenet_classifier.py) - it's the accuracy leader for its 3 classes
    (96.7% test accuracy vs the synthetic model's ~80%, see
    data/mobilenet_training_results.json for the full 11-model comparison).
    Trusts MobileNet's own confidence as the signal for whether the photo is
    one of its 3 classes: real photos of healthy/rust/spot leaves score
    >=98% in practice, so anything below the threshold is treated as "not
    one of these 3" and handed to the synthetic 25-class model instead for
    broader coverage (pests, deficiencies, stress, etc).

    Earlier version gated on the *synthetic* model's top guess instead, which
    misfired often (real photos rarely matched a synthetic-feature centroid
    even when they were textbook cases MobileNet nailed at >99% confidence) -
    don't reintroduce that.
    """
    model, metadata = _load_model()
    labels = _load_labels()

    real_ranked = None
    if mobilenet_classifier.is_available():
        real_scores = mobilenet_classifier.predict(image)
        real_ranked = sorted(real_scores.items(), key=lambda kv: kv[1], reverse=True)

    if real_ranked and real_ranked[0][1] >= REAL_MODEL_CONFIDENCE_THRESHOLD:
        top_key, top_conf = real_ranked[0]
        alternatives = real_ranked[1:]
        source = "real_photo_trained"
        model_version = "mobilenet-v3-small-finetuned-1.0"
        model_accuracy = 0.967
    else:
        x = feature_vector(image).reshape(1, -1)
        proba = model.predict_proba(x)[0]
        classes = model.classes_
        ranked = sorted(zip(classes, proba), key=lambda kv: kv[1], reverse=True)
        top_key, top_conf = ranked[0]
        alternatives = ranked[1:4]
        source = "synthetic"
        model_version = metadata.get("version", "mulberry-rf-1.0")
        model_accuracy = metadata.get("validation_accuracy")

    en, mr, hi = labels.get(top_key, (top_key, top_key, top_key))
    severity = estimate_affected_area_percent(image, is_healthy_prediction=top_key == "healthy_leaf")
    risk = _risk_from_severity(severity, top_key)

    return DiseaseResult(
        label_key=top_key,
        name_en=en,
        name_mr=mr,
        name_hi=hi,
        confidence=float(top_conf),
        severity_percent=round(severity, 1),
        risk_level=risk,
        top_alternatives=[(k, float(p)) for k, p in alternatives],
        model_version=model_version,
        model_accuracy=model_accuracy,
        source=source,
    )
