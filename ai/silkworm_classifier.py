"""Real-photo-trained classifier: frozen MobileNetV3-Small embeddings ->
an SVM (RBF), trained on the Kaggle `tatinanikhitha/silkworm` dataset
(4,862 real photos of silkworm larvae - 2,379 healthy, 2,286 generically
"infected", 197 confirmed Grasserie). Covers 3 conditions: healthy,
Grasserie (a specifically named, confirmed viral disease), and a generic
"infected" catch-all for worms that look unwell but aren't individually
diagnosed by this dataset (could be Flacherie, Muscardine, Pebrine, or
another cause - see data/silkworm_knowledge_base.json's
"infected_silkworm" entry, which is written to steer the farmer to Submit
to Expert rather than naming a specific disease this model can't actually
distinguish).

Unlike ai/mobilenet_classifier.py (mulberry), the winner of the 11-way
comparison here was NOT the fine-tuned end-to-end model - it was frozen
MobileNetV3-Small (ImageNet-pretrained) embeddings feeding a classical
SVM (RBF), at 97.9% test accuracy vs the fine-tuned model's 97.8% (see
training/silkworm_results/results.json for the full comparison). The gap
is within noise on a 1,215-image test set, but since the training script's
own tie-break picks strictly by test accuracy (matching how the mulberry
model was picked), this module serves that actual winner rather than
defaulting to the fine-tuned architecture for convenience.

See training/train_silkworm_compare.py for the training/comparison script.
There is no synthetic fallback here - see that script's docstring for why.
app_pages/scan_silkworm.py shows "inconclusive" below the confidence
threshold rather than inventing a heuristic classification.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "data" / "silkworm_classifier.joblib"
METADATA_PATH = ROOT / "data" / "silkworm_training_results.json"

# Maps this model's training class names to the app-wide label_key
# convention used in data/silkworm_knowledge_base.json.
CLASS_TO_LABEL_KEY = {
    "Healthy": "healthy_silkworm",
    "Grasserie": "grasserie",
    "Infected": "infected_silkworm",
}

# Below this, the model isn't confident enough to act on - show "inconclusive"
# rather than a possibly-wrong diagnosis.
CONFIDENCE_THRESHOLD = 0.5

RISK_BY_LABEL = {"healthy_silkworm": "low", "grasserie": "severe", "infected_silkworm": "high"}

AI_ADVISORY_DISCLAIMER = (
    "AI result is advisory only. Grasserie and other silkworm diseases spread fast - "
    "isolate any suspect larvae immediately and confirm with a sericulture expert."
)


@dataclass
class SilkwormResult:
    label_key: str
    confidence: float
    risk_level: str
    is_inconclusive: bool
    top_alternatives: list[tuple[str, float]] = field(default_factory=list)
    model_version: str = "mobilenet-embed-svm-silkworm-1.0"
    model_accuracy: float | None = None


@st.cache_resource(show_spinner=False)
def _load_model():
    import joblib

    bundle = joblib.load(WEIGHTS_PATH)
    return bundle["model"], bundle["scaler"], bundle["class_names"]


@st.cache_resource(show_spinner=False)
def _load_embedder():
    import torch
    import torchvision.models as tvm

    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = tvm.mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()
    return model


def is_available() -> bool:
    return WEIGHTS_PATH.exists()


@st.cache_resource(show_spinner=False)
def _load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def model_metadata() -> dict:
    return _load_metadata()


def _best_result_entry(meta: dict) -> dict:
    results = meta.get("results") or []
    return results[0] if results else {}


def predict(image: Image.Image) -> dict[str, float]:
    """Returns {label_key: probability} for the 3 trained classes."""
    import torch
    import torchvision.transforms as T

    embedder = _load_embedder()
    clf, scaler, class_names = _load_model()

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        embedding = embedder(x).numpy()

    embedding_scaled = scaler.transform(embedding)
    probs = clf.predict_proba(embedding_scaled)[0]

    return {CLASS_TO_LABEL_KEY[cls]: float(p) for cls, p in zip(class_names, probs)}


def diagnose(image: Image.Image) -> SilkwormResult:
    scores = predict(image)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_key, top_conf = ranked[0]
    inconclusive = top_conf < CONFIDENCE_THRESHOLD
    risk = "low" if inconclusive else RISK_BY_LABEL.get(top_key, "medium")
    meta = model_metadata()

    return SilkwormResult(
        label_key=top_key,
        confidence=float(top_conf),
        risk_level=risk,
        is_inconclusive=inconclusive,
        top_alternatives=[(k, float(p)) for k, p in ranked[1:]],
        model_accuracy=_best_result_entry(meta).get("accuracy"),
    )
