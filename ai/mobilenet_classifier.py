"""Real-photo-trained classifier: MobileNetV3-Small, fine-tuned on the Kaggle
`nahiduzzaman13/mulberry-leaf-dataset` (1,091 real DSLR photos of mulberry
leaves, CC0). Covers 3 of the app's 25 tracked conditions - healthy leaf,
leaf rust, leaf spot - the only ones with a real labeled photo dataset
available. See training/train_compare.py for the training/comparison
script and data/mobilenet_training_results.json for the full benchmark
(11 model/feature combinations compared; this one won at 96.7% test accuracy,
beating both frozen-embedding classical ML and hand-crafted-feature models).

ai/classifier.py calls into this module and prefers its output whenever the
broader synthetic model's top guess falls into one of these 3 classes -
see diagnose() there for the blending logic.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "data" / "mobilenet_leaf_classifier.pt"

# Maps this model's training folder names to the app-wide label_key convention
# used in data/labels.txt and data/diseases_knowledge_base.json.
CLASS_TO_LABEL_KEY = {
    "Disease Free leaves": "healthy_leaf",
    "Leaf Rust": "leaf_rust",
    "Leaf spot": "leaf_spot",
}

COVERED_LABEL_KEYS = set(CLASS_TO_LABEL_KEY.values())


@st.cache_resource(show_spinner=False)
def _load_model():
    import torch
    import torch.nn as nn
    import torchvision.models as tvm

    checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    class_names = checkpoint["class_names"]

    model = tvm.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names


def is_available() -> bool:
    return WEIGHTS_PATH.exists()


def predict(image: Image.Image) -> dict[str, float]:
    """Returns {label_key: probability} for the 3 real-photo-trained classes."""
    import torch
    import torchvision.transforms as T

    model, class_names = _load_model()

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].tolist()

    return {CLASS_TO_LABEL_KEY[cls]: p for cls, p in zip(class_names, probs)}
