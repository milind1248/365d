"""Real image-based feature extraction for mulberry leaf photos.

No stub/random values: every feature below is computed from the actual
pixels of the uploaded photo (resized + converted to HSV/RGB with Pillow and
analyzed with numpy). These features feed both the trained classifier
(ai/classifier.py) and the severity estimator (ai/severity.py).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "green_ratio",
    "yellow_ratio",
    "brown_ratio",
    "white_gray_ratio",
    "orange_ratio",
    "black_dark_ratio",
    "mean_saturation",
    "texture_std",
    "edge_density",
]


def _prepare_arrays(image: Image.Image, size: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb float array HxWx3 in 0-255, hsv float array HxWx3) for a
    downsized, background-cropped copy of the image."""
    img = image.convert("RGB").resize((size, size))
    rgb = np.asarray(img, dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32)
    return rgb, hsv


def _leaf_mask(rgb: np.ndarray) -> np.ndarray:
    """Excludes near-black/near-white background (shadow, paper, sky), same
    rule as the Flutter SeverityEstimator so severity stays consistent."""
    brightness = rgb.mean(axis=2)
    near_white = (rgb[:, :, 0] > 235) & (rgb[:, :, 1] > 235) & (rgb[:, :, 2] > 235)
    background = (brightness < 25) | near_white
    return ~background


def extract_features(image: Image.Image) -> dict[str, float]:
    rgb, hsv = _prepare_arrays(image)
    mask = _leaf_mask(rgb)
    total = max(int(mask.sum()), 1)

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Pillow HSV channels are 0-255; convert hue to degrees for readability.
    hue_deg = h.astype(np.float32) * (360.0 / 255.0)

    def ratio(cond: np.ndarray) -> float:
        return float((cond & mask).sum()) / total

    greenish = (g > r) & (g > b * 0.9)
    yellowish = (hue_deg >= 40) & (hue_deg <= 70) & (s > 60)
    orangeish = (hue_deg >= 15) & (hue_deg < 40) & (s > 80) & (v > 90)
    brownish = (r > g) & (g >= b) & (v < 170) & (v > 40) & (s > 40)
    whitegray = (s < 45) & (v > 150)
    blackdark = v < 60

    grayscale = rgb.mean(axis=2)
    texture_std = float(grayscale[mask].std()) if mask.any() else 0.0

    gx = np.abs(np.diff(grayscale, axis=1))
    gy = np.abs(np.diff(grayscale, axis=0))
    edge_density = float((gx.mean() + gy.mean()) / 2.0)

    return {
        "green_ratio": ratio(greenish),
        "yellow_ratio": ratio(yellowish),
        "brown_ratio": ratio(brownish),
        "white_gray_ratio": ratio(whitegray),
        "orange_ratio": ratio(orangeish),
        "black_dark_ratio": ratio(blackdark),
        "mean_saturation": float(s[mask].mean()) / 255.0 if mask.any() else 0.0,
        "texture_std": texture_std / 255.0,
        "edge_density": edge_density / 255.0,
    }


def feature_vector(image: Image.Image) -> np.ndarray:
    feats = extract_features(image)
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
