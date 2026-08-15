"""Port of the Flutter app's SeverityEstimator (services/ai/severity_estimator.dart).

Same explainable heuristic, not a second neural net: downsample the photo and
classify each non-background pixel as healthy-green vs discolored/affected,
using the same HSV/RGB thresholds as the original so severity numbers match
the mobile app for the same photo.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def estimate_affected_area_percent(image: Image.Image, is_healthy_prediction: bool) -> float:
    if is_healthy_prediction:
        return 0.0

    small = image.convert("RGB").resize((96, 96))
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    brightness = (r + g + b) / 3.0

    is_background = (brightness < 25) | ((r > 235) & (g > 235) & (b > 235))
    leaf_mask = ~is_background

    total_leaf_pixels = int(leaf_mask.sum())
    if total_leaf_pixels == 0:
        return 20.0

    is_greenish = (g > r) & (g > b * 0.9)
    affected = leaf_mask & ~is_greenish
    affected_pixels = int(affected.sum())

    percent = (affected_pixels / total_leaf_pixels) * 100.0
    return float(np.clip(percent, 2, 95))
