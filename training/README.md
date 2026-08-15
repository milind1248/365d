# Model training & comparison

Scripts used to train the real-photo classifier the app now uses for
healthy / leaf rust / leaf spot detection (`ai/mobilenet_classifier.py`),
and to benchmark it against 10 other model/feature combinations.

## Reproducing

1. Download the dataset zip (needs a free Kaggle account + API token from
   [kaggle.com/settings](https://www.kaggle.com/settings)):
   ```bash
   curl -L -H "Authorization: Bearer <your-kaggle-token>" \
     -o training/mulberry-leaf-dataset.zip \
     "https://www.kaggle.com/api/v1/datasets/download/nahiduzzaman13/mulberry-leaf-dataset"
   ```
2. Extract + sanity-check the class folders:
   ```bash
   python training/extract_dataset.py
   ```
3. Run the full comparison (loads ~1,091 photos, extracts hand-crafted and
   MobileNet features, trains 5 classical algorithms on each, fine-tunes
   MobileNetV3-Small end-to-end, saves confusion matrices + `results.json`):
   ```bash
   python -m training.train_compare
   ```
   Takes roughly 15-20 minutes on CPU, dominated by image I/O (the source
   photos are 4000x6000).
4. To pick up a newly trained model in the app, copy the outputs into `data/`:
   ```bash
   cp training/results/mobilenet_finetuned.pt data/mobilenet_leaf_classifier.pt
   cp training/results/results.json data/mobilenet_training_results.json
   ```

## Dataset

[Mulberry Leaf Dataset](https://www.kaggle.com/datasets/nahiduzzaman13/mulberry-leaf-dataset)
(Nahiduzzaman et al., CC0) — 1,091 real DSLR photos, 3 classes (440 healthy,
489 leaf rust, 162 leaf spot). Cite:

> Nahiduzzaman, M., Chowdhury, M.E.H., Salam, A., Nahid, E., Ahmed, F., Al-Emadi, N.,
> Ayari, M.A., Khandakar, A., Haider, J. (2023). "Explainable Deep Learning Model for
> Automatic Mulberry Leaf Disease Classification." *Frontiers in Plant Science* 14: 1175515.

Only covers 3 of the app's 25 tracked conditions - the rest still run on the
synthetic-feature model in `ai/classifier.py` / `ai/train_model.py` until a
labeled photo set exists for them too.

## Result summary

MobileNetV3-Small, fine-tuned end-to-end, won at **96.7% test accuracy**
(0.953 macro-F1), ahead of the same architecture used as a frozen feature
extractor + SVM/LogisticRegression (96.3%), and well ahead of hand-crafted
color/texture features + any classical algorithm (best: 90.5%, Random Forest).
Full ranked table and per-model confusion matrices in `results/`.

## Silkworm disease classifier

Same methodology, separate script (`training/train_silkworm_compare.py`),
for silkworm larva health (`ai/silkworm_classifier.py`).

1. Download + extract the dataset (Roboflow export layout) to
   `training/silkworm_raw/` (needs a free Kaggle account + API token):
   ```bash
   curl -L -H "Authorization: Bearer <your-kaggle-token>" \
     -o training/silkworm-dataset.zip \
     "https://www.kaggle.com/api/v1/datasets/download/tatinanikhitha/silkworm"
   ```
   then unzip into `training/silkworm_raw/` so `train/images/`,
   `valid/images/`, `test/images/` exist directly under it.
2. Run the comparison:
   ```bash
   python -m training.train_silkworm_compare
   ```
3. Copy the outputs into `data/`:
   ```bash
   cp training/silkworm_results/best_classical_model.joblib data/silkworm_classifier.joblib
   cp training/silkworm_results/results.json data/silkworm_training_results.json
   ```

**Dataset:** [`tatinanikhitha/silkworm`](https://www.kaggle.com/datasets/tatinanikhitha/silkworm)
— 4,862 real photos, 3 classes (2,379 healthy, 2,286 generic "infected", 197
confirmed Grasserie). Only Grasserie is individually diagnosed; "infected" is
a catch-all for other unwell-looking larvae (could be Flacherie, Muscardine,
Pébrine, etc.) — the app routes those to "Submit to Expert" rather than
guessing a specific disease.

**Result summary:** unlike the mulberry model, the fine-tuned network did
**not** win here. Frozen MobileNetV3-Small (ImageNet) embeddings + SVM (RBF)
scored highest at **97.9% test accuracy** (0.908 macro-F1), narrowly ahead of
the same embeddings + LogisticRegression (97.8% acc, 0.914 macro-F1) and the
fully fine-tuned model (97.8% acc, 0.898 macro-F1) — all three are within
noise of each other on a 1,215-photo test set. The SVM was kept as the
served model since it's what the script's own strict-accuracy tie-break rule
picked (matching the mulberry model's selection rule), and the training
script auto-saves it to `best_classical_model.joblib`. Hand-crafted
color/texture features (tuned for green mulberry leaves, not silkworm body
color) performed worst, as expected — best case 96.5%, Random Forest.
