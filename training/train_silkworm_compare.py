"""Trains & compares multiple algorithms on REAL silkworm photos
(healthy / infected / grasserie), mirroring training/train_compare.py's
methodology exactly - see that file's docstring for the full rationale.

Dataset: kaggle.com/datasets/tatinanikhitha/silkworm (Roboflow export,
YOLO-object-detection layout re-purposed here for whole-image
classification: images live under {train,valid,test}/images/, one class
per image, class encoded as the filename prefix e.g.
"Grasserie-11-_jpg.rf....jpg"). 4,862 total images: 2,379 healthy, 2,286
infected (a general "something's wrong" catch-all - this dataset doesn't
differentiate flacherie/muscardine/pebrine individually), 197 grasserie
(smaller class, still workable at a 75/25 split).

Honesty note carried over from the mulberry model: "Infected" is real
photos of genuinely sick silkworms, but not diagnosed down to a specific
disease - the app should present it as "unwell, cause unconfirmed - submit
to expert" rather than naming a specific disease it doesn't have the data
to actually distinguish. Only "Grasserie" is a named-disease-specific class.

Run from the 365D-Farm-Streamlit project root (after downloading + unzipping
the dataset to training/silkworm_raw/ - see training/README.md):
    python -m training.train_silkworm_compare
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ai.features import extract_features, FEATURE_NAMES  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent
RAW_DIR = CACHE_DIR / "silkworm_raw"
RESULTS_DIR = CACHE_DIR / "silkworm_results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.25
MAX_IMAGE_SIDE = 512
CLASS_NAME_PATTERN = re.compile(r"^([A-Za-z]+)")


def discover_images() -> tuple[list[Path], list[str]]:
    """Roboflow layout: {train,valid,test}/images/{ClassName}-###-....jpg.
    Class comes from the filename prefix, not a folder - pool all splits
    and let the script do its own stratified split below, same as
    train_compare.py does for the mulberry (folder-per-class) dataset."""
    paths, labels = [], []
    for split in ("train", "valid", "test"):
        img_dir = RAW_DIR / split / "images"
        if not img_dir.exists():
            continue
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            m = CLASS_NAME_PATTERN.match(f.name)
            if not m:
                continue
            paths.append(f)
            labels.append(m.group(1))
    return paths, labels


def load_and_resize(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def build_handcrafted_features(images: list[Image.Image]) -> np.ndarray:
    rows = []
    for img in images:
        feats = extract_features(img)
        rows.append([feats[name] for name in FEATURE_NAMES])
    return np.array(rows, dtype=np.float32)


def build_mobilenet_embeddings(images: list[Image.Image]) -> np.ndarray:
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = tvm.mobilenet_v3_small(weights=weights)
    model.classifier = torch.nn.Identity()
    model.eval()

    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.stack([preprocess(img) for img in batch])
            out = model(tensors)
            embeddings.append(out.numpy())
    return np.vstack(embeddings)


def finetune_mobilenet(train_images, y_train_idx, test_images, y_test_idx, num_classes, class_names) -> dict:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Dataset

    train_tf = T.Compose([
        T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class ImgDS(Dataset):
        def __init__(self, images, labels, tf):
            self.images, self.labels, self.tf = images, labels, tf

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.tf(self.images[idx]), self.labels[idx]

    train_ds = ImgDS(train_images, y_train_idx, train_tf)
    test_ds = ImgDS(test_images, y_test_idx, eval_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = tvm.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name or "features.12" in name

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 6
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f"    epoch {epoch + 1}/{epochs} loss={total_loss / len(train_ds):.4f}")

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            out = model(xb)
            preds.extend(out.argmax(dim=1).tolist())
            trues.extend(yb.tolist())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="macro")
    cm = confusion_matrix(trues, preds)
    report = classification_report(trues, preds, target_names=class_names, output_dict=True)
    return {"accuracy": acc, "f1_macro": f1, "confusion_matrix": cm.tolist(), "report": report, "model": model}


def run_classical_suite(X_train, X_test, y_train, y_test, class_names, feature_set_name) -> list[dict]:
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    algorithms = {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=RANDOM_STATE),
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=7),
    }

    results = []
    for name, clf in algorithms.items():
        t0 = time.time()
        clf.fit(X_train_s, y_train)
        preds = clf.predict(X_test_s)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, target_names=class_names, output_dict=True, zero_division=0)
        elapsed = time.time() - t0
        print(f"  [{feature_set_name}] {name}: acc={acc:.3f} f1_macro={f1:.3f} ({elapsed:.1f}s)")
        results.append({
            "feature_set": feature_set_name, "algorithm": name, "accuracy": acc, "f1_macro": f1,
            "confusion_matrix": cm.tolist(), "report": report, "model": clf, "scaler": scaler,
        })
    return results


def plot_confusion_matrix(cm, class_names, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    print("Discovering images...")
    paths, labels = discover_images()
    class_names = sorted(set(labels))
    print(f"Total images: {len(paths)}  Classes: {class_names}")
    from collections import Counter
    print("Class counts:", dict(Counter(labels)))
    if len(class_names) < 2:
        raise SystemExit("Could not find >=2 classes - check training/silkworm_raw/ extraction.")

    print("Loading + resizing images...")
    t0 = time.time()
    images = [load_and_resize(p) for p in paths]
    print(f"  loaded {len(images)} images in {time.time() - t0:.1f}s")

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_idx = np.array([label_to_idx[l] for l in labels])

    idx_all = np.arange(len(images))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_idx
    )
    train_images = [images[i] for i in idx_train]
    test_images = [images[i] for i in idx_test]
    y_train, y_test = y_idx[idx_train], y_idx[idx_test]
    print(f"Train: {len(train_images)}  Test: {len(test_images)}")

    all_results = []

    print("\n=== Hand-crafted features (ai/features.py) ===")
    X_train_hc = build_handcrafted_features(train_images)
    X_test_hc = build_handcrafted_features(test_images)
    all_results += run_classical_suite(X_train_hc, X_test_hc, y_train, y_test, class_names, "hand_crafted")

    print("\n=== MobileNetV3-Small embeddings (frozen, ImageNet pretrained) ===")
    X_train_mn = build_mobilenet_embeddings(train_images)
    X_test_mn = build_mobilenet_embeddings(test_images)
    all_results += run_classical_suite(X_train_mn, X_test_mn, y_train, y_test, class_names, "mobilenet_embed")

    print("\n=== MobileNetV3-Small fine-tuned end-to-end ===")
    ft_result = finetune_mobilenet(train_images, y_train.tolist(), test_images, y_test.tolist(), len(class_names), class_names)
    ft_result["feature_set"] = "mobilenet_finetuned"
    ft_result["algorithm"] = "MobileNetV3-Small (fine-tuned)"
    print(f"  MobileNetV3-Small (fine-tuned): acc={ft_result['accuracy']:.3f} f1_macro={ft_result['f1_macro']:.3f}")
    all_results.append(ft_result)

    summary = []
    for r in all_results:
        label = f"{r['feature_set']} / {r['algorithm']}"
        cm = np.array(r["confusion_matrix"])
        plot_path = RESULTS_DIR / f"cm_{r['feature_set']}_{r['algorithm'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
        plot_confusion_matrix(cm, class_names, label, plot_path)
        summary.append({
            "feature_set": r["feature_set"], "algorithm": r["algorithm"],
            "accuracy": round(r["accuracy"], 4), "f1_macro": round(r["f1_macro"], 4),
            "confusion_matrix": r["confusion_matrix"], "confusion_matrix_png": str(plot_path.name),
            "per_class": {k: v for k, v in r["report"].items() if k in class_names},
        })

    summary.sort(key=lambda r: r["accuracy"], reverse=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps({
        "class_names": class_names,
        "train_size": len(train_images), "test_size": len(test_images),
        "class_counts": dict(Counter(labels)),
        "results": summary,
    }, indent=2))

    print("\n=== FINAL RANKING (by test accuracy) ===")
    for r in summary:
        print(f"  {r['accuracy']*100:5.1f}%  f1={r['f1_macro']:.3f}  {r['feature_set']:20s} {r['algorithm']}")

    best = max(all_results, key=lambda r: r["accuracy"])
    import joblib
    if "model" in best and best["feature_set"] != "mobilenet_finetuned":
        joblib.dump({"model": best["model"], "scaler": best.get("scaler"), "class_names": class_names,
                     "feature_set": best["feature_set"]}, RESULTS_DIR / "best_classical_model.joblib")
        print(f"\nSaved best classical model: {best['feature_set']} / {best['algorithm']} -> silkworm_results/best_classical_model.joblib")

    import torch
    ft = next(r for r in all_results if r["feature_set"] == "mobilenet_finetuned")
    torch.save({"state_dict": ft["model"].state_dict(), "class_names": class_names},
               RESULTS_DIR / "mobilenet_finetuned.pt")
    print("Saved fine-tuned MobileNet weights -> silkworm_results/mobilenet_finetuned.pt")


if __name__ == "__main__":
    main()
