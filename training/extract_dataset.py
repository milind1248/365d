"""One-off script: extract the Kaggle mulberry-leaf-dataset zip and report
the actual folder/class structure (Kaggle discussion flagged a possible
leaf_rust/leaf_spot label swap - verify before trusting labels blindly)."""
import zipfile
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
ZIP_PATH = HERE / "mulberry-leaf-dataset.zip"
EXTRACT_DIR = HERE / "raw"

with zipfile.ZipFile(ZIP_PATH) as z:
    bad = z.testzip()
    if bad:
        raise SystemExit(f"Corrupt member in zip: {bad}")
    print(f"Zip OK, {len(z.namelist())} entries. Extracting...")
    z.extractall(EXTRACT_DIR)

image_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
counts = Counter()
folder_for_class = {}
for p in EXTRACT_DIR.rglob("*"):
    if p.is_file() and p.suffix in image_exts:
        cls = p.parent.name
        counts[cls] += 1
        folder_for_class.setdefault(cls, str(p.parent))

print("\nClass folder counts:")
for cls, n in counts.items():
    print(f"  {cls}: {n} images  ({folder_for_class[cls]})")
print(f"\nTotal images: {sum(counts.values())}")
