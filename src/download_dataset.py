import os
import re
import shutil
import zipfile
import hashlib
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import requests
from tqdm.auto import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split

# CONFIG
CLASSES = [
    "ha", "na", "ca", "ra", "ka",
    "da", "ta", "sa", "wa", "la",
    "pa", "dha", "ja", "ya", "nya",
    "ma", "ga", "ba", "tha", "nga",
]

SPLIT_RATIO = {"train": 0.70, "val": 0.15, "test": 0.15}
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42

BASE_DIR = Path("dataset")
RAW_DIR  = BASE_DIR / "raw"

GITHUB_REPO_ZIP = (
    "https://github.com/vzrenggamani/aksarajawa-hanacaraka/archive/refs/heads/master.zip"
)

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# HELPERS
def file_md5(path: Path, chunk=8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk_data := f.read(chunk):
            h.update(chunk_data)
    return h.hexdigest()


def download_file(url: str, dest: Path, desc: str = "") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info(f"Already exists, skipping: {dest.name}")
        return dest

    log.info(f"Downloading {desc or url}")
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    return dest


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def normalize_class_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"^\d+[_\-]?", "", name)
    aliases = {
        "c": "ca", "r": "ra", "k": "ka", "d": "da", "t": "ta",
        "s": "sa", "w": "wa", "l": "la", "p": "pa", "j": "ja",
        "y": "ya", "m": "ma", "g": "ga", "b": "ba", "n": "na",
        "h": "ha",
    }
    return aliases.get(name, name)


# SOURCE 1: GitHub vzrenggamani
def download_github_vzrenggamani(raw_dir: Path) -> Path:
    dest_zip = raw_dir / "github_vzrenggamani.zip"
    dest_dir = raw_dir / "github_vzrenggamani"

    if dest_dir.exists() and any(dest_dir.rglob("*.jpg")):
        log.info("GitHub source already extracted.")
        return dest_dir

    download_file(GITHUB_REPO_ZIP, dest_zip, desc="GitHub vzrenggamani/aksarajawa-hanacaraka")

    log.info("Extracting GitHub zip...")
    with zipfile.ZipFile(dest_zip, "r") as zf:
        zf.extractall(raw_dir / "_gh_tmp")

    extracted = next((raw_dir / "_gh_tmp").iterdir())
    shutil.move(str(extracted), str(dest_dir))
    shutil.rmtree(raw_dir / "_gh_tmp", ignore_errors=True)

    log.info(f"GitHub source: {dest_dir}")
    return dest_dir


# SOURCE 2: Roboflow fawwaz — Object Detection → Crop per class
def download_roboflow_fawwaz(raw_dir: Path, api_key: str) -> Path:
    """
    Dataset fawwaz adalah object-detection project.
    Download format yolov8, lalu crop setiap bounding box
    menjadi gambar klasifikasi individual per kelas.
    """
    dest_dir  = raw_dir / "roboflow_fawwaz"
    crop_dir  = raw_dir / "roboflow_fawwaz_crops"

    if crop_dir.exists() and any(crop_dir.rglob("*.jpg")):
        log.info("Roboflow fawwaz crops already exist.")
        return crop_dir

    if not api_key:
        log.warning("ROBOFLOW_API_KEY tidak di-set — skip fawwaz source.")
        return crop_dir

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace("fawwaz-zaini-ahmad-ce2om").project(
            "javanese-handwriting-object-detection-aksara-jawa"
        )

        log.info("Downloading fawwaz dataset (yolov8 format).")
        proj.version(2).download(
            model_format="yolov8",
            location=str(dest_dir),
            overwrite=False,
        )
        log.info(f"Download selesai: {dest_dir}")

    except Exception as e:
        log.error(f"Download fawwaz gagal: {e}")
        return crop_dir

    # ── Crop bounding boxes → klasifikasi per kelas
    log.info("Cropping bounding boxes → class folders.")
    _crop_yolov8_to_classification(dest_dir, crop_dir)

    return crop_dir


def _crop_yolov8_to_classification(yolo_dir: Path, out_dir: Path):
    """
    YOLOv8 format:
      images/train/*.jpg
      labels/train/*.txt   (class_idx cx cy w h  — normalized 0..1)
    data.yaml berisi nama kelas.

    Output: out_dir/<class_name>/<img>_<idx>.jpg
    """
    import yaml

    # Baca class names dari data.yaml
    yaml_path = yolo_dir / "data.yaml"
    if not yaml_path.exists():
        yaml_path = next(yolo_dir.rglob("data.yaml"), None)

    if yaml_path is None:
        log.warning("data.yaml tidak ditemukan, skip cropping.")
        return

    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    class_names = meta.get("names", [])
    log.info(f"Classes dari data.yaml: {class_names}")

    n_cropped = 0
    for split in ["train", "valid", "test"]:
        img_dir   = yolo_dir / split / "images"
        label_dir = yolo_dir / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in tqdm(list(img_dir.iterdir()), desc=f"{split} images", leave=False):
            if img_path.suffix.lower() not in IMG_EXTENSIONS:
                continue

            label_path = label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                W, H = img.size
            except Exception:
                continue

            with open(label_path) as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_idx = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # Convert normalized → pixel coords
                x1 = int((cx - bw / 2) * W)
                y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W)
                y2 = int((cy + bh / 2) * H)

                # Clamp ke batas gambar
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if x2 <= x1 or y2 <= y1:
                    continue  # bbox tidak valid

                crop = img.crop((x1, y1, x2, y2))

                # Resolve class name
                if cls_idx < len(class_names):
                    raw_cls = class_names[cls_idx]
                    cls_name = normalize_class_name(raw_cls)
                else:
                    cls_name = f"class_{cls_idx}"

                if cls_name not in CLASSES:
                    log.debug(f"Unknown class '{cls_name}' (raw: '{raw_cls}'), skip.")
                    continue

                save_dir = out_dir / cls_name
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / f"{img_path.stem}_{idx}.jpg"
                crop.save(save_path, "JPEG", quality=95)
                n_cropped += 1

    log.info(f"Total crops: {n_cropped} gambar dari fawwaz dataset")


# COLLECT, DEDUPLICATE, VALIDATE
def collect_images_from_source(source_dir: Path) -> dict[str, list[Path]]:
    class_images: dict[str, list[Path]] = defaultdict(list)
    for img_path in tqdm(list(source_dir.rglob("*")), desc=f"Scanning {source_dir.name}", leave=False):
        if img_path.suffix.lower() not in IMG_EXTENSIONS:
            continue
        cls = normalize_class_name(img_path.parent.name)
        if cls not in CLASSES:
            cls = normalize_class_name(img_path.parent.parent.name)
        if cls in CLASSES:
            class_images[cls].append(img_path)
    return class_images


def deduplicate(paths: list[Path]) -> list[Path]:
    seen, unique = set(), []
    for p in tqdm(paths, desc="Deduplicating", leave=False):
        h = file_md5(p)
        if h not in seen:
            seen.add(h)
            unique.append(p)
    return unique


def validate_images(paths: list[Path]) -> list[Path]:
    valid = []
    for p in tqdm(paths, desc="Validating", leave=False):
        if is_valid_image(p):
            valid.append(p)
        else:
            log.warning(f"Corrupt image removed: {p}")
    return valid


# SPLIT & COPY
def build_splits(
    class_images: dict[str, list[Path]],
    ratio: dict,
    seed: int,
    ood_source_tag: str = "roboflow_thesis",
) -> dict[str, dict[str, list[Path]]]:
    splits: dict[str, dict[str, list[Path]]] = {
        "train": defaultdict(list),
        "val":   defaultdict(list),
        "test":  defaultdict(list),
    }

    for cls, paths in class_images.items():
        ood = [p for p in paths if ood_source_tag in str(p)]
        iid = [p for p in paths if ood_source_tag not in str(p)]

        splits["test"][cls].extend(ood)

        if len(iid) < 3:
            log.warning(f"Class '{cls}' hanya {len(iid)} gambar in-distribution.")
            splits["train"][cls].extend(iid)
            continue

        val_test_ratio = ratio["val"] + ratio["test"]
        train_imgs, val_test_imgs = train_test_split(
            iid, test_size=val_test_ratio, random_state=seed
        )
        relative_test = ratio["test"] / val_test_ratio
        val_imgs, test_imgs = train_test_split(
            val_test_imgs, test_size=relative_test, random_state=seed
        )

        splits["train"][cls].extend(train_imgs)
        splits["val"][cls].extend(val_imgs)
        splits["test"][cls].extend(test_imgs)

    return splits


def copy_splits(splits: dict, base_dir: Path):
    for split_name, class_map in splits.items():
        for cls, paths in class_map.items():
            dest_cls_dir = base_dir / split_name / cls
            dest_cls_dir.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(tqdm(paths, desc=f"Copy {cls}", leave=False)):
                dest = dest_cls_dir / f"{cls}_{i:04d}{src.suffix.lower()}"
                if not dest.exists():
                    shutil.copy2(src, dest)
    log.info("Semua gambar berhasil di-copy.")


def print_report(splits: dict):
    log.info("\n" + "=" * 45)
    log.info("DATASET SPLIT SUMMARY")
    log.info("=" * 45)
    log.info(f"{'Class':<10} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    log.info("-" * 45)

    grand = {"train": 0, "val": 0, "test": 0}
    for cls in CLASSES:
        n_train = len(splits["train"].get(cls, []))
        n_val   = len(splits["val"].get(cls, []))
        n_test  = len(splits["test"].get(cls, []))
        total   = n_train + n_val + n_test
        grand["train"] += n_train
        grand["val"]   += n_val
        grand["test"]  += n_test
        log.info(f"{cls:<10} {n_train:>7} {n_val:>7} {n_test:>7} {total:>7}")

    log.info("-" * 45)
    grand_total = sum(grand.values())
    log.info(
        f"{'TOTAL':<10} {grand['train']:>7} {grand['val']:>7} {grand['test']:>7} {grand_total:>7}"
    )
    log.info("=" * 45)


# MAIN
def parse_args():
    p = argparse.ArgumentParser(description="Aksara Jawa dataset downloader & organizer")
    p.add_argument("--roboflow-key", default=ROBOFLOW_API_KEY)
    p.add_argument("--skip-roboflow", action="store_true")
    p.add_argument("--output-dir", default="dataset")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = Path(args.output_dir)
    raw_dir  = base_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 50)
    log.info("Aksara Jawa Dataset Builder")
    log.info("=" * 50)

    source_dirs = []

    # 1. GitHub
    log.info("\n[1/4] GitHub source.")
    source_dirs.append(download_github_vzrenggamani(raw_dir))

    # 2 & 3. Roboflow
    if not args.skip_roboflow:
        api_key = args.roboflow_key

        log.info("\n[1/4] Roboflow: fawwaz (object detection → crop).")
        fawwaz_dir = download_roboflow_fawwaz(raw_dir, api_key)
        if fawwaz_dir.exists() and any(fawwaz_dir.rglob("*.jpg")):
            source_dirs.append(fawwaz_dir)

        log.info("\n[1/4] Roboflow: thesis (OOD test, auto-probe version).")
        thesis_dir = download_roboflow_thesis(raw_dir, api_key)
        if thesis_dir.exists() and any(thesis_dir.rglob("*.jpg")):
            source_dirs.append(thesis_dir)

    # 2. Collect & clean
    log.info("\n[2/4] Collecting dan cleaning images.")
    all_class_images: dict[str, list[Path]] = defaultdict(list)

    for src_dir in source_dirs:
        found = collect_images_from_source(src_dir)
        for cls, paths in found.items():
            all_class_images[cls].extend(paths)
        n = sum(len(v) for v in found.values())
        log.info(f"  {src_dir.name}: {n} images")

    log.info("  Deduplicating & validating.")
    for cls in all_class_images:
        before = len(all_class_images[cls])
        all_class_images[cls] = deduplicate(all_class_images[cls])
        all_class_images[cls] = validate_images(all_class_images[cls])
        after = len(all_class_images[cls])
        if before != after:
            log.info(f"  {cls}: removed {before - after} duplicates/corrupt")

    total = sum(len(v) for v in all_class_images.values())
    log.info(f"  Total clean images: {total}")

    # 3. Split
    log.info("\n[3/4] Splitting.")
    splits = build_splits(all_class_images, SPLIT_RATIO, SEED)

    # 4. Copy
    log.info("\n[4/4] Writing dataset structure.")
    copy_splits(splits, base_dir)

    print_report(splits)
    log.info(f"\nDone. Dataset: {base_dir.resolve()}")


if __name__ == "__main__":
    main()