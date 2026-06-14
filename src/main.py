import warnings
warnings.filterwarnings("ignore")
import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from config import get_config, set_seed, CLASSES, IDX_TO_CLASS, NUM_CLASSES
from dataset import build_dataloaders, build_transforms, AksaraJawaDataset
from model import SimpleCNN, ImprovedCNN, count_parameters
from engine import fit, detailed_evaluation
from explainability import GradCAM, overlay_cam_on_image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MODEL_FOLDER_MAP = {
    "simple":   "simple_cnn",
    "improved": "improved_cnn",
}


def parse_args():
    p = argparse.ArgumentParser(description="Klasifikasi Aksara Jawa (Hanacaraka)")
    p.add_argument(
        "--mode",
        choices=["eda", "train", "eval", "gradcam", "compare", "all"],
        default="all",
    )
    p.add_argument(
        "--model",
        choices=["simple", "improved", "both"],
        default="both",
    )
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--batch-size", type=int,   default=None)
    p.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
    )
    p.add_argument("--output-dir",       default="outputs")
    p.add_argument("--gradcam-samples",  type=int, default=2)
    p.add_argument("--no-save-plots",    action="store_true")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def build_model(model_name: str, cfg) -> torch.nn.Module:
    if model_name == "simple":
        return SimpleCNN(
            num_classes=NUM_CLASSES,
            in_channels=cfg.in_channels,
            dropout=cfg.dropout,
        )
    return ImprovedCNN(
        num_classes=NUM_CLASSES,
        in_channels=cfg.in_channels,
        dropout=cfg.dropout,
    )


def model_display_name(model_name: str) -> str:
    return "SimpleCNN" if model_name == "simple" else "ImprovedCNN"


def get_model_out_dir(base_out_dir: Path, model_name: str) -> Path:
    return base_out_dir / MODEL_FOLDER_MAP[model_name]


def get_model_ckpt_dir(artifact_dir: Path, model_name: str) -> Path:
    d = artifact_dir / MODEL_FOLDER_MAP[model_name]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig: plt.Figure, path: Path, save: bool):
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info(f"Saved: {path}")
    plt.close(fig)


def get_model_list(model_arg: str) -> list[str]:
    if model_arg == "both":
        return ["simple", "improved"]
    return [model_arg]


def run_eda(cfg, out_dir: Path, save_plots: bool):
    log.info("EDA")
    eda_dir = out_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    split_counts = {}
    for split in ["train", "val", "test"]:
        split_dir = cfg.data_dir / split
        if not split_dir.exists():
            log.warning(f"Split '{split}' tidak ditemukan: {split_dir}")
            split_counts[split] = {}
            continue
        counts = {}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                counts[cls] = len([
                    f for f in cls_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                ])
            else:
                counts[cls] = 0
        split_counts[split] = counts

    total_per_split = {s: sum(c.values()) for s, c in split_counts.items()}
    log.info(f"Total images per split: {total_per_split}")

    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(CLASSES))
    width = 0.28
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (split, color) in enumerate(zip(["train", "val", "test"], colors)):
        vals = [split_counts[split].get(cls, 0) for cls in CLASSES]
        ax.bar(x + i * width, vals, width, label=split, color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Jumlah Gambar")
    ax.set_title("Distribusi Kelas per Split")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, eda_dir / "class_distribution.png", save_plots)

    log.info("Mengumpulkan sampel per kelas.")
    train_ds = AksaraJawaDataset(cfg.data_dir / "train", transform=None)

    samples_by_class: dict[str, list] = {cls: [] for cls in CLASSES}
    for img_path, label in train_ds.samples:
        cls = IDX_TO_CLASS[label]
        if len(samples_by_class[cls]) < 3:
            samples_by_class[cls].append(img_path)

    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.suptitle("Satu Sampel per Kelas (setelah preprocessing)", fontsize=13)
    for i, cls in enumerate(CLASSES):
        ax = axes[i // 5][i % 5]
        paths = samples_by_class[cls]
        if paths:
            img = Image.open(paths[0]).convert("L")
            img = ImageOps.autocontrast(img, cutoff=2)
            img = ImageOps.invert(img)
            img = ImageOps.pad(img, (max(img.size), max(img.size)), color=0, centering=(0.5, 0.5))
            ax.imshow(img, cmap="gray")
        ax.set_title(cls, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, eda_dir / "samples_per_class.png", save_plots)

    log.info("Analisis ukuran gambar original.")
    widths, heights = [], []
    for img_path, _ in train_ds.samples[:500]:
        try:
            with Image.open(img_path) as im:
                w, h = im.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(widths, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Frekuensi")
    axes[0].set_title("Distribusi Width Gambar Original")
    axes[0].axvline(np.median(widths), color="red", linestyle="--", label=f"Median={np.median(widths):.0f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(heights, bins=30, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Height (px)")
    axes[1].set_ylabel("Frekuensi")
    axes[1].set_title("Distribusi Height Gambar Original")
    axes[1].axvline(np.median(heights), color="red", linestyle="--", label=f"Median={np.median(heights):.0f}")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, eda_dir / "image_size_distribution.png", save_plots)

    if train_ds.samples:
        sample_path = train_ds.samples[0][0]
        raw_img = Image.open(sample_path).convert("L")
        after_ac = ImageOps.autocontrast(raw_img, cutoff=2)
        after_inv = ImageOps.invert(after_ac)
        after_pad = ImageOps.pad(after_inv, (max(after_inv.size), max(after_inv.size)), color=0, centering=(0.5, 0.5))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        stages = [raw_img, after_ac, after_inv, after_pad]
        titles = ["Original", "AutoContrast", "Invert", "Square Pad"]
        for ax, img, title in zip(axes, stages, titles):
            ax.imshow(img, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle("Tahapan Preprocessing", fontsize=12)
        fig.tight_layout()
        save_fig(fig, eda_dir / "preprocessing_pipeline.png", save_plots)

    log.info("EDA selesai.")


def run_training(cfg, model_name: str, device: torch.device, out_dir: Path, save_plots: bool):
    display = model_display_name(model_name)
    log.info(f"TRAINING [{display}]")
    log.info(f"Device: {device}")

    model_out = get_model_out_dir(out_dir, model_name)
    ckpt_dir = get_model_ckpt_dir(cfg.artifact_dir, model_name)

    original_ckpt_name = cfg.checkpoint_name
    cfg.checkpoint_name = str(ckpt_dir / original_ckpt_name)

    loaders, _ = build_dataloaders(cfg)
    model = build_model(model_name, cfg).to(device)

    total_params, trainable_params = count_parameters(model)
    log.info(f"Model: {display}")
    log.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
    log.info(f"Training: {len(loaders['train'].dataset)} train | {len(loaders['val'].dataset)} val")

    ckpt_path_obj = Path(cfg.checkpoint_name)
    ckpt_path_obj.parent.mkdir(parents=True, exist_ok=True)

    metrics, ckpt_path = fit(model, loaders, cfg, device)

    cfg.checkpoint_name = original_ckpt_name

    train_dir = model_out / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {
        "model":        model_name,
        "display_name": display,
        "train_loss":   metrics.train_loss,
        "train_acc":    metrics.train_acc,
        "val_loss":     metrics.val_loss,
        "val_acc":      metrics.val_acc,
        "lr":           metrics.lr,
        "epoch_time":   metrics.epoch_time,
        "best_val_acc": max(metrics.val_acc) if metrics.val_acc else 0.0,
        "total_params": total_params,
        "checkpoint":   str(ckpt_path),
    }
    with open(train_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    log.info(f"Metrics saved: {train_dir / 'metrics.json'}")

    epochs_range = list(range(1, len(metrics.train_loss) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Training Curves - {display}", fontsize=13)

    axes[0].plot(epochs_range, metrics.train_loss, label="Train Loss", color="#4C72B0")
    axes[0].plot(epochs_range, metrics.val_loss,   label="Val Loss",   color="#C44E52")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, [v * 100 for v in metrics.train_acc], label="Train Acc", color="#4C72B0")
    axes[1].plot(epochs_range, [v * 100 for v in metrics.val_acc],   label="Val Acc",   color="#C44E52")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs_range, metrics.lr, color="#55A868")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, train_dir / "training_curves.png", save_plots)

    log.info(f"Training [{display}] selesai. Best val_acc = {max(metrics.val_acc):.4f}")
    return metrics_dict


def run_evaluation(cfg, model_name: str, device: torch.device, out_dir: Path, save_plots: bool):
    display = model_display_name(model_name)
    log.info(f"EVALUATION [{display}]")

    ckpt_dir = get_model_ckpt_dir(cfg.artifact_dir, model_name)
    ckpt_path = ckpt_dir / cfg.checkpoint_name
    if not ckpt_path.exists():
        log.error(f"Checkpoint tidak ditemukan: {ckpt_path}. Jalankan --mode train dulu.")
        return None

    model = build_model(model_name, cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Loaded checkpoint dari epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    loaders, _ = build_dataloaders(cfg)
    results = detailed_evaluation(model, loaders["test"], device, CLASSES)

    log.info(f"[{display}] Test Accuracy : {results['accuracy']:.4f}")
    log.info(f"[{display}] F1 Macro      : {results['f1_macro']:.4f}")
    log.info(f"[{display}] F1 Weighted   : {results['f1_weighted']:.4f}")
    log.info("\n" + results["classification_report"])

    model_out = get_model_out_dir(out_dir, model_name)
    eval_dir = model_out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in results.items()
        if k not in ("confusion_matrix", "predictions", "labels",
                     "per_class_precision", "per_class_recall",
                     "per_class_f1", "per_class_support")
    }
    serializable["confusion_matrix"] = results["confusion_matrix"].tolist()
    serializable["per_class_f1"] = results["per_class_f1"].tolist()
    serializable["model"] = model_name
    serializable["display_name"] = display
    with open(eval_dir / "eval_results.json", "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Eval results saved: {eval_dir / 'eval_results.json'}")

    cm = results["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Ground Truth")
    ax.set_title(f"Confusion Matrix - {display} (acc={results['accuracy']:.4f})")
    thresh = cm.max() / 2.0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    save_fig(fig, eval_dir / "confusion_matrix.png", save_plots)

    f1_scores = results["per_class_f1"]
    sorted_idx = np.argsort(f1_scores)
    sorted_classes = [CLASSES[i] for i in sorted_idx]
    sorted_f1 = [f1_scores[i] for i in sorted_idx]
    colors_bar = ["#C44E52" if v < 0.85 else "#55A868" for v in sorted_f1]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(sorted_classes, [v * 100 for v in sorted_f1], color=colors_bar, edgecolor="white")
    ax.set_xlabel("F1-Score (%)")
    ax.set_title(f"F1-Score per Kelas - {display}")
    ax.axvline(results["f1_macro"] * 100, color="navy", linestyle="--", label=f"Macro F1={results['f1_macro']:.4f}")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, sorted_f1):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    save_fig(fig, eval_dir / "f1_per_class.png", save_plots)

    return results


def run_gradcam(cfg, model_name: str, device: torch.device, out_dir: Path,
                samples_per_class: int, save_plots: bool):
    display = model_display_name(model_name)
    log.info(f"GRAD-CAM [{display}]")

    ckpt_dir = get_model_ckpt_dir(cfg.artifact_dir, model_name)
    ckpt_path = ckpt_dir / cfg.checkpoint_name
    if not ckpt_path.exists():
        log.error(f"Checkpoint tidak ditemukan: {ckpt_path}. Jalankan --mode train dulu.")
        return

    model = build_model(model_name, cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    target_layer = model.features[-3]

    _, eval_tf = build_transforms(cfg)
    test_ds = AksaraJawaDataset(cfg.data_dir / "test", transform=None)

    samples_by_class: dict[str, list] = {cls: [] for cls in CLASSES}
    for img_path, label in test_ds.samples:
        cls = IDX_TO_CLASS[label]
        if len(samples_by_class[cls]) < samples_per_class:
            samples_by_class[cls].append(img_path)

    gradcam = GradCAM(model, target_layer)
    model_out = get_model_out_dir(out_dir, model_name)
    gradcam_dir = model_out / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        paths = samples_by_class[cls]
        if not paths:
            log.warning(f"Tidak ada sampel untuk kelas '{cls}'")
            continue

        n = len(paths)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3.5 * n))
        if n == 1:
            axes = [axes]
        fig.suptitle(f"Grad-CAM [{display}]: Kelas '{cls}'", fontsize=12)

        for row, img_path in enumerate(paths):
            raw = Image.open(img_path).convert("L")
            raw = ImageOps.autocontrast(raw, cutoff=2)
            raw = ImageOps.invert(raw)
            raw = ImageOps.pad(raw, (max(raw.size), max(raw.size)), color=0, centering=(0.5, 0.5))

            tensor = eval_tf(raw).unsqueeze(0).to(device)
            img_np = np.array(raw.resize((cfg.image_size, cfg.image_size)))

            cam, pred_class, confidence = gradcam(tensor)
            pred_name = IDX_TO_CLASS[pred_class]
            overlay = overlay_cam_on_image(img_np, cam, alpha=0.4)

            axes[row][0].imshow(img_np, cmap="gray")
            axes[row][0].set_title("Input")
            axes[row][0].axis("off")

            axes[row][1].imshow(cam, cmap="jet")
            axes[row][1].set_title("Heatmap")
            axes[row][1].axis("off")

            axes[row][2].imshow(overlay)
            correct = pred_name == cls
            color = "green" if correct else "red"
            axes[row][2].set_title(
                f"Pred: {pred_name} ({confidence:.2%})",
                color=color,
            )
            axes[row][2].axis("off")

        fig.tight_layout()
        save_fig(fig, gradcam_dir / f"gradcam_{cls}.png", save_plots)

    gradcam.remove_hooks()
    log.info(f"Grad-CAM [{display}] selesai. Hasil di: {gradcam_dir}")


def run_comparison(out_dir: Path, save_plots: bool):
    log.info("COMPARISON SimpleCNN vs ImprovedCNN")

    compare_dir = out_dir / "comparison"
    compare_dir.mkdir(parents=True, exist_ok=True)

    eval_data = {}
    train_data = {}
    for mname in ["simple", "improved"]:
        model_out = get_model_out_dir(out_dir, mname)
        eval_path = model_out / "evaluation" / "eval_results.json"
        train_path = model_out / "training" / "metrics.json"

        if eval_path.exists():
            with open(eval_path) as f:
                eval_data[mname] = json.load(f)
        else:
            log.warning(f"Eval results tidak ditemukan untuk {mname}: {eval_path}")

        if train_path.exists():
            with open(train_path) as f:
                train_data[mname] = json.load(f)
        else:
            log.warning(f"Training metrics tidak ditemukan untuk {mname}: {train_path}")

    if len(eval_data) < 2:
        log.error("Perlu eval results dari kedua model untuk comparison. Jalankan --mode all --model both dulu.")
        return

    summary = {}
    for mname, data in eval_data.items():
        display = model_display_name(mname)
        params = train_data.get(mname, {}).get("total_params", "N/A")
        summary[display] = {
            "accuracy":     data["accuracy"],
            "f1_macro":     data["f1_macro"],
            "f1_weighted":  data["f1_weighted"],
            "precision_macro": data["precision_macro"],
            "recall_macro":    data["recall_macro"],
            "total_params": params,
        }

    with open(compare_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Comparison summary saved: {compare_dir / 'comparison_summary.json'}")

    log.info("\n" + "=" * 60)
    log.info("MODEL COMPARISON")
    log.info("=" * 60)
    log.info(f"{'Metric':<22} {'SimpleCNN':>12} {'ImprovedCNN':>12} {'Delta':>10}")
    log.info("-" * 60)
    s = summary.get("SimpleCNN", {})
    imp = summary.get("ImprovedCNN", {})
    for metric in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
        sv = s.get(metric, 0)
        iv = imp.get(metric, 0)
        delta = iv - sv
        sign = "+" if delta >= 0 else ""
        log.info(f"{metric:<22} {sv:>11.4f} {iv:>12.4f} {sign}{delta:>9.4f}")
    sp = s.get("total_params", "N/A")
    ip = imp.get("total_params", "N/A")
    log.info(f"{'total_params':<22} {sp:>12,} {ip:>12,}")
    log.info("=" * 60)

    metrics_list = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    simple_vals = [s.get(m, 0) * 100 for m in metrics_list]
    improved_vals = [imp.get(m, 0) * 100 for m in metrics_list]

    x = np.arange(len(metrics_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, simple_vals, width, label="SimpleCNN", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, improved_vals, width, label="ImprovedCNN", color="#C44E52", alpha=0.85)
    ax.set_ylabel("Score (%)")
    ax.set_title("SimpleCNN vs ImprovedCNN - Perbandingan Metrik")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics_list], rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 105)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    save_fig(fig, compare_dir / "metrics_comparison.png", save_plots)

    simple_f1 = np.array(eval_data["simple"].get("per_class_f1", [0] * NUM_CLASSES))
    improved_f1 = np.array(eval_data["improved"].get("per_class_f1", [0] * NUM_CLASSES))
    delta_f1 = improved_f1 - simple_f1
    sorted_idx = np.argsort(delta_f1)

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(NUM_CLASSES)
    bar_colors = ["#55A868" if d >= 0 else "#C44E52" for d in delta_f1[sorted_idx]]
    bars = ax.barh(y, delta_f1[sorted_idx] * 100, color=bar_colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([CLASSES[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel("F1 Delta (%): ImprovedCNN - SimpleCNN")
    ax.set_title("Perubahan F1 per Kelas (ImprovedCNN vs SimpleCNN)")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, delta_f1[sorted_idx]):
        offset = 0.3 if val >= 0 else -0.3
        ha = "left" if val >= 0 else "right"
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{val * 100:+.1f}%", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    save_fig(fig, compare_dir / "f1_delta_per_class.png", save_plots)

    if len(train_data) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Training Curves - SimpleCNN vs ImprovedCNN", fontsize=13)

        for mname, color, label in [("simple", "#4C72B0", "SimpleCNN"), ("improved", "#C44E52", "ImprovedCNN")]:
            td = train_data[mname]
            epochs = list(range(1, len(td["val_loss"]) + 1))
            axes[0].plot(epochs, td["val_loss"], label=f"{label}", color=color)
            axes[1].plot(epochs, [v * 100 for v in td["val_acc"]], label=f"{label}", color=color)

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Val Loss")
        axes[0].set_title("Validation Loss")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Val Accuracy (%)")
        axes[1].set_title("Validation Accuracy")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        save_fig(fig, compare_dir / "training_comparison.png", save_plots)

    log.info(f"Comparison selesai. Hasil di: {compare_dir}")


def main():
    args = parse_args()
    cfg = get_config()
    set_seed(cfg.seed)

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    save_plots = not args.no_save_plots
    models = get_model_list(args.model)

    log.info(f"Mode   : {args.mode}")
    log.info(f"Model  : {args.model} -> {[model_display_name(m) for m in models]}")
    log.info(f"Device : {device}")
    log.info(f"Output : {out_dir}")

    mode = args.mode

    if mode in ("eda", "all"):
        run_eda(cfg, out_dir, save_plots)

    if mode in ("train", "all"):
        for m in models:
            set_seed(cfg.seed)
            run_training(cfg, m, device, out_dir, save_plots)

    if mode in ("eval", "all"):
        for m in models:
            result = run_evaluation(cfg, m, device, out_dir, save_plots)
            if result is None:
                log.error(f"Evaluasi [{model_display_name(m)}] dibatalkan.")

    if mode in ("gradcam", "all"):
        for m in models:
            run_gradcam(cfg, m, device, out_dir, args.gradcam_samples, save_plots)

    if mode in ("compare", "all"):
        if args.model == "both" or mode == "compare":
            run_comparison(out_dir, save_plots)

    log.info("Done")


if __name__ == "__main__":
    main()