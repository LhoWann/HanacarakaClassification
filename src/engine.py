import math
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from config import Config
from tqdm.auto import tqdm


log = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    train_loss: list[float] = field(default_factory=list)
    train_acc:  list[float] = field(default_factory=list)
    val_loss:   list[float] = field(default_factory=list)
    val_acc:    list[float] = field(default_factory=list)
    lr:         list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_clip: float = 0.0,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    use_amp = scaler is not None

    for images, labels in tqdm(loader, desc="Train batches", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        total_loss    += loss.item() * batch_size
        total_correct += (logits.argmax(1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Eval batches", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

        total_loss    += loss.item() * labels.size(0)
        total_samples += labels.size(0)

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    accuracy   = accuracy_score(all_labels, all_preds)
    avg_loss   = total_loss / total_samples

    return avg_loss, accuracy, all_preds, all_labels


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps  = cfg.epochs * steps_per_epoch
    warmup_steps = cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def fit(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    cfg: Config,
    device: torch.device,
) -> tuple[TrainMetrics, Path]:
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(loaders["train"])
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None
    log.info(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    metrics = TrainMetrics()
    best_val_acc = 0.0
    patience_counter = 0
    ckpt_path = Path(cfg.checkpoint_name)
    if not ckpt_path.is_absolute() and ckpt_path.parent == Path("."):
        ckpt_path = cfg.artifact_dir / cfg.checkpoint_name
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    epoch_iter = tqdm(range(1, cfg.epochs + 1), desc="Epochs", leave=True)
    for epoch in epoch_iter:
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device,
            scaler=scaler, grad_clip=cfg.grad_clip_norm, scheduler=scheduler,
        )

        val_loss, val_acc, _, _ = evaluate(model, loaders["val"], criterion, device)

        epoch_time = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        metrics.train_loss.append(train_loss)
        metrics.train_acc.append(train_acc)
        metrics.val_loss.append(val_loss)
        metrics.val_acc.append(val_acc)
        metrics.lr.append(current_lr)
        metrics.epoch_time.append(epoch_time)

        epoch_iter.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc":  f"{train_acc:.3f}",
            "val_loss":   f"{val_loss:.4f}",
            "val_acc":    f"{val_acc:.3f}",
            "lr":         f"{current_lr:.2e}",
            "s":          f"{epoch_time:.1f}s",
        })

        log.info(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, ckpt_path)
            log.info(f"  -> New best val_acc={val_acc:.4f}, saved to {ckpt_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                log.info(f"Early stopping: no improvement {patience_counter} epochs.")
                break

    log.info(f"Training done. Best val_acc = {best_val_acc:.4f}")
    return metrics, ckpt_path


def detailed_evaluation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    criterion = nn.CrossEntropyLoss()
    loss, acc, preds, labels = evaluate(model, loader, criterion, device)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )

    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    report_str = classification_report(
        labels, preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    per_class = precision_recall_fscore_support(
        labels, preds,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    return {
        "loss":                  loss,
        "accuracy":              acc,
        "precision_macro":       precision_macro,
        "recall_macro":          recall_macro,
        "f1_macro":              f1_macro,
        "precision_weighted":    precision_w,
        "recall_weighted":       recall_w,
        "f1_weighted":           f1_w,
        "confusion_matrix":      cm,
        "classification_report": report_str,
        "per_class_precision":   per_class[0],
        "per_class_recall":      per_class[1],
        "per_class_f1":          per_class[2],
        "per_class_support":     per_class[3],
        "predictions":           preds,
        "labels":                labels,
    }