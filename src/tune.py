import warnings
warnings.filterwarnings("ignore")
import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import torch

from config import get_config, set_seed, NUM_CLASSES
from dataset import build_dataloaders
from model import SimpleCNN, ImprovedCNN
from engine import fit


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def parse_args():
    p = argparse.ArgumentParser(description="Optuna HPO untuk Klasifikasi Hanacaraka")
    p.add_argument("--model", choices=["simple", "improved", "both"], default="both")
    p.add_argument("--n-trials",    type=int, default=30)
    p.add_argument("--tune-epochs", type=int, default=20)
    p.add_argument("--device",      choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--output-dir",  default="artifacts")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def resolve_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def build_model(model_name, cfg):
    if model_name == "simple":
        return SimpleCNN(num_classes=NUM_CLASSES, in_channels=cfg.in_channels, dropout=cfg.dropout)
    return ImprovedCNN(num_classes=NUM_CLASSES, in_channels=cfg.in_channels, dropout=cfg.dropout)


def make_objective(model_name, tune_epochs, device, base_seed):
    def objective(trial):
        cfg = get_config()
        cfg.learning_rate    = trial.suggest_float("learning_rate",    5e-4, 5e-3, log=True)
        cfg.weight_decay     = trial.suggest_float("weight_decay",     1e-5, 1e-3, log=True)
        cfg.dropout          = trial.suggest_float("dropout",          0.2,  0.5)
        cfg.batch_size       = trial.suggest_categorical("batch_size", [32, 64, 128])
        cfg.label_smoothing  = trial.suggest_float("label_smoothing",  0.0,  0.15)
        cfg.aug_rotation_deg = trial.suggest_float("aug_rotation_deg", 5.0, 15.0)
        cfg.aug_erasing_prob = trial.suggest_float("aug_erasing_prob", 0.0,  0.3)
        cfg.warmup_epochs    = trial.suggest_categorical("warmup_epochs", [1, 2, 3])
        cfg.epochs                  = tune_epochs
        cfg.early_stopping_patience = 5
        cfg.checkpoint_name         = f"tune_trial_{trial.number}_{model_name}.pt"
        set_seed(base_seed + trial.number)
        try:
            loaders, _ = build_dataloaders(cfg)
            model = build_model(model_name, cfg).to(device)
            metrics, _ = fit(model, loaders, cfg, device)
        except Exception as e:
            log.warning(f"Trial {trial.number} failed: {e}")
            raise optuna.exceptions.TrialPruned()
        best_val_acc = max(metrics.val_acc) if metrics.val_acc else 0.0
        ckpt = cfg.artifact_dir / cfg.checkpoint_name
        if ckpt.exists():
            ckpt.unlink()
        return best_val_acc
    return objective


def run_study(model_name, n_trials, tune_epochs, device, output_dir, seed):
    display = "SimpleCNN" if model_name == "simple" else "ImprovedCNN"
    log.info(f"Optuna study: {display} | n_trials={n_trials} | tune_epochs={tune_epochs} | device={device}")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name=f"hanacaraka_{model_name}",
    )
    study.optimize(make_objective(model_name, tune_epochs, device, seed), n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial
    log.info("=" * 60)
    log.info(f"[{display}] Study selesai. Best trial #{best.number} val_acc={best.value:.4f}")
    log.info("Best params:")
    for k, v in best.params.items():
        log.info(f"  {k}: {v}")
    top5 = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value, reverse=True)[:5]
    log.info(f"\nTop-5 Trials [{display}]:")
    log.info(f"{'#':<6} {'val_acc':<12} {'lr':<12} {'wd':<12} {'dropout':<10} {'bs':<6}")
    log.info("-" * 60)
    for t in top5:
        p = t.params
        log.info(f"{t.number:<6} {t.value:<12.4f} {p.get('learning_rate',0):<12.2e} {p.get('weight_decay',0):<12.2e} {p.get('dropout',0):<10.3f} {p.get('batch_size',0):<6}")
    out_path = output_dir / f"best_params_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"model": model_name, "display_name": display, "best_val_acc": best.value, "best_params": best.params, "n_trials": n_trials, "tune_epochs": tune_epochs}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Best params saved: {out_path}")
    return result


def main():
    args = parse_args()
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    models_to_tune = ["simple", "improved"] if args.model == "both" else [args.model]
    all_results = {}
    for model_name in models_to_tune:
        all_results[model_name] = run_study(
            model_name=model_name,
            n_trials=args.n_trials,
            tune_epochs=args.tune_epochs,
            device=device,
            output_dir=output_dir,
            seed=args.seed,
        )
    log.info("\n" + "=" * 60)
    log.info("SUMMARY BEST HYPERPARAMETERS")
    log.info("=" * 60)
    for model_name, result in all_results.items():
        log.info(f"\n[{result['display_name']}] best_val_acc={result['best_val_acc']:.4f}")
        for k, v in result["best_params"].items():
            log.info(f"  {k}: {v}")
    log.info("=" * 60)
    log.info("Gunakan best params di atas untuk full training di main.py")


if __name__ == "__main__":
    main()
