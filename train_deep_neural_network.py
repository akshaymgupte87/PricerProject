"""Train the Pricer text-regression network and create a local .pth checkpoint."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from agents.deep_neural_network import DeepNeuralNetwork
from agents.items import Item


TRAINING_PROFILES = {
    "legacy": {
        "epochs": 5,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "input_size": 5_000,
        "hidden_size": 512,
        "num_layers": 4,
        "dropout": 0.2,
        "loss": "log-mse",
        "patience": 0,
        "vectorizer": {
            "stop_words": "english",
            "binary": True,
            "ngram_range": (1, 1),
            "alternate_sign": True,
            "norm": "l2",
        },
    },
    "improved": {
        "epochs": 10,
        "batch_size": 1_024,
        "learning_rate": 3e-4,
        "input_size": 20_000,
        "hidden_size": 1_024,
        "num_layers": 4,
        "dropout": 0.2,
        "loss": "dollar-huber",
        "patience": 3,
        "vectorizer": {
            "stop_words": "english",
            "binary": False,
            "ngram_range": (1, 2),
            "alternate_sign": False,
            "norm": "l2",
        },
    },
}


class ProductDataset(Dataset):
    """Keep product text sparse until a batch is assembled."""

    def __init__(self, items, target_mean: float, target_std: float):
        self.items = [
            item
            for item in items
            if item.summary and math.isfinite(item.price) and item.price >= 0
        ]
        self.target_mean = target_mean
        self.target_std = target_std

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        normalized_target = (math.log1p(item.price) - self.target_mean) / self.target_std
        return item.summary, normalized_target


def make_collate_fn(vectorizer: HashingVectorizer):
    def collate(batch):
        texts, targets = zip(*batch)
        # Densify only one batch; never create an 800k x 5000 matrix in memory.
        features = vectorizer.transform(texts).toarray().astype(np.float32, copy=False)
        return torch.from_numpy(features), torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

    return collate


def calculate_training_loss(
    predictions,
    targets,
    loss_name: str,
    target_mean: float,
    target_std: float,
) -> torch.Tensor:
    """Calculate either the original log-MSE or a loss aligned with dollar MAE."""
    if loss_name == "log-mse":
        return F.mse_loss(predictions, targets)

    # Smooth L1 behaves like squared error within $25 and absolute dollar error
    # outside it. Scaling by the median price keeps gradients well-conditioned.
    price_scale = max(math.expm1(target_mean), 1.0)
    predicted_log_prices = predictions.float() * target_std + target_mean
    true_log_prices = targets.float() * target_std + target_mean
    predicted_prices = torch.expm1(predicted_log_prices)
    true_prices = torch.expm1(true_log_prices)
    return F.smooth_l1_loss(
        predicted_prices / price_scale,
        true_prices / price_scale,
        beta=25.0 / price_scale,
    )


def evaluate(model, loader, device, target_mean: float, target_std: float) -> float:
    model.eval()
    use_cuda = device.type == "cuda"
    absolute_error = 0.0
    examples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_cuda
            ):
                predictions = model(features)
            # Price conversion is more numerically stable in float32.
            predictions = predictions.float()
            predicted_prices = torch.expm1(predictions * target_std + target_mean).clamp_min(0)
            true_prices = torch.expm1(targets * target_std + target_mean).clamp_min(0)
            absolute_error += torch.abs(predicted_prices - true_prices).sum().item()
            examples += len(features)
    return absolute_error / max(examples, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a local text-to-price neural network; no pretrained .pth download is needed."
    )
    parser.add_argument("--dataset", default="ed-donner/items_full")
    parser.add_argument("--output", type=Path, default=Path("artifacts/deep_neural_network.pth"))
    parser.add_argument(
        "--profile",
        choices=TRAINING_PROFILES,
        default="improved",
        help="Use the improved defaults, or legacy to reproduce the previous trainer.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--input-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--loss", choices=("log-mse", "dollar-huber"))
    parser.add_argument(
        "--patience",
        type=int,
        help="Stop after this many non-improving epochs; 0 disables early stopping.",
    )
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=20_000)
    args = parser.parse_args()
    profile = TRAINING_PROFILES[args.profile]
    for name in (
        "epochs",
        "batch_size",
        "learning_rate",
        "input_size",
        "hidden_size",
        "num_layers",
        "dropout",
        "loss",
        "patience",
    ):
        if getattr(args, name) is None:
            setattr(args, name, profile[name])
    return args


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(f"Loading {args.dataset} ...")
    train_items, validation_items, _ = Item.from_hub(args.dataset)
    if args.max_train is not None:
        train_items = train_items[: args.max_train]
    if args.max_validation is not None:
        validation_items = validation_items[: args.max_validation]

    train_items = [
        item for item in train_items if item.summary and math.isfinite(item.price) and item.price >= 0
    ]
    validation_items = [
        item
        for item in validation_items
        if item.summary and math.isfinite(item.price) and item.price >= 0
    ]
    log_prices = np.log1p(np.asarray([item.price for item in train_items], dtype=np.float64))
    target_mean = float(log_prices.mean())
    target_std = float(log_prices.std())
    if not math.isfinite(target_std) or target_std == 0:
        raise ValueError("Training prices must have a non-zero finite standard deviation.")

    vectorizer_config = {
        "n_features": args.input_size,
        **TRAINING_PROFILES[args.profile]["vectorizer"],
    }
    vectorizer = HashingVectorizer(**vectorizer_config)
    collate_fn = make_collate_fn(vectorizer)
    train_dataset = ProductDataset(train_items, target_mean, target_std)
    validation_dataset = ProductDataset(validation_items, target_mean, target_std)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Portable on Windows and safe for notebook environments.
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model_config = {
        "input_size": args.input_size,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "dropout_prob": args.dropout,
    }
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        # Let Ampere and newer NVIDIA GPUs use their fast TensorFloat-32 paths.
        torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"CUDA detected: all neural-network computation will use {gpu_name}.")
    else:
        print("CUDA was not detected; neural-network computation will use the CPU.")

    model = DeepNeuralNetwork(**model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
        fused=use_cuda,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_mae = float("inf")
    epochs_without_improvement = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Training profile={args.profile}, loss={args.loss}; "
        f"{len(train_dataset):,} examples on {device}; "
        f"validating on {len(validation_dataset):,}."
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for features, targets in progress:
            features = features.to(device, non_blocking=use_cuda)
            targets = targets.to(device, non_blocking=use_cuda)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_cuda
            ):
                predictions = model(features)
            loss = calculate_training_loss(
                predictions, targets, args.loss, target_mean, target_std
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * len(features)
            progress.set_postfix(
                loss=f"{running_loss / max(progress.n * args.batch_size, 1):.4f}"
            )

        scheduler.step()
        validation_mae = evaluate(model, validation_loader, device, target_mean, target_std)
        training_loss = running_loss / len(train_dataset)
        print(
            f"Epoch {epoch}: train loss={training_loss:.4f}, "
            f"validation MAE=${validation_mae:,.2f}"
        )

        if validation_mae < best_mae:
            best_mae = validation_mae
            torch.save(
                {
                    "format_version": 2,
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "vectorizer_config": vectorizer_config,
                    "target_mean": target_mean,
                    "target_std": target_std,
                    "dataset": args.dataset,
                    "training_profile": args.profile,
                    "training_loss": args.loss,
                    "validation_mae": validation_mae,
                },
                args.output,
            )
            print(f"Saved improved checkpoint to {args.output}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience and epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping after {args.patience} epochs without "
                    "validation MAE improvement."
                )
                break

    print(f"Done. Best validation MAE: ${best_mae:,.2f}")


if __name__ == "__main__":
    main()
