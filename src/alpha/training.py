"""Shared training helpers for phase-three alpha models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class TrainingHistory:
    train_losses: list[float]
    val_losses: list[float]
    best_val_loss: float


def sharpe_ratio_loss(predictions: torch.Tensor, actual_returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Negative Sharpe-ratio objective on realized strategy returns."""
    realized = predictions.reshape(-1) * actual_returns.reshape(-1)
    mean = realized.mean()
    std = realized.std(unbiased=False)
    return -(mean / (std + eps))


def temporal_train_val_split(dataset: Dataset, validation_fraction: float) -> tuple[Dataset, Dataset]:
    if len(dataset) <= 1:
        return dataset, dataset

    sample_dates = list(getattr(dataset, "sample_dates"))
    unique_dates = sorted(set(sample_dates))
    if len(unique_dates) <= 1:
        return dataset, dataset

    cutoff_idx = max(1, int(len(unique_dates) * (1.0 - validation_fraction)))
    cutoff_idx = min(cutoff_idx, len(unique_dates) - 1)
    train_dates = set(unique_dates[:cutoff_idx])
    val_dates = set(unique_dates[cutoff_idx:])

    train_indices = [idx for idx, date in enumerate(sample_dates) if date in train_dates]
    val_indices = [idx for idx, date in enumerate(sample_dates) if date in val_dates]

    if not train_indices or not val_indices:
        midpoint = max(1, len(dataset) // 2)
        train_indices = list(range(midpoint))
        val_indices = list(range(midpoint, len(dataset)))
        if not val_indices:
            val_indices = train_indices

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def train_model(
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> TrainingHistory:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_loss = float("inf")
    stale_epochs = 0
    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        batch_train_losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = sharpe_ratio_loss(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_train_losses.append(float(loss.detach().cpu()))

        model.eval()
        batch_val_losses: list[float] = []
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                predictions = model(features)
                loss = sharpe_ratio_loss(predictions, targets)
                batch_val_losses.append(float(loss.detach().cpu()))

        mean_train_loss = float(np.mean(batch_train_losses)) if batch_train_losses else 0.0
        mean_val_loss = float(np.mean(batch_val_losses)) if batch_val_losses else mean_train_loss
        train_losses.append(mean_train_loss)
        val_losses.append(mean_val_loss)
        scheduler.step(mean_val_loss)

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return TrainingHistory(train_losses=train_losses, val_losses=val_losses, best_val_loss=best_val_loss)


def predict_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if len(dataset) == 0:
        return np.array([], dtype=np.float32)

    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for features, _ in loader:
            predictions = model(features.to(device)).detach().cpu().numpy().reshape(-1)
            outputs.append(predictions.astype(np.float32))
    return np.concatenate(outputs)
