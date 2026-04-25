#!/usr/bin/env python3
"""Train a small MLP on precomputed CLIP features."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_features(path: Path):
    data = np.load(path, allow_pickle=True)
    X = torch.from_numpy(data["features"]).float()
    y = torch.from_numpy(data["labels"]).float().unsqueeze(1)
    meta = {key: data[key] for key in data.files if key not in {"features", "labels"}}
    return X, y, meta


def build_model(input_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    X_train, y_train, _ = load_features(args.train)
    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    if args.val:
        X_val, y_val, _ = load_features(args.val)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
    else:
        val_loader = None

    model = build_model(X_train.size(1), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_state = None

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)
        avg_loss = epoch_loss / len(train_ds)

        record = {"epoch": epoch, "train_loss": avg_loss}

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict()
            record.update({"val_loss": val_loss, "val_acc": val_acc})
        else:
            print(f"Epoch {epoch}: train_loss={avg_loss:.4f}")
            record.update({"val_loss": None, "val_acc": None})

        history.append(record)

    if best_state is None:
        best_state = model.state_dict()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "state_dict": best_state,
        "input_dim": X_train.size(1),
        "hidden_dim": args.hidden_dim,
        "history": history,
    }
    torch.save(checkpoint_payload, args.checkpoint)

    history_path = args.checkpoint.with_suffix(".history.json")
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)
    print(f"Saved checkpoint -> {args.checkpoint}")
    print(f"Saved training history -> {history_path}")
    print(f"Saved checkpoint -> {args.checkpoint}")


if __name__ == "__main__":
    main()
