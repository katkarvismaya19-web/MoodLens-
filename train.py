"""
MoodLens training: download FER-2013 and train a CNN from scratch in PyTorch.
Run on Google Colab with a GPU (the notebook does this for you).
Needs kaggle.json in this folder or ~/.kaggle/. EPOCHS env var changes epochs (default 40).
"""
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import IMG_SIZE, EmotionCNN

DATASET = "msambare/fer2013"     # 48x48 grayscale faces, folders train/ and test/
RAW = Path("data/fer2013")
OUT = Path("models")
EPOCHS = int(os.environ.get("EPOCHS", 40))
BATCH = 128


def setup_kaggle():
    kdir = Path.home() / ".kaggle"
    kdir.mkdir(exist_ok=True)
    target = kdir / "kaggle.json"
    if not target.exists() and Path("kaggle.json").exists():
        shutil.copy("kaggle.json", target)
    if not target.exists():
        raise SystemExit("kaggle.json not found. Put it in this folder and run again.")
    os.chmod(target, 0o600)


def download():
    if (RAW / "train").exists():
        print("Dataset already downloaded.")
        return
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET,
                    "-p", str(RAW), "--unzip"], check=True)


def loaders():
    norm = transforms.Normalize([0.5], [0.5])
    train_tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([
        transforms.Grayscale(1), transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(), norm])
    train_ds = datasets.ImageFolder(RAW / "train", train_tf)
    test_ds = datasets.ImageFolder(RAW / "test", test_tf)
    print(f"Classes: {train_ds.classes}")
    print(f"Train images: {len(train_ds)}, Test images: {len(test_ds)}")
    kw = dict(batch_size=BATCH, num_workers=2, pin_memory=torch.cuda.is_available())
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(test_ds, shuffle=False, **kw), train_ds.classes)


def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, preds, targets = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            preds.append(out.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return loss_sum / len(targets), (preds == targets).mean(), preds, targets


def plot_history(hist):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, [h["train_loss"] for h in hist], label="Train")
    ax[0].plot(ep, [h["val_loss"] for h in hist], label="Validation")
    ax[0].set(title="Loss", xlabel="Epoch")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="Train")
    ax[1].plot(ep, [h["val_acc"] for h in hist], label="Validation")
    ax[1].set(title="Accuracy", xlabel="Epoch")
    for a in ax:
        a.legend()
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "graphs" / "training_curves.png", dpi=120)
    plt.close(fig)


def plot_confusion(cm, classes):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (normalised)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "graphs" / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_dl, test_dl, classes = loaders()
    (OUT / "graphs").mkdir(parents=True, exist_ok=True)

    model = EmotionCNN(len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, epochs=EPOCHS, steps_per_epoch=len(train_dl))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_acc, hist = 0.0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)

        val_loss, val_acc, _, _ = evaluate(model, test_dl, device, criterion)
        hist.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_acc": correct / seen,
                     "val_loss": val_loss, "val_acc": float(val_acc)})
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes},
                       OUT / "emotion_cnn.pt")
            flag = "  <- best, saved"
        print(f"Epoch {epoch:2d}/{EPOCHS} | train loss {loss_sum / seen:.3f} "
              f"acc {correct / seen:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f} "
              f"| {time.time() - t0:.0f}s{flag}")

    # final evaluation with the best weights
    ckpt = torch.load(OUT / "emotion_cnn.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    _, acc, preds, targets = evaluate(model, test_dl, device, criterion)
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    per_class = {c: round(float(cm[i, i] / max(cm[i].sum(), 1)), 4) for i, c in enumerate(classes)}

    with open(OUT / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hist[0].keys())
        w.writeheader()
        w.writerows(hist)
    plot_history(hist)
    plot_confusion(cm, classes)
    metrics = {"test_accuracy": round(float(acc), 4), "per_class_accuracy": per_class,
               "epochs": EPOCHS, "parameters": n_params, "classes": classes,
               "train_images": len(train_dl.dataset), "test_images": len(test_dl.dataset)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"\nDone! Best test accuracy: {acc:.2%}")
    for c, a in per_class.items():
        print(f"  {c:9s} {a:.2%}")
    print("Model saved to models/emotion_cnn.pt")


if __name__ == "__main__":
    setup_kaggle()
    download()
    main()
