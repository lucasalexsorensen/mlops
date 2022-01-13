import sys, os
sys.path.append(os.path.dirname(__file__) + '/../data')
from data import MaskDataset
from torch.utils.data import DataLoader
import kornia as K
import torch.nn as nn
import torchvision
import torch
import wandb
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import argparse
import torchmetrics

def main():
    parser = argparse.ArgumentParser(description="Script for fitting model")
    parser.add_argument("--lr", default=0.00015, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--dropout_attn", default=0.0, type=float)
    parser.add_argument("--dropout_rate", default=0.4, type=float)
    args = parser.parse_args()
    wandb.init(project="mlops", job_type="train_model", config=args)
    config = wandb.config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE", device)

    # num_epochs = 50
    train_set = MaskDataset(root_dir="data/processed/train")
    val_set = MaskDataset(root_dir="data/processed/test")
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=True)

    augmentations = nn.Sequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomErasing(),
        K.augmentation.RandomAffine(degrees=15.0),
        K.augmentation.PatchSequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.8),
            grid_size=(
                64 // config.patch_size,
                64 // config.patch_size,
            ),  # cifar-10 is 64x64 and vit is patch 16
            patchwise_apply=False,
        ),
    ).to(device)

    model = nn.Sequential(
        K.contrib.VisionTransformer(
            image_size=64,
            embed_dim=config.embed_dim,
            patch_size=config.patch_size,
            depth=config.depth,
            num_heads=config.num_heads,
            dropout_attn=config.dropout_attn,
            dropout_rate=config.dropout_rate,
        ),
        K.contrib.ClassificationHead(embed_size=config.embed_dim, num_classes=2),
    ).to(device)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    mode = "min"
    scores = np.array([])
    patience = 3

    for epoch in range(50):
        print("========= EPOCH %d =========" % epoch)
        model.train()
        train_loss = 0
        for i, data in enumerate(train_loader):
            x, y = data[0].to(device), data[1].to(device)
            x = augmentations(x)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print("Train loss: %.3f" % train_loss)
        wandb.log({"train_loss": train_loss})

        model.eval()
        val_accuracy = torchmetrics.Accuracy().to(device)
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                x, y = data[0].to(device), data[1].to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                val_accuracy(y_hat, y.long())
                val_loss += loss.item()
        print("Validation loss: %.3f" % val_loss)
        print("Validation acc: %.3f" % val_accuracy.compute())
        wandb.log({"val_acc": val_accuracy.compute(), "val_loss": val_loss})

        scores = np.append(scores, [val_loss])
        # check for early stopping
        if np.sum(np.diff(scores[-(patience - 1) :]) > 0) == patience:
            break


if __name__ == "__main__":
    main()
