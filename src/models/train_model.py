import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))
from data import MaskDataset
from torch.utils.data import DataLoader
import kornia as K
from kornia.x import ImageClassifierTrainer, ModelCheckpoint, Configuration
import torch.nn as nn
import torchvision
import torch
import wandb
import pandas as pd
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import argparse
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM
import torchmetrics


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
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
        )
        self.val_accuracy = torchmetrics.Accuracy()
        self.config = config
        self.augmentations = nn.Sequential(
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
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        x = self.augmentations(x)
        outputs = self.model(x)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss / batch.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        self.val_accuracy(y_hat, y)
        return F.cross_entropy(y_hat, y)

    def validation_epoch_end(self, out):
        self.log("val_acc", self.val_accuracy)


def main():
    parser = argparse.ArgumentParser(description="Script for fitting model")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--patch_size", default=8, type=int)
    parser.add_argument("--depth", default=12, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--dropout_attn", default=0.0, type=float)
    parser.add_argument("--dropout_rate", default=0.4)
    args = parser.parse_args()
    wandb.init(project="mlops", job_type="train_model", config=args)
    config = wandb.config

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE", device)

    num_epochs = 50
    train_set = MaskDataset(root_dir="data/processed/train")
    val_set = MaskDataset(root_dir="data/processed/test")
    train_loader = DataLoader(train_set, batch_size=config.batch_size)
    val_loader = DataLoader(val_set, batch_size=config.batch_size)

    model = Model(config)

    # for epoch in range(num_epochs):
    #     print("========= EPOCH %d =========" % epoch)
    #     epoch_loss = 0.0
    #     model.train()
    #     for i, data in enumerate(train_loader):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         inputs = _augmentations(inputs)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item() / data.shape[0]
    #     print("Epoch loss: %.3f" % epoch_loss)
    #     wandb.log({"train_loss": epoch_loss})

    #     model.eval()
    #     val_correct = 0
    #     val_total = 0
    #     val_loss = 0.0
    #     with torch.no_grad():
    #         for i, data in enumerate(val_loader):
    #             inputs, labels = data[0].to(device), data[1].to(device)
    #             inputs = _augmentations(inputs)
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #             _, top_outputs = outputs.topk(1, dim=1)
    #             _, top_labels = labels.topk(1, dim=1)
    #             equals = top_outputs == top_labels
    #             val_total += inputs.shape[0]
    #             val_correct += torch.sum(equals)
    #             val_loss += loss.item()
    #     val_acc = val_correct / val_total
    #     print("Validation loss: %.3f" % val_loss)
    #     print("Validation acc: %.3f" % val_acc)
    #     wandb.log({"val_acc": val_acc, "val_loss": val_loss})


if __name__ == "__main__":
    main()
