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
from torchsummary import summary


if __name__ == '__main__':
    wandb.init(project='mlops')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('USING DEVICE', device)

    num_epochs = 50
    lr = 1e-4
    embed_dim = 256

    batch_size = 32

    train_set = MaskDataset(root_dir="data/processed/train")
    val_set = MaskDataset(root_dir="data/processed/test")
    train_loader = DataLoader(train_set, batch_size=batch_size)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    model = nn.Sequential(
    K.contrib.VisionTransformer(image_size=64, dropout_attn=0.1, dropout_rate=0.4, embed_dim=embed_dim, patch_size=8),
    K.contrib.ClassificationHead(embed_size=embed_dim, num_classes=2),
    ).to(device)
    summary(model, input_size=(3,64,64))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    _augmentations = nn.Sequential(
        K.augmentation.RandomHorizontalFlip(p=0.75),
        K.augmentation.RandomVerticalFlip(p=0.75),
        K.augmentation.RandomAffine(degrees=10.),
        K.augmentation.PatchSequential(
            K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=0.8),
            grid_size=(4, 4),  # cifar-10 is 32x32 and vit is patch 16
            patchwise_apply=False,
        ),
    )

    for epoch in range(num_epochs):
        print('========= EPOCH %d =========' % epoch)
        epoch_loss = 0.0
        model.train()
        for i,data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            inputs = _augmentations(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        #print('Epoch loss: %.3f' % epoch_loss)
        wandb.log({'train_loss': epoch_loss})

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for i,data in enumerate(val_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = _augmentations(inputs)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, top_outputs = outputs.topk(1, dim=1)
                _, top_labels = labels.topk(1, dim=1)
                equals = top_outputs == top_labels
                val_total += inputs.shape[0]
                val_correct += torch.sum(equals)
                val_loss += loss.item()
        val_acc = val_correct / val_total
        #print('Validation acc: %.3f' % val_acc)
        #print('Validation loss: %.3f' % val_loss)
        wandb.log({'val_acc': val_acc, 'val_loss': val_loss})