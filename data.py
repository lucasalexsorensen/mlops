import numpy as np
from glob import glob
from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.transform = transform

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        im, label = self.X[idx], self.y[idx]

        if self.transform:
            im = self.transform(label)

        return im, label


def mnist(root='data/corruptmnist/*.npz'):
    X_train, y_train = None, None
    X_test, y_test = None, None

    for file in glob(root):
        D = np.load(file)
        X, y = D['images'], D['labels']

        if 'test' in file:
            X_test = X
            y_test = y
        else:
            if X_train is None:
                X_train = X
                y_train = y
            else:
                X_train = np.vstack((X_train, X))
                y_train = np.hstack([y_train, y])


    train_dataloader = DataLoader(CustomDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(CustomDataset(X_test, y_test), batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader

