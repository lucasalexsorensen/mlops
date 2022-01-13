from ..data import MaskDataset
from torch.utils.data import DataLoader
import kornia as K
import torch.nn as nn
import torchvision
import torch
import wandb
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
from .model import make_model
import argparse
import json
import torchmetrics


def main():
    parser = argparse.ArgumentParser(description="Script for fitting model")
    parser.add_argument("output_path", type=str)
    parser.add_argument("test_data", type=str)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("USING DEVICE", device)

    test_set = MaskDataset(root_dir=args.test_data)
    test_loader = DataLoader(test_set, batch_size=64)

    saved = torch.load(args.output_path)

    config, weights = saved["config"], saved["weights"]
    model = make_model(config).to(device)
    model.load_state_dict(weights)
    model.eval()
    test_accuracy = torchmetrics.Accuracy().to(device)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = data[0].to(device), data[1].to(device)
            y_hat = model(x)
            test_accuracy(y_hat, y.long())

    metrics = {}
    metrics["test_accuracy"] = float(test_accuracy.compute())

    print("METRICS")
    print(metrics)
    with open("outputs/metrics.json", "w") as f:
        f.write(json.dumps(metrics))


if __name__ == "__main__":
    main()
