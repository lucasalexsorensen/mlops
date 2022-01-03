from argparse import ArgumentParser
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import torch.nn as nn
import sys

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x


def fit(parser: ArgumentParser, model, data: DataLoader):
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.001)
    args = parser.parse_args(sys.argv[2:])

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    for epoch in range(args.epochs):
        running_loss = 0
        model.train()
        for images, labels in data:
            optimizer.zero_grad()
            images = images.view(images.shape[0], -1)
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print('Epoch: %d - loss=%.3f' % (epoch, running_loss))



    return model