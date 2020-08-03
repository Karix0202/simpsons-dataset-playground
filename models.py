import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)

        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.linear1 = nn.Linear(in_features=128 * 26 * 26, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=42)

        self.dropout1 = nn.Dropout(p=.25)
        self.dropout2 = nn.Dropout(p=.5)

    def forward(self, x):
        # conv 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # conv 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # conv 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # flatten
        x = x.view(-1, 128 * 26 * 26)

        # linear 1
        x = self.linear1(x)
        x = F.relu(x)

        # linear 2 - output
        x = self.linear2(x)
        return x
        # return F.softmax(x, dim=1)
