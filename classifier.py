import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from skimage import io, transform
import numpy as np
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from models import Classifier
import warnings
warnings.filterwarnings("ignore")

composed = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

data = ImageFolder(root='simpsons_dataset', transform=composed)

batch_size = 128

train_loader = DataLoader(data, batch_size=batch_size, num_workers=2, shuffle=True)

classifier = Classifier()

optimizer = optim.Adam(classifier.parameters(), lr=.001)
criterion = CrossEntropyLoss()
epochs = 1

n_data = len(data)

for i in range(5):
    total_loss = 0
    for j, data in enumerate(train_loader, 0):
        images, labels = data

        optimizer.zero_grad()

        outputs = classifier(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print('Epoch: {}, loss: {}'.format(i+1, total_loss))
