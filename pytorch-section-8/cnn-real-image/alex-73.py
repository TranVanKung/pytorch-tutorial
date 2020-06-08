import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
from PIL import Image
from IPython.display import display
import warnings

# pretrain network
AlexNetmodel = models.alexnet(pretrained=True)
print(AlexNetmodel)

for param in AlexNetmodel.parameters():
    param.requires_grad = False

torch.manual_seed(42)
AlexNetmodel.classifier = nn.Sequential(
    nn.Linear(9216, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 2), nn.LogSoftmax(dim=1))
print(AlexNetmodel)

for param in AlexNetmodel.parameters():
    print(param.numel())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(AlexNetmodel.classifier.parameters(), lr=0.001)

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = '../Data/CATS_DOGS/'
train_data = datasets.ImageFolder(os.path.join(
    root, 'train'), transform=train_transform)
test_data = datasets.ImageFolder(os.path.join(
    root, 'test'), transform=test_transform)
torch.manual_seed(42)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

start_time = time.time()
epochs = 1
# limnit on number of batches
max_test_batch = 300
max_train_batch = 800  # 1 batch contain 10 images -> 18000 images total

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_corr = 0
    test_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        if b == max_train_batch:
            break
        b += 1
        y_pred = AlexNetmodel(X_train)
        loss = criterion(y_pred, y_train)

        # totally the number of correct predictions
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        train_corr += batch_corr

        # update the paramters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 100 == 0:
            print(
                f'epoch: {i:2} batch: {b:4} [{b*10:6}/{len(train_data)}] loss: {loss.item():10.8f} accuracy: {train_corr.item() * 100/(10*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(train_corr)

    # test set
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_test_batch:
                break

            y_val = AlexNetmodel(X_test)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_test).sum()
            test_corr += batch_corr

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)

total_time = time.time() - start_time
print(f'Total time: {total_time/60} minutes')

# save model
torch.save(AlexNetmodel.state_dict(), 'pretrain_model.pt')

print(f'{test_correct[-1].item()}/{len(test_data)}')

inv_nomalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.223, 1/0.225])
image_index = 2019
im = inv_nomalize(test_data[image_index][0])
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.show()

class_names = train_data.classes
AlexNetmodel.eval()
with torch.no_grad():
    new_pred = AlexNetmodel(
        test_data[image_index][0].view(1, 3, 224, 224)).argmax()
print(class_names[new_pred.item()])
