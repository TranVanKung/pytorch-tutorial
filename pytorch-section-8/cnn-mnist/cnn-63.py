import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.MNIST(
    root="../Data", train=True, download=False, transform=transform)
test_data = datasets.MNIST(root='../Data', train=False,
                           download=False, transform=transform)
print(train_data)
print(test_data)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# 1 color channel, 6 filters(output channels), 3 by 3 kernel, stride = 1
# conv1 = nn.Conv2d(1, 6, 3, 1)  # 6 filters -> pooling -> conv2
# # 6 input filters conv1, 16 filters, 3 by 3, stride=1
# conv2 = nn.Conv2d(6, 16, 3, 1)
# for i, (X_train, y_train) in enumerate(train_data):
#     break
# x = X_train.view(1, 1, 28, 28)  # -> 4D batch (batch of 1 image)
# print(x)
# x = F.relu(conv1(x))
# print(x.shape)
# x = F.max_pool2d(x, 2, 2)
# print(x.shape)
# x = F.relu(conv2(x))
# print(x.shape)
# x = F.max_pool2d(x, 2, 2)
# print(x.shape)
# x = x.view(-1, 16*5*5)
# print(x.shape)


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


torch.manual_seed(42)
model = ConvolutionalNetwork()
print(model)

for param in model.parameters():
    print(param.numel())

# --> 60074 parameters
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
epochs = 2
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_corr = 0
    test_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        train_corr += batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 600 == 0:
            print(f'Epoch: {i} batch: {b} loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(train_corr)

    # test
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            test_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)

current_time = time.time()
total = current_time - start_time
print(f'Trainging took {total/60} minutes')

plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="validation loss")
plt.title('Loss at epoch')
plt.legend()
plt.show()

plt.plot([t/600 for t in train_correct], label="training accuracy")
plt.plot([t/100 for t in test_correct], label="validation accuracy")
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()

test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)
with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

print(correct.item()/len(test_data))

# load a single example
sample_data = test_data[1997][0].reshape(28, 28)
plt.imshow(sample_data)
plt.show()

model.eval()
with torch.no_grad():
    new_prediction = model(test_data[1997][0].view(1, 1, 28, 28))

print(new_prediction.argmax())
