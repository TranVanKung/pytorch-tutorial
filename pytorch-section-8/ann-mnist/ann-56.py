import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# MNIST image to tensor
transform = transforms.ToTensor()
train_data = datasets.MNIST(
    root='../Data', train=True, transform=transform, download=False)
test_data = datasets.MNIST(root='../Data', train=False,
                           download=False, transform=transform)
print(train_data)
print(test_data)
print(type(train_data))
# print(train_data[0])
print(type(train_data[0]))

image, label = train_data[0]
print(image.shape, label)

# plt.imshow(image.reshape((28, 28)), cmap='gist_yarg')  # 'virdis' or 'gray'
# plt.show()

torch.manual_seed(101)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=500, shuffle=False)

np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))  # FORMATTING

# FIRST BATCH
for images, labels in train_loader:
    break

print(images.shape)
print(labels.shape)

# print the first 12 labels
print('Labels: ', labels[:12].numpy())

# print the first 12 images
im = make_grid(images[:12], nrow=12)  # the default nrow is 8
# print(im[0][0])
print(im.shape)
# plt.figure(figsize=(10, 4))

# we need to transpose the images from C, H, H to W, H, C
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()


# Model
class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz=784, out_sz=10, layers=[120, 84]):
        super().__init__()

        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)  # multi class classification


torch.manual_seed(101)
model = MultilayerPerceptron()
print(model)

# ANN -> CNN
for param in model.parameters():
    print(param.numel())
# -> 105, 214 total parameters

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
images.view(100, -1)
print(images.view(100, -1).shape)

start_time = time.time()

# traning
epochs = 10

# trackers
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_corr = 0
    test_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        # 10 neurons
        y_pred = model(X_train.view(100, -1))
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]

        batch_corr = (predicted == y_train).sum()
        train_corr += batch_corr  # return a tensor

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 200 == 0:
            accuracy = train_corr.item()*100/(100*b)
            print(
                f'Epoch {i} batch: {b} loss: {loss.item()} accuracy: {accuracy}')

    train_losses.append(loss)
    train_correct.append(train_corr)

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(500, -1))
            predicted = torch.max(y_val.data, 1)[1]
            test_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)

total_time = time.time() - start_time
print(f'Duration: {total_time/60} mins')

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label="Test/ Validation loss")
plt.legend()
plt.show()

train_acc = [t/600 for t in train_correct]
test_acc = [t/100 for t in test_correct]

plt.plot(train_acc, label="Train acc")
plt.plot(test_acc, label='Test acc')
plt.legend()
plt.show()

# new unseen data
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test.view(len(X_test), -1))
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()
    acc = 100 * correct.item()/len(test_load_all)
    cf_maxtrix = confusion_matrix(predicted.view(-1), y_test.view(-1))
    print(cf_maxtrix)
