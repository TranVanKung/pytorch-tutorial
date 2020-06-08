import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import time
import numpy as np
import pandas as pd
import seaborn as sn  # for heatmaps
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
train_data = datasets.CIFAR10(
    root='../Data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(
    root='../Data', train=False, download=True, transform=transform)

print(train_data)
print(test_data)

torch.manual_seed(101)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

class_names = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for images, labels in train_loader:
    break
print(labels)

# print the labels
print('Label: ', labels.numpy())
print('Class: ', *np.array([class_names[i] for i in labels]))

# im = make_grid(images, nrow=5)
# plt.figure(figsize=(10, 4))
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(6*6*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 6*6*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)


torch.manual_seed(101)
model = ConvolutionalNetwork()
print(model)

for param in model.parameters():
    print(param.numel())

# -> 81302 parameter
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()

epochs = 10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    train_corr = 0
    test_corr = 0

    # run the traning batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1

        # apply the model
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # totaly the number of correct prediction
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        train_corr += batch_corr

        # update the parameter
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print results
        if b % 100 == 0:
            print(
                f'epoch: {i:2} batch: {b:4} [{10*b:6}/50000] loss: {loss.item():10.8f} \ accuracy: {train_corr.item()*100/(10*b):7.3f}%')

    train_losses.append(loss)
    train_correct.append(train_corr)

    # run the testing batches
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            test_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)

print(f'\nDuration: {time.time() - start_time: .0f} seconds')

# saving the model
torch.save(model.state_dict(), 'nyCIFARmodel.pt')
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label='validation loss')
plt.title('Loss at the end of each epoch')
plt.legend()
plt.show()

plt.plot([t/500 for t in train_correct], label="training accuracy")
plt.plot([t/100 for t in test_correct], label='validationa accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()

print(test_correct)

# create loader for entire test set
test_load_all = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
    correct = 0
    for X_test, y_test in test_load_all:
        y_val = model(X_test)
        predicted = torch.max(y_val, 1)[1]
        correct += (predicted == y_test).sum()

arr = confusion_matrix(y_test.view(-1), predicted.view(-1))
df_cm = pd.DataFrame(arr, class_names, class_names)
plt.figure(figsize=(9, 6))
sn.heatmap(df_cm, annot=True, fmt='d', cmap='BuGn')
plt.xlabel('prediction')
plt.ylabel('label (ground truth)')
plt.show()
