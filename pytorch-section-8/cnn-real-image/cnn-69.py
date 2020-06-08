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

warnings.filterwarnings('ignore')

# check the full file path

with Image.open('../Data/CATS_DOGS/test/CAT/10107.jpg') as im:
    display(im)

path = '../Data/CATS_DOGS/'
img_names = []

for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '/' + img)

print(len(img_names))
print(img_names[0])

img_sizes = []
rejected = []

for item in img_names:
    try:
        with Image.open(item) as img:
            img_sizes.append(img.size)
    except:
        rejected.append(item)

print(len(rejected))
print(len(img_sizes))

df = pd.DataFrame(img_sizes)
print(df.head())
print(df[0].describe())

# read size of image
dog = Image.open('../Data/CATS_DOGS/train/DOG/14.jpg')
display(dog)
print(dog.size)
print(dog.getpixel((0, 0)))

# display image
img_dog = mpimg.imread('../Data/CATS_DOGS/train/DOG/14.jpg')
# plt.figure()
# plt.imshow(img_dog)
# plt.show()

transform = transforms.Compose([
    # transforms.RandomRotation(30),    transforms.RandomHorizontalFlip(p=1),
    # transforms.Resize((250, 250)),
    # transforms.CenterCrop((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

im = transform(dog)
print(type(im))
print(im.shape)

# plt.show()
print(im[:, 0, 0])
print(type(dog.getpixel((0, 0))))
print(np.array(dog.getpixel((0, 0))) / 255)

# (width, height, channel)
img = np.transpose(im.numpy(), (1, 2, 0))
# plt.imshow(img)
# plt.show()
# print(im)

inv_nomalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.223, 1/0.225])
im_inv = inv_nomalize(im)
# plt.figure()
# plt.imshow(np.transpose(im_inv.numpy(), (1, 2, 0)))
# plt.show()


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

class_names = train_data.classes
print(class_names)
print(len(train_data))
print(len(test_data))

for images, labels in train_loader:
    break
print(images.shape)

img = make_grid(images, nrow=5)
img_inv = inv_nomalize(img)
plt.figure()
print(img_inv.shape)
plt.imshow(np.transpose(img_inv.numpy(), (1, 2, 0)))
# plt.show()


class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


torch.manual_seed(101)
CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.001)
print(CNNmodel)

for param in CNNmodel.parameters():
    print(param.numel())

start_time = time.time()
epochs = 3
# limnit on number of batches
max_test_batch = 500
max_train_batch = 1800  # 1 batch contain 10 images -> 18000 images total

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
        y_pred = CNNmodel(X_train)
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
                f'epoch: {i} batch: {b*10}/{len(train_data)} loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(train_corr)

    # test set
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            if b == max_test_batch:
                break

            y_val = CNNmodel(X_test)
            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_test).sum()
            test_corr += batch_corr

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_corr)

total_time = time.time() - start_time
print(f'Total time: {total_time/60} minutes')

# save model
torch.save(CNNmodel.state_dict(), 'my_model.pt')

# visualize
plt.plot(train_losses, label="training loss")
plt.plot(test_losses, label="validation loss")
plt.title('Loss at the and of each epoch')
plt.legend()
plt.show()

plt.plot([t/80 for t in train_correct], label="training accuracy")
plt.plot([t/30 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend()
plt.show()
