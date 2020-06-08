import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv('iris.csv')
print(df, df.shape)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
fig.tight_layout()

plots = [(0, 1), (2, 3), (0, 2), (1, 3)]
color = ['b', 'r', 'g']
labels = ['Iris setosa', 'Iris virginica', 'Iris versicolor']

for i, ax in enumerate(axes.flat):
    for j in range(3):
        x = df.columns[plots[i][0]]
        y = df.columns[plots[i][1]]
        ax.scatter(df[df['target'] == j][x],
                   df[df['target'] == j][y], color=color[j])
        ax.set(xlabel=x, ylabel=y)

fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0, 0.85))
# plt.show()

features = df.drop('target', axis=1).values
label = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(
    features, label, test_size=0.2, random_state=33)

# print(X_train)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train).reshape(-1, 1)
y_test = torch.LongTensor(y_test).reshape(-1, 1)
# print(y_test)

# using tensorDataset, DataLoader
data = df.drop('target', axis=1).values
labels = df['target'].values

iris = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
print(iris, len(iris))

# for i in iris:
#     print(i)

iris_loader = DataLoader(iris, batch_size=50, shuffle=True)
for i_batch, sample_batch in enumerate(iris_loader):
    print(i_batch, sample_batch)
