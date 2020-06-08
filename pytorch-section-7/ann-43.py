import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        # how  many layers
        # input (4 features) -> h1 -> h2 -> output (3 classes)
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


torch.manual_seed(32)
model = Model()

df = pd.read_csv('iris.csv')
# print(df.head())
# print(df.tail())


def draw_data(mystery_iris):
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

        if mystery_iris is not None:
            ax.scatter(mystery_iris[plots[i][0]],
                       mystery_iris[plots[i][1]], color='y')

    fig.legend(labels=labels, loc=3, bbox_to_anchor=(1.0, 0.85))
    plt.show()


# draw_data(None)

X = df.drop('target', axis=1)
y = df['target']
# print(type(y))
# print(type(X))

# convert X, y to numpy array
X = X.values
y = y.values
print(type(X))
print(type(y))
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=33)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)
# print(y_train)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# print(model.parameters)
# print(len(X_train))
# print(len(X_test))

epochs = 200
losses = []

for i in range(epochs):
    # forward and get a prediction
    y_pred = model.forward(X_train)

    # calculate the loss
    loss = criterion(y_pred, y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'epoch {i} and loss is: {loss}')

    # backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# plt.plot(range(epochs), losses)
# plt.show()
# plt.ylabel('Loss')
# plt.xlabel('Epoch')

with torch.no_grad():
    y_val = model.forward(X_test)
    loss = criterion(y_val, y_test)

print(loss)
correct = 0

with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        # 1.) 2.) 3.)
        print(f'{i+1}.)  {str(y_val)}  {y_test[i]}')

        if y_val.argmax().item() == y_test[i]:
            correct += 1

print(f'We got {correct} correct')

# save model in format of dictinary
torch.save(model.state_dict(), 'my_iris_model.pt')

# save model in format of pickle file
# torch.save(model, 'my_iris_model.pt')


new_model = Model()
new_model.load_state_dict(torch.load('my_iris_model.pt'))
# print(new_model.eval())

mystery_iris = torch.tensor([5.6, 3.7, 2.2, 0.5])

# draw_data(mystery_iris)

with torch.no_grad():
    print(new_model(mystery_iris))
    print(new_model(mystery_iris).argmax())
