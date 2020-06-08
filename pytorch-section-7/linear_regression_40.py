import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

X = torch.linspace(1, 50, 50).reshape(-1, 1)
print(X)

torch.manual_seed(71)
e = torch.randint(-8, 9, (50, 1), dtype=torch.float)
print(e)

y = 2*X + 1 + e
print(y.shape)

print(X.numpy())
# plt.scatter(X.numpy(), y.numpy())
# plt.show()

torch.manual_seed(59)
model = nn.Linear(in_features=1, out_features=1)
print(model.weight)
print(model.bias)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


torch.manual_seed(59)
model = Model(1, 1)
print(model.linear.weight)
print(model.linear.bias)

for name, param in model.named_parameters():
    print(name, '\t', param.item())

x = torch.tensor([2.0])
print(model.forward(x))

x1 = np.linspace(0.0, 50.0, 50)
print(x1)

w1 = 0.1059
b1 = 0.9637

y1 = w1 * x1 + b1
# plt.scatter(X.numpy(), y.numpy())
# plt.plot(x1, y1)
# plt.show()

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epochs = 50
losses = []

for i in range(epochs):
    i = i + 1

    # prediction on the foraward pass
    y_pred = model.forward(X)

    # calculate the loss
    loss = criterion(y_pred, y)

    # record that error
    losses.append(loss)

    print(
        f'epoch {i} loss: {loss.item()} weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# plt.plot(range(epochs), losses)
# plt.ylabel('MSE LOSS')
# plt.xlabel('Epoch')
# plt.show()

x = np.linspace(0.0, 50.0, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()
predicted_bias = model.linear.bias.item()
predicted_y = current_weight*x + current_bias
print(x)
print(predicted_y)

plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y, 'r')
plt.show()
