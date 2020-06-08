import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import MinMaxScaler
import time

register_matplotlib_converters()

df = pd.read_csv('./Alcohol_Sales.csv', index_col=0, parse_dates=True)
print(df.head())
print(df.tail())
print(len(df))

df = df.dropna()
print(len(df))

# df.plot(figsize=(12, 4))
# plt.show()

y = df['S4248SM144NCEN'].values.astype(float)
test_size = 12
train_set = y[:-test_size]
test_set = y[-test_size:]
print(test_set)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(train_set.reshape(-1, 1))
train_norm = scaler.transform(train_set.reshape(-1, 1))
# print(train_norm)

train_norm = torch.FloatTensor(train_norm).view(-1)
print(type(train_norm))

window_size = 12


def input_data(seq, ws):
    out = []
    L = len(seq)

    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))

    return out


train_data = input_data(train_norm, window_size)
print(len(train_data))


class LSTMnetwork(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, out_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        # add an LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size)
        # add a fully connected layer
        self.linear = nn.Linear(hidden_size, out_size)
        # (H, C), initial h0 and c0
        self.hidden = (torch.zeros(1, 1, hidden_size),
                       torch.zeros(1, 1, hidden_size))

    def forward(self, seq):
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), 1, -1), self.hidden)
        pred = self.linear(lstm_out.view(len(seq), -1))
        # [1, 2, 3] [4] -> [1, 2, 3, 4 -> 4]
        # we only want the last value
        return pred[-1]


torch.manual_seed(101)
model = LSTMnetwork()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
# epochs = 100
# start_time = time.time()

# for epoch in range(epochs):
#     for seq, y_train in train_data:
#         optimizer.zero_grad()
#         model.hidden = (torch.zeros(1, 1, model.hidden_size),
#                         torch.zeros(1, 1, model.hidden_size))
#         y_pred = model(seq)
#         loss = criterion(y_pred, y_train)
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch {epoch} Loss {loss.item()}')

# total_time = time.time() - start_time
# print(total_time)

# future = 12
# preds = train_norm[-window_size:].tolist()
# model.eval()

# for i in range(future):
#     seq = torch.FloatTensor(preds[-window_size:])
#     with torch.no_grad():
#         model.hidden = (torch.zeros(1, 1, model.hidden_size),
#                         torch.zeros(1, 1, model.hidden_size))
#         preds.append(model(seq).item())

# print(preds[window_size:])
# true_predictions = scaler.inverse_transform(
#     np.array(preds[window_size:]).reshape(-1, 1))
# print(true_predictions)

# print(df['S4248SM144NCEN'][-12:])
# x = np.arange('2018-02-01', '2019-02-01', dtype='datetime64[M]')

# print(x)
# print(df.index)

# plt.figure(figsize=(12, 4))
# plt.title('Beer, Wine, and Alcohol Sales')
# plt.ylabel('Sales (millions of dollars)')
# plt.grid(True)
# plt.autoscale(axis='x', tight=True)
# plt.plot(df['S4248SM144NCEN'])
# plt.plot(x, true_predictions)
# plt.show()

# # Select the end of the graph with slice notation:
# plt.plot(df['S4248SM144NCEN']['2017-01-01':])
# plt.plot(x, true_predictions)
# plt.show()

epochs = 100
# set the model back to training mode
model.train()
# feature scale the entire dataset
y_norm = scaler.fit_transform(y.reshape(-1, 1))
y_norm = torch.FloatTensor(y_norm).view(-1)
all_data = input_data(y_norm, window_size)

start_time = time.time()

for epoch in range(epochs):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        y_pred = model(seq)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch} Loss {loss.item()}')

total_time = time.time() - start_time
print(f'Duration: {total_time}')

future = 12
L = len(y)
preds = y_norm[-window_size:].tolist()
model.eval()

for i in range(future):
    seq = torch.FloatTensor(preds[-window_size:])
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_size),
                        torch.zeros(1, 1, model.hidden_size))
        preds.append(model(seq).item())

true_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

x = np.arange('2019-02-01', '2020-02-01', dtype='datetime64[M]')
plt.figure(figsize=(12, 4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(df['S4248SM144NCEN'])
plt.plot(x, true_predictions[window_size:])
plt.show()


fig = plt.figure(figsize=(12, 4))
plt.title('Beer, Wine, and Alcohol Sales')
plt.ylabel('Sales (millions of dollars)')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
fig.autofmt_xdate()
plt.plot(df['S4248SM144NCEN'])
plt.plot(x, true_predictions[window_size:])
plt.show()
