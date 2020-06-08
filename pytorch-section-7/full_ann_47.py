import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('NYCTaxiFares.csv')

# print(df.head())
# print(df['fare_amount'].describe())
# print(df.columns)


def haversine_distance(df, lat1, long1, lat2, long2):
    """
        Calculate the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])

    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lamda = np.radians(df[long2] - df[long1])

    a = np.sin(delta_phi/2) ** 2 + np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lamda / 2) * np.sin(delta_lamda / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers

    return d


df['dist_km'] = haversine_distance(
    df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')

# print(df.head())
# print(df.info())

df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
# Eastern date time
df['EDTdate'] = df['pickup_datetime'] - pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour'] < 12, 'am', 'pm')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")

# print(df.info())
# print(df.head())
# my_time = df['pickup_datetime'][0]
# print(my_time.hour)
# print(df.columns)

# categorical columns
cat_cols = ['Hour', 'AMorPM', 'Weekday']
# continuous columns
cont_cols = [
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'dist_km'
]

y_col = ['fare_amount']
# print(df.dtypes)

for cat in cat_cols:
    df[cat] = df[cat].astype('category')
# print(df.dtypes)
# print(df['Hour'].head())
# print(df['AMorPM'].head())
# print(df['Weekday'].head())
# print(df['AMorPM'].cat.categories)
# print(df['Weekday'].cat.categories)
# print(df['Hour'].cat.categories)


hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

# stack 2 numpy array theo chiều dọc
cats = np.stack([hr, ampm, wkdy], axis=1)
# print(hr)
# print(ampm)
# print(wkdy)
# print(cats)
cats = torch.tensor(cats, dtype=torch.int64)
# print(cats)

# another way to make cats
# cats = np.stack([df[col].cat.codes.values for col in cat_cols], axis=1)

conts = np.stack([df[col].values for col in cont_cols], axis=1)
conts = torch.tensor(conts, dtype=torch.float)
# print(conts)

y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1, 1)
# print(y)

# print(cats.shape)
# print(conts.shape)
# print(y.shape)

cat_szs = [len(df[col].cat.categories) for col in cat_cols]
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

# print(cat_szs)
# print(emb_szs)

# catz = cats[:2]
# selfembeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
# # print(selfembeds)

# # forward method (cats)
# embeddingz = []
# for i, e in enumerate(selfembeds):
#     embeddingz.append(e(catz[:, i]))
# # print(embeddingz)

# z = torch.cat(embeddingz, 1)
# # print(z)

# selfemdrop = nn.Dropout(0.4)
# z = selfemdrop(z)
# print(z)


class TabularModel(nn.Module):
    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super.__init__()

        self.embeds = nn.ModuleList([nn.Embedding(ni, nf)
                                     for ni, nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum([nf for ni, nf in emb_szs])
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i

        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []

        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, 1]))

        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x
