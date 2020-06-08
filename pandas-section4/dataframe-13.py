import pandas as pd
import numpy as np
from numpy.random import randn

np.random.seed(101)
rand_mat = randn(4, 4)

print(rand_mat)
df = pd.DataFrame(data=rand_mat, index='A B C D'.split(),
                  columns='W X Y Z'.split())
print(df)
print(df > 0)
df_bool = df > 0
print(df[df_bool])

print(df['W'] > 0)
print(df[df['W'] > 0]['Y'].loc['A'])

cond1 = df['W'] > 0
cond2 = df['Y'] > 0

print(df[(cond1) & (cond2)])

df.reset_index(inplace=True)
print(df)

new_ind = 'CA NY WY OR'.split()
df['States'] = new_ind
print(df)

df.set_index('States', inplace=True)
print(df)

print(df.info())
print(df.dtypes)
print(df.describe())

ser_w = df['W'] > 0
print(ser_w.value_counts())
print(sum(ser_w))
print(len(ser_w))
