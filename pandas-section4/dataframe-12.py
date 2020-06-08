import pandas as pd
import numpy as np
from numpy.random import randn

np.random.seed(101)
rand_mat = randn(4, 4)

print(rand_mat)
df = pd.DataFrame(data=rand_mat, index='A B C D'.split(),
                  columns='W X Y Z'.split())
print(df)
my_list = ['X', 'Y']
print(df[my_list])


# working with column
df['NEW'] = df['W'] + df['Y']
print(df)

df.drop('NEW', axis=1, inplace=True)
print(df)

df.drop('A', axis=0, inplace=True)
print(df)

# working with row
print(df.loc[['B', 'C']])
print(df.iloc[[0, 1]])

print(df.loc[['B', 'C']][['Y', 'Z']])
