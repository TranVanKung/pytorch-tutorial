import pandas as pd

df = pd.DataFrame({
    'col1': [1, 2, 3, 4],
    'col2': [444, 555, 666, 444],
    'col3': ['abc', 'def', 'ghi', 'xyz']
})
print(df)

print(df['col2'].unique())
print(df['col2'].value_counts())

print(df[(df['col1'] > 2) & (df['col2'] == 444)])


def times_two(number):
    return number*2


print(df['col1'].apply(times_two))

df['new'] = df['col1'].apply(times_two)
print(df)

del df['new']
print(df)
print(df.columns)
print(df.index)
print(df.info())
print(df.describe())
print(df.sort_values('col2', ascending=False))
