import pandas as pd

df = pd.read_csv('example.csv')
print(df)
print(df[['a', 'b']])

df1 = pd.read_excel('Excel_Sample.xlsx', sheet_name='Sheet1')
print(df1)
print(df1.drop('Unnamed: 0', axis=1))
