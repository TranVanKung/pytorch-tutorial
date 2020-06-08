import pandas as pd

data = {
    'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB', ],
    'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', "Sarah"],
    "Sales": [200, 120, 340, 124, 243, 350]
}

df = pd.DataFrame(data)
print(df)

print(df.groupby('Company').mean())
print(df.groupby('Company').max())
print(df.groupby('Company').describe())
print(df.groupby('Company').describe().transpose())
