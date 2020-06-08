import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
my_list = [10, 20, 30]

arr = np.array(my_list)
print(arr)

d = {'a': 10, 'b': 20, 'c': 30}
print(pd.Series(data=my_list))
print(pd.Series(data=arr, index=labels))
print(pd.Series(data=[10, 'a', 4.4]))

ser1 = pd.Series([1, 2, 3, 4], index=['USA', "Germany", 'USSR', 'Japan'])
print(ser1)
print(ser1['USA'])
ser2 = pd.Series([1, 4, 5, 6], index=['USA', 'Germany', 'Italy', 'Japan'])
print(ser1 + ser2)
