import numpy as np

# numpy operation
arr = np.arange(0, 10)

print(arr + 100)
print(np.sin(arr))
print(np.sqrt(arr))
print(np.log(arr))
print(np.sum(arr))
print(np.min(arr))
print(np.max(arr))

arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(arr_2d.sum(axis=0)) 
print(arr_2d.sum(axis=1))
