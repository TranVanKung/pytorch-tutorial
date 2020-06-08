import numpy as np

arr = np.arange(0, 11)
print(arr)

print(arr[8])
print(arr[0:5])
print(arr[:5])
print(arr[5:])

# broadcast numpy array
print(arr + 100)
print(arr ** 2)

slice_of_array = arr[0:6]
print(slice_of_array)
slice_of_array[:] = 99
print(slice_of_array)
print(arr)

arr_copy = arr.copy()
arr_copy[:] = 1000
print(arr_copy)
print(arr)

# indexing on 2d array
arr_2d = np.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(arr_2d)
print(arr_2d.shape)
print(arr_2d[1])
print(arr_2d[1][1])
print(arr_2d[1, 1])

print(arr_2d[:2, 1:])


# conditional selection
arr1 = np.arange(1, 11)
print(arr1)
print(arr1 > 4)
print(arr1[arr1 > 4])
