import torch
import numpy as np

print(torch.__version__)
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print(arr.dtype)

x = torch.from_numpy(arr)
print(type(x))
print(x.dtype)
print(torch.as_tensor(arr))

arr2d = np.arange(0.0, 12.0)
arr2d = arr2d.reshape(4, 3)
print(arr2d)
print(torch.from_numpy(arr2d))

arr[0] = 99
print(arr)
print(x)

# from_numpy create direct link to array
# to avoid this, use tensor() instead
my_arr = np.arange(0, 10)
my_tensor = torch.tensor(my_arr)
my_other_tensor = torch.from_numpy(my_arr)

print(my_arr)
print(my_tensor)
print(my_other_tensor)

my_arr[0] = 999
print(my_arr)
print(my_tensor)
print(my_other_tensor)
