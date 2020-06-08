import torch
import numpy as np

new_arr = np.array([1, 2, 3])
print(new_arr.dtype)

# tensor() preserve type of new_arr
# Tensor() convert type of new_arr to float = FloatTensor()
print(torch.tensor(new_arr))
print(torch.Tensor(new_arr))
print(torch.FloatTensor(new_arr))

print(torch.empty(4, 2))
print(torch.zeros(4, 2, dtype=torch.int64))
print(torch.ones(4, 2, dtype=torch.int64))
print(torch.arange(0, 18, 2).reshape(3, 3))
print(torch.linspace(1, 18, 12).reshape(3, 4))
print(torch.tensor([1, 2, 3]))

my_tensor = torch.tensor([1, 2, 3])
print(my_tensor.dtype)
my_tensor = my_tensor.type(torch.int32)
print(my_tensor.dtype)

print(torch.rand(4, 3))
print(torch.randint(low=0, high=10, size=(5, 5)))

x = torch.zeros(2, 5)
print(x)
print(torch.rand_like(x))
print(torch.randn_like(x))
print(torch.randint_like(x, low=0, high=11))

torch.manual_seed(42)
print(torch.rand(2, 3))
