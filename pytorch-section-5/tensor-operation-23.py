import torch
import numpy as np

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(a.dot(b))

x = torch.tensor([[0, 2, 4], [1, 3, 5]])
y = torch.tensor([[6, 7], [8, 9], [10, 11]])

# torch.mm() is equal to @
print(torch.mm(x, y))
print(x @ y)

z = torch.tensor([3.0, 4.0])
print(z.norm())
print(z.numel())

z.apply_(lambda x: x ** 2)
print(z)
