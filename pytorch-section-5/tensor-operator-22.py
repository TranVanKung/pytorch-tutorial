import torch
import numpy as np

# indexing, slicing
x = torch.arange(6).reshape(3, 2)
print(x)
print(x[1, 1])

# indexing
print(x[:, 1])

# slicing
print(x[:, 1:])

# view() and reshape() is similar, but view() reflect the most current data
y = torch.arange(10)
print(y)
y.view(2, 5)
print(y.view(2, 5))
y.reshape(2, 5)
print(y)


a = torch.arange(10)
b = a.view(2, 5)
c = a.reshape(2, 5)
print(b)
a[0] = 999
print(a)
print(b)
print(c)
print(a.view(2, -1))
print(a.view(-1, 5))

e = torch.tensor([1, 2, 3])
f = torch.tensor([4, 5, 6])
print(e + f)
print(e.add(f))
print(e)

# add_ change the e in place, reassign e in place
# it is equal to e = e.add(f)
e.add_(f)
print(e)
