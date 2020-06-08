#  y = 2x4 + x3 + 3x2 + 5x + 1
# y' = 8x3 + 3x2 + 6x + 5
import torch

x = torch.tensor(2.0, requires_grad=True)
y = 2 * x**4 + x**3 + 3 * x**2 + 5*x + 1
print(y)
print(type(y))
print(y.backward())
print(x.grad)

z = torch.tensor([[1.0, 2.0, 3.0], [3., 2., 1.]], requires_grad=True)
print(z)
a = 3*z + 2
b = 2*a**2
print(a)
print(b)

out = b.mean()
print(out)
out.backward()
print(z.grad)
print(a.grad)
