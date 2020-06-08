import numpy as np

print(np.random.rand(4, 5))

print(np.random.randn(10))
print(np.random.randint(1, 100, 10))
np.random.seed(42)
print(np.random.rand(4))

arr = np.arange(25)
print(arr)
print(arr.shape)

ranarr = np.random.randint(0, 50, 10)
print(ranarr)

print(arr.reshape(5, 5))

print(ranarr.max())
print(ranarr.argmin())
