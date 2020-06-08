import numpy as np

mylist = [1, 2, 3]
print(type(mylist))

print(np.array(mylist))
arr = np.array(mylist)
print(arr, type(arr))

mylist1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
my_matrix = np.array(mylist1)
print(my_matrix.shape)

print(np.arange(0, 10))
print(np.arange(0, 10, 2))

print(np.zeros(5))

print(np.zeros((4, 10)))
print(np.ones((1, 10)))
print(np.ones((1, 10)) * 5)

print(np.linspace(0, 10, 4))
print(np.eye(5))
