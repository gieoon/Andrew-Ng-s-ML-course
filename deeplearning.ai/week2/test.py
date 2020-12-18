import numpy as np
import time 

# a = np.array([1,2,3,4,5,6])
# print(a)

# Testing time differences in a 1 million dimension array
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print('Vectorized version: ' + str(1000 * (toc - tic)) + "ms")

c = 0
tic = time.time()
for i in range(10000000):
    c += a[i] * b[i]

toc = time.time()
print(c)
print('For loop: ' + str(1000 * (toc - tic)) + "ms")

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a*b
print(c)