import numpy as np

A = np.array([[1,2], [3,4]])

print("A = ", A)
print("A.shape: ", A.shape)
print("A.dtype: ", A.dtype)

B = np.array([[1,2], [3,4]])

print("B = ", B)
print("B.shape: ", B.shape)
print("B.dtype: ", B.dtype)

print("A + B = ", A+B)
print("A - B = ", A-B)
print("A * B = ", A*B)
print("A / B = ", A/B)