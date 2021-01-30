from numpy import array
from scipy.linalg import svd
import numpy as np

A=array([[1,2],[3,4]])
print('Original Matriz A')
print(A)
U,s,V=svd(A)
print('Matrix U (Left)')
print(U)
print('Matrix s (Diagonal)')
print(s)
print('Matrix V (Right)')
print(V)



