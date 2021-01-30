from numpy import array
from scipy.linalg import svd

A=array([[1,2],[3,4],[5,6]])
print('Original Matriz A')
print(A)
U,s,V=svd(A)
print('Matrix U (Left)')
print(U)
print('Matrix s (Diagonal)')
print(s)
print('Matrix V (Right)')
print(V)
