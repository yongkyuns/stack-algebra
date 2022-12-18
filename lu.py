import numpy as np
import scipy.linalg as la
from numpy.linalg import inv,det
from numpy import dot

# A = np.array([[1,2,3,4],
# 			  [2,5,7,3],
# 			  [4,10,14,6],
# 			  [3,4,2,7]])

# A = np.array([[11, 9, 24, 2],
# 			  [1, 5, 2, 6],
# 			  [3, 17, 18, 1],
# 			  [2,5,7,1]])

# A = np.array([[2, 3, 0, 9, 0, 1, 0, 1, 1, 2, 1],
# 			  [1, 1, 0, 3, 0, 0, 0, 9, 2, 3, 1],
# 			  [1, 4, 0, 2, 8, 5, 0, 3, 6, 1, 9],
# 			  [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0],
# 			  [2, 2, 4, 1, 1, 2, 1, 6, 9, 0, 7],
# 			  [0, 0, 0, 6, 0, 7, 0, 1, 0, 0, 0],
# 			  [2, 5, 0, 7, 0, 4, 6, 8, 5, 1, 3],
# 			  [0, 0, 0, 1, 0, 4, 0, 1, 0, 0, 0],
# 			  [0, 0, 0, 8, 0, 2, 0, 0, 0, 0, 0],
# 			  [2, 1, 0, 0, 0, 1, 0, 0, 2, 1, 1],
# 			  [2, 6, 0, 1, 0,30, 0, 2, 3, 2, 1]
# 			  ])

# A = np.array([[1,2],
# 			  [0,0]])

A = np.array([[1,-2],
			  [-3,6]])

# A = np.array([[6,2,3],
# 			  [1, 1, 1],
# 			  [0, 4, 9]])

# A = np.array([[2, 4, 6],
# 			  [0,-1,-8],
# 			  [0, 0,96]])

# A = np.array([[1, 0, 0],
# 			  [8, 1, 0],
# 			  [4, 9, 1]])

print(np.linalg.norm(A))

# print(np.linalg.inv(A))
# print(np.linalg.det(A))
# P, L, U = la.lu(A)
# print(np.dot(P.transpose(), A))
# print(L@U)
# print(P@A)
# print(P)
# print(la.inv(P))
# print(L)
# print(U)
# print(P)
# print(inv(L))
# print(inv(U))
# print(np.dot(np.linalg.inv(U),np.linalg.inv(L)))
# print(inv(U) @ inv(L) @ P)
# print(inv(A))
# print(P)
# print(inv(P))
# print(A @ inv(A))
# print(np.dot(P,A))