import numpy as np
import matplotlib.pyplot as plt

# EJERCICIO 1.1
M = np.array([(3, -9, 0, 5), (2, -5, -3, 1), (-1, 5, 8, 4)])
A = np.array([-1, 1, 2])
B = np.array([-4, 2, 1, -1])

# 1
Z = np.dot(A, M)
print Z

# 2 ESTA MAL
# AT = np.transpose(A)
# Z = np.dot(AT, B)
# print Z

# 3
BT = np.transpose(B)
Z = np.dot(M, BT)
print Z

# 4
AT = np.transpose(A)
Z = np.dot(A, AT)
print Z

# EJERCICIO 1.2
def f1(x):
    evaluation = 1/(1+np.exp(-x))
    return evaluation

def f2(x):
    evaluation = np.tanh(x)
    return evaluation

def f3(x):
    evaluation = np.sign(x)
    return evaluation

def f4(x):
    evaluation = f1(x) * (1 - f1(x))
    return evaluation

def f5(x):
    evaluation = 1 - np.power(f2(x), 2)
    return evaluation

x = np.arange(-10, 10, 0.1)
y = f1(x)
plt.plot(x, y)
plt.show()