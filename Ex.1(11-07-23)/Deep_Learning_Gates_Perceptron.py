import numpy as np

def rectangular_func(Y):
    if Y <= 0:
        return 0
    else:
        return 1

def forward_propagation(X1, X2, weights, bias):
    X = np.array([X1, X2])
    W = np.array(weights)
    Y = np.sum(X * W) + bias
    return Y

def and_gate(X1, X2):
    weights = [1, 1]
    bias = -1
    Y = forward_propagation(X1, X2, weights, bias)
    return Y

def or_gate(X1, X2):
    weights = [1, 1]
    bias = 0
    Y = forward_propagation(X1, X2, weights, bias)
    return Y

def nand_gate(X1, X2):
    weights = [-1, -1]
    bias = 2
    Y = forward_propagation(X1, X2, weights, bias)
    return Y

def nor_gate(X1, X2):
    weights = [-1, -1]
    bias = 1
    Y = forward_propagation(X1, X2, weights, bias)
    return Y

def not_gate(X):
    weights = [0]
    bias = 1
    Y = forward_propagation(X, 0, weights, bias)
    return Y

def tautology(X1, X2):
    weights = [0, 0]
    bias = 1
    Y = forward_propagation(X1, X2, weights, bias)
    return Y

X1 = [0, 0, 1, 1]
X2 = [0, 1, 0, 1]
Y_l= []
for i in range(4):
    Yp = and_gate(X1[i], X2[i])
    new_Yp = rectangular_func(Yp)
    print("(",X1[i],",",X2[i],")-->",new_Yp)
    Y_l.append(new_Yp)
print(Y_l)
