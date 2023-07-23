import numpy as np

X1 = np.array([0, 0, 1, 1])
X2 = np.array([0, 1, 0, 1])
Y = np.array([0, 0, 0, 1])

W1 = np.random.randn()
W2 = np.random.randn()
B = np.random.randn()
learning_rate = 0.01

def forward_propagation(X1, X2):
    Y_pred = (X1 * W1) + (X2 * W2) + B
    return Y_pred

def rectangular_function(Y_pred):
    if Y_pred <= 0:
        return 0
    else:
        return 1

def error_func(Y, Y_pred):
    return (1/2) * np.sum((Y - Y_pred) ** 2)

def back_propagation(X1, X2, Y_pred, Y):
    global W1, W2, B  # Define global variables before using them
    dW1 = np.sum(X1 * X2 * W2 + B)
    dW2 = np.sum(X1 * W1 + X2 * B)
    dB = np.sum(X1 * W1 + X2 * W2 + B)

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    B -= learning_rate * dB
    return W1,W2,B

for epoch in range(20):
    Y_pred = forward_propagation(X1, X2)
    Y_pred = np.array([rectangular_function(y) for y in Y_pred])

    if np.all(Y == Y_pred):
        print("Weights are W1 =", W1, "and W2 =", W2, "with bias B =", B)
        break

    W1,W2,B=back_propagation(X1, X2, Y_pred, Y)
    print("After",epoch+1, "Epochs\n","Weights are W1 =", W1, "and W2 =", W2, "with bias B =", B)
