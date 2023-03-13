import numpy as np
import matplotlib.pyplot as plt
import math


# Need to use numpy array because cant multiply list with float/int
X = np.array([2,3,4,5,6,7])
Y = np.array([4,9,16,25,36,49])


W = 0.0


def forward(x,w):
    z = w*x
    y_pred = [math.exp(item) for item in z]
    return y_pred


def loss(y,y_pred):
    return ((y_pred-y)**2).mean()
    # return 1/len(y) * sum((y_pred-y)**2)


# MSE
# 1/N * (y_pred - y)**2
# 1/N * 2*(y_pred-y) * x
# 1/N * 2*x*(y_pred-y)
def gradient(x,y,y_pred,w):
    # Make sure to correctly set precedence of mean(), otherwise mean will be calc first and then multiplied by x
    z = x*w
    return (2*x*(y_pred-y)).mean() * np.array([math.exp(item) for item in z]).mean()
    # return 1/len(y) *sum(2*x*(y_pred-y))


# Only need 40 iterations to converge because loss is very high with multiple inputs => heavier weight updates 
num_iter = 120
lr = 0.00001

for epoch in range(num_iter):

    y_pred = forward(X,W)

    l = loss(Y,y_pred)

    grad = gradient(X,Y,y_pred,W)

    W -= grad*lr

    print(f"E: {epoch} , W: {W:.2f}, L: {l:.2f}, G: {grad:.2f} P: {[round(item,2) for item in forward(X,W)]} ")
    print("-----------------------------------------------")
    plt.plot(X, Y, linestyle='-', label="ground truth")  # solid
    plt.scatter(X, Y)
    plt.plot(X, forward(X,W), linestyle='-' , label="predicted")  # solid
    plt.scatter(X, forward(X,W))
    plt.legend(loc='upper center')
    plt.show(block=False)
    plt.pause(0.03)
    plt.clf()


plt.plot(X, Y, linestyle='-')  # solid
plt.plot(X, forward(X,W), linestyle='-')  # solid
plt.show()