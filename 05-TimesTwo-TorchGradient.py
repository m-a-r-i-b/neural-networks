import torch
import numpy as np


X = torch.tensor([2,3,4,5,6,7])
Y = torch.tensor([4,6,8,10,12,14])


W = torch.tensor(0.0, requires_grad=True)
# W = torch.tensor([0.0,0.0,0.0,0.0], requires_grad=True)


def forward(x,w):
    return w*x


def loss(y,y_pred):
    return ((y_pred-y)**2).mean()


num_iter = 15
lr = 0.01

for epoch in range(num_iter):

    y_pred = forward(X,W)

    l = loss(Y,y_pred)

    # Calculate gradient
    l.backward()

    # Update gradient
    grad = W.grad
    with torch.no_grad():
        W -= lr*grad

    # Reset gradient to zero, to avoid accumulation
    W.grad.zero_()

    print(f"E: {epoch} , W: {W:.2f}, L: {l:.2f}, G: {grad:.2f} P: {[round(item.item(),2) for item in forward(X,W)]} ")
    print("-----------------------------------------------")


