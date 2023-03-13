import torch
import numpy as np
import torch.nn as nn


X = torch.tensor([1,2,3,4,5,6,7])
# Need to specify dtype as float32, otherwise loss function gives error about long being fond when expected float
Y = torch.tensor([2,4,6,8,10,12,14], dtype=torch.float32)


W = torch.tensor(0.0, requires_grad=True)
# W = torch.tensor([0.0,0.0,0.0,0.0], requires_grad=True)


def forward(x,w):
    return w*x

lr = 0.01
loss = nn.MSELoss()
optimiser = torch.optim.SGD([W],lr)

# Takes a lot of iterations to converge because we are using a weights array, as opposed to a single weight
num_iter = 20

for epoch in range(num_iter):

    y_pred = forward(X,W)

    l = loss(Y,y_pred)

    # Calculate gradient
    l.backward()

    # Update gradient
    optimiser.step()

    print(f"E: {epoch} , W: {W:.2f}, L: {l:.2f}, G: {W.grad:.2f} P: {[round(item.item(),2) for item in forward(X,W)]} ")
    print("-----------------------------------------------")

    # Reset gradient to zero, to avoid accumulation
    optimiser.zero_grad()