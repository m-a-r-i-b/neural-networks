import torch
import numpy as np
import torch.nn as nn


# Need to specify dtype as float32, otherwise model(X) gives error
X = torch.tensor([1,2,3,4,5,6,7], dtype=torch.float32)
# X = torch.tensor([2], dtype=torch.float32)
# Need to specify dtype as float32, otherwise loss function gives error about long being found when expected float
Y = torch.tensor([2,4,6,8,10,12,14], dtype=torch.float32)
# Y = torch.tensor([4], dtype=torch.float32)

# 7,7
model = nn.Linear(X.shape[0],Y.shape[0])

lr = 0.01
loss = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(),lr)

num_iter = 20

for epoch in range(num_iter):

    y_pred = model(X)

    l = loss(Y,y_pred)

    # Calculate gradient
    l.backward()

    # Update gradient
    optimiser.step()

    # Reset gradient to zero, to avoid accumulation
    print(f"E: {epoch} , L: {l:.2f}, P: {[round(item.item(),2) for item in model(X)]} ")
    for param in model.parameters():
        print(param)
    print("-----------------------------------------------")

    # Reset gradient to zero, to avoid accumulation
    optimiser.zero_grad()


# X = torch.tensor([2,2,3,3,4,8,9], dtype=torch.float32)
# print(f"E: {epoch} , L: {l:.2f}, P: {[round(item.item(),2) for item in model(X)]} ")
