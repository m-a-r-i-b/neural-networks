import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import numpy as np

# Use train=False i.e. Testset because of its smaller size (10k vs 60k)
# dataset = torchvision.datasets.MNIST(root="./data",transform=torchvision.transforms.ToTensor(),train=False)
dataset = torchvision.datasets.MNIST(root="./data",transform=torchvision.transforms.ToTensor(), download=True,train=False)

BATCH_SIZE = 400
dataLoader = DataLoader(dataset=dataset,batch_size=BATCH_SIZE,shuffle=True)


dataIterator = iter(dataLoader)
data = next(dataIterator)
features, labels = data

print(len(dataset))
print(features.shape)
print(labels.shape)

# Show single feature with label
singleImage = features[0].numpy()
singleImage = np.moveaxis(singleImage, 0, -1)
singleImage = cv2.resize(singleImage, (780, 540))
cv2.imshow(str(labels[0]),singleImage)
cv2.waitKey(0)



epochs = 1
for epoch in range(epochs):
    for i, (features,labels) in enumerate(dataLoader):
        print(f"Epoch: {epoch}, Iteration: {i}, Data Seen So Far: {(i+1)*BATCH_SIZE}/{len(dataset)} ")
        # Forward pass
        # Backward pass
        # Update weights