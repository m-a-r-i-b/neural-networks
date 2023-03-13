import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import time

class MyMNISTModel(nn.Module):
    
    def __init__(self,inputLayerSize,hiddenLayerSize,outputLayerSize) -> None:
        super().__init__()
        self.inputLayerSize=inputLayerSize
        self.layer1 = nn.Linear(inputLayerSize,hiddenLayerSize)
        self.activationFunction = nn.ReLU()
        self.layer2 = nn.Linear(hiddenLayerSize,outputLayerSize)

    def forward(self,x):
        out = self.layer1(x)
        out = self.activationFunction(out)
        out = self.layer2(out)
        return out



BATCH_SOZE = 200

trainDataset = torchvision.datasets.MNIST(root="./data",train=True,transform=torchvision.transforms.ToTensor())
testDataset = torchvision.datasets.MNIST(root="./data",train=False,transform=torchvision.transforms.ToTensor())

trainDatasetLoader = DataLoader(trainDataset,batch_size=BATCH_SOZE,shuffle=True)
testDatasetLoader = DataLoader(testDataset,batch_size=BATCH_SOZE,shuffle=True)



# We flatten out the image and pass every pixel to a node/neuron
INPUT_LAYER_SIZE = 784 #1*28*28 
HIDDEN_LAYER_SIZE = 100
OUTPUT_LAYER_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyMNISTModel(INPUT_LAYER_SIZE,HIDDEN_LAYER_SIZE,OUTPUT_LAYER_SIZE)
# model = MyMNISTModel(INPUT_LAYER_SIZE,HIDDEN_LAYER_SIZE,OUTPUT_LAYER_SIZE).to(device)

lossFn = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)


t1 = time.time()
for epoch in range(EPOCHS):

    for iteration, (images,labels) in enumerate(trainDatasetLoader):

        # -1 tells it to keep first column (batchSize) the same and adjust remaining (flatten out)
        images = images.reshape(-1,28*28)
        # images = images.reshape(-1,28*28).to(device)
        
        # labels = labels.to(device)

        # Foward pass
        predictions = model(images)
        loss = lossFn(predictions,labels)

        # Backwards
        # Calc gradients
        loss.backward()
        # Update params w.r.t gradients
        optimiser.step()
        # Reset gradients to zero for next iteration
        optimiser.zero_grad()


        # Print stats every 5th iteration
        if((iteration+1)%5==0):
            print(f"Epoch: {epoch} , Iteration: {iteration}, L: {loss:.4f}")

t2 = time.time()
print("Time taken to train = ",t2-t1)



correctPredictions = 0
totalSamples = 0
for iteration ,(images,labels) in enumerate(testDatasetLoader):
    
    images = images.reshape(-1,28*28)
    predictions = model(images)

    if iteration == 0:
        print(predictions.data[0])
        print(torch.max(predictions.data[0],0))
        print("-"*30)
    
    # Get max values & their index 
    maxValues, maxValueIndex = torch.max(predictions.data,1)

    totalSamples += len(labels)
    correctPredictions += (maxValueIndex == labels).sum().item()


print(f"Accuracy  = {correctPredictions}/{totalSamples} = {100*correctPredictions/totalSamples}%")   