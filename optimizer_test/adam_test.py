#!/home/youli/miniconda3/bin/python3
# coding=utf8
"""
# Author: youli
# Created Time : 2021-12-27 15:38:05

# File Name: model_construct.py
# Description:
    test for Pytorch model

"""
print(f"pytorch test")

import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt



input_size = 20000
train_size = int(input_size*0.9)
test_size  = input_size-train_size
batch_size = 1000

x_total = np.linspace(-1.0, 1.0, input_size, dtype=np.float32)
x_total = np.random.choice(x_total,size=input_size,replace=False) #random sampling
x_train = x_total[0:train_size]
x_train= x_train.reshape((train_size,1))
x_test  = x_total[train_size:input_size]
x_test= x_test.reshape((test_size,1))

x_train=torch.from_numpy(x_train)
x_test=torch.from_numpy(x_test)

y_train = torch.from_numpy(np.sinc(10.0 * x_train))
y_test  = torch.from_numpy(np.sinc(10.0 * x_test))

training_data = TensorDataset(x_train,y_train)
test_data = TensorDataset(x_test,y_test)


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break
for X, y in train_dataloader:
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break




# Get cpu or gpu device for training.
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.tanh_linear= nn.Sequential(
                nn.Linear(1,20),
                nn.Tanh(),
               # nn.Linear(20,20),
               # nn.Tanh(),
                nn.Linear(20,1),
                )
        return

    def forward(self, x):
        out = self.tanh_linear(x)
        return out

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)


        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % train_size == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss_train

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")
    return test_loss

# training

opt_label=f'adam_t20'
epochs = 1000
print(f"test for {opt_label}")
optimizer=optimizer_adam
loss_train=[]
loss_test=[]

t1= time.perf_counter()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    loss_train+=[
            test(train_dataloader, model, loss_fn)
            ]
    loss_test+=[
            test(test_dataloader, model, loss_fn)
            ]
    print("Done!")

t2= time.perf_counter()
print("Elapsed time: ", t2- t1)
record=pd.DataFrame({
    'epochs':np.arange(epochs)
    ,'loss_train':np.array(loss_train)
    ,'loss_test':np.array(loss_test)
    })
record.to_csv(f"{opt_label}",sep=' ')




torch.save(model.state_dict(), f"model{opt_label}.pth")
print(f"Saved PyTorch Model State to model{opt_label}.pth")
