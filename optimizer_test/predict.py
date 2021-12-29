#!/home/youli/miniconda3/bin/python
# coding=utf8
"""
# Author: youli
# Created Time : 2021-12-28 21:35:40

# File Name: predict.py
# Description:
    prediction of saved model

"""


print(f"pytorch prediction")

import numpy as np
import pandas as pd
import time
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
import matplotlib.pyplot as plt

# Define model
class t20(nn.Module):
    def __init__(self):
        super(t20, self).__init__()
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

class t20_t20(nn.Module):
    def __init__(self):
        super(t20_t20, self).__init__()
        self.tanh_linear= nn.Sequential(
                nn.Linear(1,20),
                nn.Tanh(),
                nn.Linear(20,20),
                nn.Tanh(),
                nn.Linear(20,1),
                )
        return

    def forward(self, x):
        out = self.tanh_linear(x)
        return out

input_size = 20000
train_size = int(input_size*0.9)
test_size  = input_size-train_size
batch_size = 1000

x_total = np.linspace(-1.0, 1.0, input_size, dtype=np.float32)
x_total= x_total.reshape((input_size,1))
#x_total = np.random.choice(x_total,size=input_size,replace=False) #random sampling

#x_train = x_total[0:train_size]
#x_train= x_train.reshape((train_size,1))
#x_test  = x_total[train_size:input_size]
#x_test= x_test.reshape((test_size,1))
#
#x_train=torch.from_numpy(x_train)
#x_test=torch.from_numpy(x_test)
y_total = torch.from_numpy(np.sinc(10.0 * x_total))
x_total = torch.from_numpy(x_total)

total_data=TensorDataset(x_total,y_total)

model1 = t20()
model1.load_state_dict(torch.load("modeladam.pth"))

x, y = total_data[:][0], total_data[:][1]
plt.figure(figsize=(6,4),dpi=200)
plt.plot(x,y,'b',label="sinc function")


def line_predict(model,line_style,title):
    model.eval()
    with torch.no_grad():
        pred = model(x)
    plt.plot(x,pred,line_style,label=title)
    return 

line_predict(model1,'g--','adam t20')

model1.load_state_dict(torch.load("modellbfgs.pth"))
line_predict(model1,'r--','lbfgs t20')

model1 = t20_t20()
model1.load_state_dict(torch.load("modeladam-t20-t20.pth"))
line_predict(model1,'g','adam t20-t20')

model1.load_state_dict(torch.load("modellbfgs-t20-t20.pth"))
line_predict(model1,'r','lbfgs t20-t20')

plt.legend()

plt.show()



