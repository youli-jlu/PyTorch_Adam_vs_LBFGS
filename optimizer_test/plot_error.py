#!/home/youli/miniconda3/bin/python
# coding=utf8
"""
# Author: youli
# Created Time : 2021-12-28 20:58:28

# File Name: summary.py
# Description:

"""
print(f"plot test")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

adam1=pd.read_csv("adam-t20",sep=' ')
lbfgs1=pd.read_csv("lbfgs-t20",sep=' ')
adam2=pd.read_csv("adam-t20-t20",sep=' ')
lbfgs2=pd.read_csv("lbfgs-t20-t20",sep=' ')

adam1.loss_train=np.log(adam1.loss_train)
adam2.loss_train=np.log(adam2.loss_train)
lbfgs1.loss_train=np.log(lbfgs1.loss_train)
lbfgs2.loss_train=np.log(lbfgs2.loss_train)


plt.figure(figsize=(6,4),dpi=200)

plt.title(" trainning error in log() ")

lwidth=3.0

plt.plot(adam1.epochs, adam1.loss_train, 'g--', label="adam 1-20-1"       ,linewidth=lwidth)
plt.plot(adam2.epochs, adam2.loss_train, 'g', label="adam 1-20-20-1"      ,linewidth=lwidth)
plt.plot(lbfgs1.epochs*10, lbfgs1.loss_train, 'r--', label="lbfgs 1-20-1" ,linewidth=lwidth)
plt.plot(lbfgs2.epochs*10, lbfgs2.loss_train, 'r', label="lbfgs 1-20-20-1",linewidth=lwidth )
#plt.plot(lbfgs1.epochs, lbfgs1.loss_train, 'r--', label="lbfgs 1-20-1" ,linewidth=lwidth)
#plt.plot(lbfgs2.epochs, lbfgs2.loss_train, 'r', label="lbfgs 1-20-20-1",linewidth=lwidth )

plt.text(0.0,-12,"L-BFGS epochs*10",color='r',size='large')
plt.xlabel("epochs")
plt.ylabel("log(MAE)")
plt.legend()
plt.show()
