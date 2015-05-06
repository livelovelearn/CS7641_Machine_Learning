# -*- coding: utf-8 -*-
"""
Created on Mon Feb 02 00:11:45 2015

@author: yancheng
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer


path = 'C:/Users/yancheng/Desktop/breast-cancer-wisconsin.data'
# reading in all data into a NumPy array
all_data = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )

X=all_data[:,1:10]
y=all_data[:,10]
y = y.astype(np.int64, copy=False)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, 
                                                random_state=42)
Xtrain_train, Xtrain_val, ytrain_train, ytrain_val = train_test_split(Xtrain, ytrain, test_size=0.1, 
                                                random_state=42)

ds_train_train = SupervisedDataSet(9, 1)
for i in range(len(Xtrain_train)): 
    ds_train_train.addSample(Xtrain_train[i], ytrain_train[i])

ds_train_val = SupervisedDataSet(9, 1)
for i in range(len(Xtrain_val)): 
    ds_train_val.addSample(Xtrain_val[i], ytrain_val[i])

ds_test = SupervisedDataSet(9, 1)
for i in range(len(Xtest)): 
    ds_test.addSample(Xtest[i], ytest[i])

fnn = buildNetwork(9, 12, 1, bias=True)

size=[]                   
test_score=[]
train_score=[]
epochs_number=[]
trainer = BackpropTrainer(fnn, ds_train_val, learningrate = 0.001, momentum = 0.1, weightdecay=0.01)
# train network
for i in range (10, 498, 50):
 trainer.trainUntilConvergence(trainingData=ds_train_train[:i], validationData=ds_train_val, maxEpochs=60, continueEpochs = 20)
 
 size.append(i)
 y_train = []
 for inp, tar in ds_train_train[:i]:
     y_train.append(round(fnn.activate(inp)/2, 0)*2)
        
 y_predict = []
 for inp, tar in ds_test:
     y_predict.append(round(fnn.activate(inp)/2, 0)*2)
 
 test_score.append(accuracy_score(ytest, y_predict))
 train_score.append(accuracy_score(ytrain_train, y_train))
 
print(size)

plt.plot(size, train_score, 'r-', label='train (L=0.001)')
plt.plot(size, test_score, 'b-', label='test (L=0.001)')
plt.title('FFNN (cancer)', fontsize=16)
plt.xlabel('training sample size', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.legend()
plt.show()

