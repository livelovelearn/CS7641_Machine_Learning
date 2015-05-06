# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:11:40 2015

@author: yancheng
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import timeit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn.mixture import GMM

#load data
#please change path when rerun the data
path = 'C:/Users/yancheng/Desktop/breast-cancer-wisconsin.data'
# reading in all data into a NumPy array
all_data = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )
# load class labels from column 1
cancer_y = all_data[:,10]
# conversion of the class labels to integer-type array
cancer_y = cancer_y.astype(np.int64, copy=False)
# load all features
cancer_X = all_data[:,1:10]

kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
Z = kmeans.fit_predict(cancer_X)


g = GMM(n_components=2, covariance_type='spherical')
g.fit(cancer_X)
Z2 = g.predict(cancer_X)

######################################################################################
X_new = np.concatenate((cancer_X, Z.reshape(1,699L).T),axis=1)
Xtrain, Xtest, ytrain, ytest = train_test_split(Z, cancer_y, test_size=0.25, 
                                                random_state=42)  # change Z to Z2 or X_new to use different features for training ANN
Xtrain_train, Xtrain_val, ytrain_train, ytrain_val = train_test_split(Xtrain, ytrain, test_size=0.1, 
                                                random_state=42)

ds_train_train = SupervisedDataSet(10, 1)
for i in range(len(Xtrain_train)): 
    ds_train_train.addSample(Xtrain_train[i], ytrain_train[i])

ds_train_val = SupervisedDataSet(10, 1)
for i in range(len(Xtrain_val)): 
    ds_train_val.addSample(Xtrain_val[i], ytrain_val[i])

ds_test = SupervisedDataSet(10, 1)
for i in range(len(Xtest)): 
    ds_test.addSample(Xtest[i], ytest[i])

fnn = buildNetwork(10, 12, 1, bias=True)
                   
test_score=[]
train_score=[]
epochs_number=[]
trainer = BackpropTrainer(fnn, ds_train_val, learningrate = 0.001, momentum = 0.1, weightdecay=0.01)
# train network
start = timeit.default_timer()
for i in range (1, 10, 1): # change epoches number here
 trainer.trainUntilConvergence(trainingData=ds_train_train, validationData=ds_train_val, maxEpochs=i, continueEpochs = 20)
 epochs_number.append(i)
 y_train = []
 for inp, tar in ds_train_train:
     y_train.append(round(fnn.activate(inp)/2, 0)*2)
        
 y_predict = []
 for inp, tar in ds_test:
     y_predict.append(round(fnn.activate(inp)/2, 0)*2)
 
 test_score.append(accuracy_score(ytest, y_predict))
 train_score.append(accuracy_score(ytrain_train, y_train))

stop = timeit.default_timer()
print stop-start
plt.plot(epochs_number, train_score, 'r-', label='train (L=0.001)')
plt.plot(epochs_number, test_score, 'b-', label='test (L=0.001)')
plt.title('FFNN (cancer)', fontsize=16)
plt.xlabel('# of epochs', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.legend()
plt.show()

###################
#np.concatenate((X_r, Z.reshape(1,699L).T),axis=1) 