import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from sklearn import preprocessing

path = 'C:/Users/yancheng/.spyder2/covtype.data'
# reading in all data into a NumPy array
all_data = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )

#all_data = all_data[y>2]

X=all_data[:,:54]
std_scale = preprocessing.StandardScaler().fit(X)
X = std_scale.transform(X)

y=all_data[:,54]
for i in range(len(y)):
    if y[i]>2:
        y[i]=3

np.random.seed(0)
indices = np.random.permutation(len(X))
Xtrain_train = X[indices[:10000]]
ytrain_train = y[indices[:10000]]
Xtrain_val = X[indices[10000:11000]]
ytrain_val = y[indices[10000:11000]]
Xtest = X[indices[-1000:]]
ytest = y[indices[-1000:]]


ds_train_train = SupervisedDataSet(54, 1)
for i in range(len(Xtrain_train)): 
    ds_train_train.addSample(Xtrain_train[i], ytrain_train[i])

ds_train_val = SupervisedDataSet(54, 1)
for i in range(len(Xtrain_val)): 
    ds_train_val.addSample(Xtrain_val[i], ytrain_val[i])

ds_test = SupervisedDataSet(54, 1)
for i in range(len(Xtest)): 
    ds_test.addSample(Xtest[i], ytest[i])

fnn = buildNetwork(54, 25, 25, 1, bias=True)
                   
trainer = BackpropTrainer(fnn, ds_train_val, learningrate = 0.0001, momentum = 0.9, weightdecay=0.01, verbose= True)

test_score=[]
train_score=[]
epochs_number=[]
for i in range (1, 3, 1):
    trainer.trainUntilConvergence(verbose=True,trainingData=ds_train_train, validationData=ds_train_val, maxEpochs=i)
    epochs_number.append(i)
    y_train = []
    for inp, tar in ds_train_train:
        y_train.append(round(fnn.activate(inp), 0))
        
    y_predict = []
    for inp, tar in ds_test:
        y_predict.append(round(fnn.activate(inp), 0))
 
    test_score.append(accuracy_score(ytest, y_predict))
    train_score.append(accuracy_score(ytrain_train, y_train))
    

plt.plot(epochs_number, train_score, 'r-', label='train')
plt.plot(epochs_number, test_score, 'b-', label='test')
plt.title('FFNN (coverForrest-3 category)', fontsize=16)
plt.xlabel('# of epochs', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.legend()
plt.show()

