import numpy as np
import timeit
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import matplotlib.pyplot as plt

#load data
#please change path when rerun the data
path = 'C:/Users/yl949/.spyder2/breast-cancer-wisconsin.data'
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

np.random.seed(0)
indices = np.random.permutation(len(cancer_X))
cancer_x_train = cancer_X[indices[:-200]]
cancer_y_train = cancer_y[indices[:-200]]
cancer_x_test = cancer_X[indices[-200:]]
cancer_y_test  = cancer_y[indices[-200:]]

####################### KNN ####################################
# KNN -- change k unweighted
from sklearn.neighbors import KNeighborsClassifier
train_score = []
test_score = []
k=[]
time = []
for i in range(1,50):
    start = timeit.default_timer()
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(cancer_x_train, cancer_y_train)
    train_y_predict=knn.predict(cancer_x_train)
    y_predict=knn.predict(cancer_x_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    k.append(i)
    train_score.append(accuracy_score(cancer_y_train, train_y_predict))
    test_score.append(accuracy_score(cancer_y_test, y_predict))

plt.plot(k, train_score, 'r-', label='train')
plt.plot(k, test_score, 'b-', label='test')

plt.title("KNN unweighted (cancer)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(k, time)
plt.title("KNN unweighted (cancer)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("time")
plt.legend()
plt.show()

# KNN -- change k weighted by distance
from sklearn.neighbors import KNeighborsClassifier
train_score = []
test_score_weighted = []
k=[]
time = []
for i in range(1,50):
    start = timeit.default_timer()
    knn = KNeighborsClassifier(n_neighbors=i, weights = 'distance')
    knn.fit(cancer_x_train, cancer_y_train)
    train_y_predict=knn.predict(cancer_x_train)
    y_predict=knn.predict(cancer_x_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    k.append(i)
    train_score.append(accuracy_score(cancer_y_train, train_y_predict))
    test_score_weighted.append(accuracy_score(cancer_y_test, y_predict))

plt.plot(k, train_score, 'r-', label='train')
plt.plot(k, test_score_weighted, 'b-', label='test')

plt.title("KNN weighted (cancer)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(k, time)
plt.title("KNN weighted (cancer)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("time")
plt.legend()
plt.show()

# kNN unweighted vs weighted
    
plt.plot(k, test_score, 'r-', label='unweighted')
plt.plot(k, test_score_weighted, 'b-', label='weighted')

plt.title("KNN weighted vs unweighted (cancer)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()
    
       
#kNN -- training size
size = []
score_test = []
score_train = []
for i in range(10,498):
    knn = KNeighborsClassifier(weights = 'distance')
    knn.fit(cancer_x_train[:i], cancer_y_train[:i])
    y_predict=knn.predict(cancer_x_test)
    y_predict_train=knn.predict(cancer_x_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cancer_y_test, y_predict))
    score_train.append(accuracy_score(cancer_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("KNN (cancer)")
plt.xlabel("# of training samples")
plt.ylabel("accuracy")
plt.legend()
plt.show()

##################### SVM ################################# 
#SVM -- change kernel
from sklearn import svm
start = timeit.default_timer()
svc = svm.SVC(kernel='linear')
svc.fit(cancer_x_train, cancer_y_train)  
stop = timeit.default_timer()  
y_predict=svc.predict(cancer_x_test)
stop2 = timeit.default_timer()
train_y_predict=svc.predict(cancer_x_train)
print (stop-start, stop2-stop)
accuracy_score(cancer_y_test, y_predict)
accuracy_score(cancer_y_train, train_y_predict)

start = timeit.default_timer()
svc = svm.SVC(kernel='poly', degree = 3, gamma=1)
svc.fit(cancer_x_train, cancer_y_train)  
stop = timeit.default_timer()  
y_predict=svc.predict(cancer_x_test)
stop2 = timeit.default_timer()
train_y_predict=svc.predict(cancer_x_train)
print (stop-start, stop2-stop)
accuracy_score(cancer_y_test, y_predict)
accuracy_score(cancer_y_train, train_y_predict)

start = timeit.default_timer()
svc = svm.SVC(kernel='rbf')
svc.fit(cancer_x_train, cancer_y_train)  
stop = timeit.default_timer()  
y_predict=svc.predict(cancer_x_test)
stop2 = timeit.default_timer()
train_y_predict=svc.predict(cancer_x_train)
print (stop-start, stop2-stop)
accuracy_score(cancer_y_test, y_predict)
accuracy_score(cancer_y_train, train_y_predict)

#SVM -- training size
size = []
score_test = []
score_train = []
for i in range(10,498,20):
    svc = svm.SVC(kernel='linear')
    svc = svc.fit(cancer_x_train[:i], cancer_y_train[:i]) 
    y_predict=svc.predict(cancer_x_test)
    y_predict_train=svc.predict(cancer_x_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cancer_y_test, y_predict))
    score_train.append(accuracy_score(cancer_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("SVM (cancer)",fontsize = 16)
plt.xlabel("# of training samples",fontsize = 16)
plt.ylabel("accuracy",fontsize = 16)
plt.legend()
plt.show()

##################### boosting ################################# 
# AdaBoost - iteration
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
train_score = []
test_score = []
iteration=[]
time = []
for i in range(1,200,5):
    start = timeit.default_timer()
    clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=20), n_estimators=i)
    clf = clf.fit(cancer_x_train, cancer_y_train)
    train_y_predict=clf.predict(cancer_x_train)
    y_predict=clf.predict(cancer_x_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    iteration.append(i)
    train_score.append(accuracy_score(cancer_y_train, train_y_predict))
    test_score.append(accuracy_score(cancer_y_test, y_predict))

plt.plot(iteration, train_score, 'r-', label='train')
plt.plot(iteration, test_score, 'b-', label='test')
plt.title("boosted decision tree -- pruned (cancer)", fontsize = 16)
plt.xlabel("# of iterations", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()

plt.plot(iteration, time)
plt.title("boosted decision tree -- pruned (cancer)", fontsize = 16)
plt.xlabel("# of iterations", fontsize = 16)
plt.ylabel("time", fontsize = 16)
plt.legend()
plt.show()

# AdaBoost - pruned with learning rate
train_score = []
test_score = []
iteration=[]
time = []
for i in range(1, 20, 1):
    start = timeit.default_timer()
    clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=20), n_estimators=100, learning_rate = i/10.0)
    clf = clf.fit(cancer_x_train, cancer_y_train)
    train_y_predict=clf.predict(cancer_x_train)
    y_predict=clf.predict(cancer_x_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    iteration.append(i/10.0)
    train_score.append(accuracy_score(cancer_y_train, train_y_predict))
    test_score.append(accuracy_score(cancer_y_test, y_predict))

plt.plot(iteration, train_score, 'r-', label='train')
plt.plot(iteration, test_score, 'b-', label='test')
plt.title("boosted decision tree -- pruned (cancer)", fontsize = 16)
plt.xlabel("learning rate", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()

plt.plot(iteration, time)
plt.title("boosted decision tree -- pruned (cancer)", fontsize = 16)
plt.xlabel("learning rate", fontsize = 16)
plt.ylabel("time", fontsize = 16)
plt.legend()
plt.show()

#AdaBoost -- training size
size = []
score_test = []
score_train = []
for i in range(10,498,5):
    clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=20), n_estimators=100)
    clf = clf.fit(cancer_x_train[:i], cancer_y_train[:i]) 
    y_predict=clf.predict(cancer_x_test)
    y_predict_train=clf.predict(cancer_x_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cancer_y_test, y_predict))
    score_train.append(accuracy_score(cancer_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("boosted decision tree (cancer)", fontsize = 16)
plt.xlabel("# of training samples", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()

##################### DecisionTree #################################   
# Decision tree with pruning by min_samples_split
from sklearn import tree
train_score = []
test_score = []
min_samples_split_num=[]
time = []
for i in range(80):
    start = timeit.default_timer()
    clf = tree.DecisionTreeClassifier(min_samples_split=i+2)
    clf = clf.fit(cancer_x_train, cancer_y_train)
    train_y_predict=clf.predict(cancer_x_train)
    y_predict=clf.predict(cancer_x_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    min_samples_split_num.append(i+2)
    train_score.append(accuracy_score(cancer_y_train, train_y_predict))
    test_score.append(accuracy_score(cancer_y_test, y_predict))

plt.plot(min_samples_split_num, train_score, 'r-', label='train')
plt.plot(min_samples_split_num, test_score, 'b-', label='test')

plt.title("pruning by min_sample_split_num")
plt.xlabel("min_sample_split_num")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(min_samples_split_num, time)

plt.title("pruning by min_sample_split_num")
plt.xlabel("min_sample_split_num")
plt.ylabel("time")
plt.legend()
plt.show()

# Decision tree with pruning by max_depth
from sklearn import tree
train_score = []
test_score = []
min_samples_split_num=[]
time = []
for i in range(90):
    start = timeit.default_timer()
    clf = tree.DecisionTreeClassifier(max_depth = i+1)
    clf = clf.fit(cancer_x_train, cancer_y_train)
    train_y_predict=clf.predict(cancer_x_train)
    y_predict=clf.predict(cancer_x_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    min_samples_split_num.append(i+2)
    train_score.append(accuracy_score(cancer_y_train, train_y_predict))
    test_score.append(accuracy_score(cancer_y_test, y_predict))

plt.plot(min_samples_split_num, train_score, 'r-', label='train')
plt.plot(min_samples_split_num, test_score, 'b-', label='test')

plt.title("pruning by max_depth")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(min_samples_split_num, time)

plt.title("pruning by max_depth")
plt.xlabel("min_sample_split_num")
plt.ylabel("time")
plt.legend()
plt.show()

#Decision tree -- training size
size = []
score_test = []
score_train = []
for i in range(10,498):
    clf = tree.DecisionTreeClassifier(min_samples_split=20)
    clf = clf.fit(cancer_x_train[:i], cancer_y_train[:i]) 
    y_predict=clf.predict(cancer_x_test)
    y_predict_train=clf.predict(cancer_x_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cancer_y_test, y_predict))
    score_train.append(accuracy_score(cancer_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("decision tree (cancer)")
plt.xlabel("# of training samples")
plt.ylabel("accuracy")
plt.legend()
plt.show()

#scatter plot -3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

for label,marker,color in zip(
        [2,4],('o', '^'),('blue','red')):

    ax.scatter(cancer_X[:,3][cancer_y == label],
           cancer_X[:,1][cancer_y == label],
           cancer_X[:,2][cancer_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('shape')
ax.set_ylabel('thickness')
ax.set_zlabel('size')

plt.title('Cancer dataset')

plt.show()