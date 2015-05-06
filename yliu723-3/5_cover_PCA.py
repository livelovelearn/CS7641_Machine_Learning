import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.mixture import GMM
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.decomposition import PCA, FastICA
from sklearn import random_projection

#load data
#please change path when rerun the data
path = 'C:/Users/yancheng/Desktop/covtype.data'
# reading in all data into a NumPy array
all_data_cover = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )
       
cover_y = all_data_cover[:200000,54]
# load all features
cover_X = all_data_cover[:200000,:54]

#scaling
std_scale = preprocessing.StandardScaler().fit(cover_X)
cover_X = std_scale.transform(cover_X)

kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

#Eigen value vs number of dimensions
pca = PCA(n_components=54)
X_r = pca.fit(cover_X).transform(cover_X)
Z = kmeans.fit_predict(X_r)
#Percentage of variance explained for each components
print('explained variance ratio: %s'
      % str(pca.explained_variance_ratio_))

#choose n_components= 3
pca = PCA(n_components=3)
X_r = pca.fit(cover_X).transform(cover_X)
Z = kmeans.fit_predict(X_r)

lda = LDA(n_components=3)
X_r2 = lda.fit(cover_X, cover_y).transform(cover_X)
Z2 = kmeans.fit_predict(X_r2)

ica = FastICA(n_components=3)
X_r3 = ica.fit(cover_X).transform(cover_X)
Z3 = kmeans.fit_predict(X_r3)

transformer_1 = random_projection.GaussianRandomProjection(n_components=3)
X_r4 = transformer_1.fit(cover_X).transform(cover_X)
Z4 = kmeans.fit_predict(X_r4)

transformer_2 = random_projection.SparseRandomProjection(n_components=3)
X_r5 = transformer_2.fit(cover_X).transform(cover_X)
Z5 = kmeans.fit_predict(X_r5)


print "pca:", metrics.adjusted_rand_score(Z, cover_y)
print "lda:", metrics.adjusted_rand_score(Z2, cover_y)
print "ica", metrics.adjusted_rand_score(Z3, cover_y)
print "RP-Gaussian:", metrics.adjusted_rand_score(Z4, cover_y)
print "RP-Sparse:", metrics.adjusted_rand_score(Z5, cover_y)

#scatter plot -3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#original
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [1,2,3,4,5,6,7],('o','x','^','o','x','^','o'),('blue','red','green','yellow','black','cyan','magenta')):

    ax.scatter(cover_X[:,0][cover_y == label],
           cover_X[:,1][cover_y == label],
           cover_X[:,2][cover_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('Elevation')
ax.set_ylabel('Aspect')
ax.set_zlabel('Slope')
plt.title('Cover Forrest dataset')
plt.show()

#PCA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [1,2,3,4,5,6,7],('o','x','^','o','x','^','o'),('blue','red','green','yellow','black','cyan','magenta')):

    ax.scatter(X_r[:,0][cover_y == label],
           X_r[:,1][cover_y == label],
           X_r[:,2][cover_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.title('PCA of Cover Forrest dataset')
plt.show()

#LDA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [1,2,3,4,5,6,7],('o','x','^','o','x','^','o'),('blue','red','green','yellow','black','cyan','magenta')):

    ax.scatter(X_r2[:,0][cover_y == label],
           X_r2[:,1][cover_y == label],
           X_r2[:,2][cover_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_ylabel('dimension 3')
plt.title('LDA of Cover Forrest dataset')
plt.show()

#ICA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [1,2,3,4,5,6,7],('o','x','^','o','x','^','o'),('blue','red','green','yellow','black','cyan','magenta')):

    ax.scatter(X_r3[:,0][cover_y == label],
           X_r3[:,1][cover_y == label],
           X_r3[:,2][cover_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.title('ICA of Cover Forrest dataset')
plt.show()

#RCA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
         [1,2,3,4,5,6,7],('o','x','^','o','x','^','o'),('blue','red','green','yellow','black','cyan','magenta')):

    ax.scatter(X_r4[:,0][cover_y == label],
           X_r4[:,1][cover_y == label],
           X_r4[:,2][cover_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.title('RP of Cover Forrest dataset (2nd round)')
plt.show()