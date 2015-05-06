# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:11:40 2015

@author: yancheng
"""

import numpy as np

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, FastICA
from sklearn.lda import LDA
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import random_projection

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
# load all features
cancer_X = all_data[:,1:10]

kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)

#Eigen value vs number of dimensions
pca = PCA(n_components=9)
X_r = pca.fit(cancer_X).transform(cancer_X)
Z = kmeans.fit_predict(X_r)
#Percentage of variance explained for each components
print('explained variance ratio: %s'
      % str(pca.explained_variance_ratio_))
      
#choose n_components= 3
pca = PCA(n_components=3)
X_r = pca.fit(cancer_X).transform(cancer_X)
Z = kmeans.fit_predict(X_r)

lda = LDA(n_components=3)
X_r2 = lda.fit(cancer_X, cancer_y).transform(cancer_X)
Z2 = kmeans.fit_predict(X_r2)

ica = FastICA(n_components=3)
X_r3 = ica.fit(cancer_X).transform(cancer_X)
Z3 = kmeans.fit_predict(X_r3)

transformer_1 = random_projection.GaussianRandomProjection(n_components=3)
X_r4 = transformer_1.fit(cancer_X).transform(cancer_X)
Z4 = kmeans.fit_predict(X_r4)

transformer_2 = random_projection.SparseRandomProjection(n_components=3)
X_r5 = transformer_2.fit(cancer_X).transform(cancer_X)
Z5 = kmeans.fit_predict(X_r5)


print "pca:", metrics.adjusted_rand_score(Z, cancer_y)
print "lda:", metrics.adjusted_rand_score(Z2, cancer_y)
print "ica", metrics.adjusted_rand_score(Z3, cancer_y)
print "RP-Gaussian:", metrics.adjusted_rand_score(Z4, cancer_y)
print "RP-Sparse:", metrics.adjusted_rand_score(Z5, cancer_y)

#scatter plot -3D
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#original
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

#PCA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [2,4],('o', '^'),('blue','red')):

    ax.scatter(X_r[:,0][cancer_y == label],
           X_r[:,1][cancer_y == label],
           X_r[:,2][cancer_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.title('PCA of Cancer dataset')
plt.show()

#LDA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [2,4],('o', '^'),('blue','red')):

    ax.scatter(X_r2[:,0][cancer_y == label],
           X_r2[:,0][cancer_y == label],
           #X_r2[:,0][cancer_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 1')
plt.title('LDA of Cancer dataset')
plt.show()

#ICA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [2,4],('o', '^'),('blue','red')):

    ax.scatter(X_r3[:,0][cancer_y == label],
           X_r3[:,1][cancer_y == label],
           X_r3[:,2][cancer_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.title('ICA of Cancer dataset')
plt.show()

#RCA
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
for label,marker,color in zip(
        [2,4],('o', '^'),('blue','red')):

    ax.scatter(X_r4[:,0][cancer_y == label],
           X_r4[:,1][cancer_y == label],
           X_r4[:,2][cancer_y == label],
           marker=marker,
           color=color,
           s=40,
           alpha=0.7,
           label='class {}'.format(label))

ax.set_xlabel('dimension 1')
ax.set_ylabel('dimension 2')
ax.set_zlabel('dimension 3')
plt.title('RP of Cancer dataset (1st round)')
plt.show()