import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.lda import LDA
from sklearn import random_projection
import timeit

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

#PCA
pca = PCA(n_components = 3)
cancer_X = pca.fit_transform(cancer_X)
print(pca.explained_variance_ratio_) 

#LDA
lda = LDA(n_components=3)
cancer_X = lda.fit(cancer_X, cancer_y).transform(cancer_X)
print(lda.explained_variance_ratio_) 

#ICA
ica = FastICA(n_components=3)
cancer_X = ica.fit_transform(cancer_X) 

#Random Projection - Gaussian
transformer = random_projection.SparseRandomProjection(n_components = 3)
print cancer_X.shape
cancer_X = transformer.fit_transform(cancer_X)

##############clustering#################
start = timeit.default_timer()
g = GMM(n_components=2, covariance_type='spherical')
g.fit(cancer_X)
Z = g.predict(cancer_X)
stop = timeit.default_timer()

ARI = metrics.adjusted_rand_score(cancer_y, Z)
AMI = metrics.adjusted_mutual_info_score(cancer_y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(cancer_y, Z)

print ARI
print AMI
print h_score, c_score, V_measure
print stop-start

    