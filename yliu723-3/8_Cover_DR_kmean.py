import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.mixture import GMM
from sklearn.decomposition import PCA, FastICA
from sklearn.lda import LDA
from sklearn import random_projection
import timeit
from sklearn import preprocessing

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

#PCA
pca = PCA(n_components = 3)
cover_X = pca.fit_transform(cover_X)
print(pca.explained_variance_ratio_) 

#LDA
lda = LDA(n_components=3)
cover_X = lda.fit(cover_X, cover_y).transform(cover_X)

#ICA
ica = FastICA(n_components=3)
cover_X = ica.fit_transform(cover_X) 

#Random Projection - Gaussian
transformer = random_projection.SparseRandomProjection(n_components = 3)
cover_X = transformer.fit_transform(cover_X)

##############clustering#################
start = timeit.default_timer()
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
Z = kmeans.fit_predict(cover_X)
stop = timeit.default_timer()

ARI = metrics.adjusted_rand_score(cover_y, Z)
AMI = metrics.adjusted_mutual_info_score(cover_y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(cover_y, Z)

print ARI
print AMI
print h_score, c_score, V_measure
print stop-start

    