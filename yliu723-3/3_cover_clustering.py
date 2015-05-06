import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA, FastICA
from sklearn.lda import LDA
from sklearn.mixture import GMM

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

#ARI vs k
for k in range(1,10):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    Z = kmeans.fit_predict(cover_X)
    ARI = metrics.adjusted_rand_score(cover_y, Z)
    print ARI

for k in range(1,10):
    g = GMM(n_components=k, covariance_type='spherical')
    Z =  g.fit(cover_X).predict(cover_X)
    ARI = metrics.adjusted_rand_score(cover_y, Z)
    print ARI

#use the best cluster number 2 got from above
#kmean
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
Z = kmeans.fit_predict(cover_X)

ARI = metrics.adjusted_rand_score(cover_y, Z)
AMI = metrics.adjusted_mutual_info_score(cover_y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(cover_y, Z)

print ARI
print AMI
print h_score, c_score, V_measure

#EM
g = GMM(n_components=2, covariance_type='tied')
g.fit(cover_X)
Z = g.predict(cover_X)

ARI = metrics.adjusted_rand_score(cover_y, Z)
AMI = metrics.adjusted_mutual_info_score(cover_y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(cover_y, Z)

print ARI
print AMI
print h_score, c_score, V_measure

#tweak cover forrest dataset to improve clustering performance
y=all_data_cover[:,54]
for i in range(len(y)):
    if y[i]>2:
        y[i]=3
        
#kmean
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)
Z = kmeans.fit_predict(cover_X)

ARI = metrics.adjusted_rand_score(y, Z)
AMI = metrics.adjusted_mutual_info_score(y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(y, Z)

print ARI
print AMI
print h_score, c_score, V_measure