import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.mixture import GMM


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

#ARI vs k, find optimized cluster number
for k in range(1,10):
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    Z = kmeans.fit_predict(cancer_X)
    ARI = metrics.adjusted_rand_score(cancer_y, Z)
    print ARI


for k in range(1,10):
    g = GMM(n_components=k, covariance_type='spherical')
    Z =  g.fit(cancer_X).predict(cancer_X)
    ARI = metrics.adjusted_rand_score(cancer_y, Z)
    print ARI
    
#use the best cluster number 2 got from above
#kmean
kmeans = KMeans(init='k-means++', n_clusters=2, n_init=10)
Z = kmeans.fit_predict(cancer_X)

ARI = metrics.adjusted_rand_score(cancer_y, Z)
AMI = metrics.adjusted_mutual_info_score(cancer_y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(cancer_y, Z)

print ARI
print AMI
print h_score, c_score, V_measure

#EM
g = GMM(n_components=2, covariance_type='spherical')
g.fit(cancer_X)
Z = g.predict(cancer_X)

ARI = metrics.adjusted_rand_score(cancer_y, Z)
AMI = metrics.adjusted_mutual_info_score(cancer_y, Z)
h_score, c_score, V_measure = metrics.homogeneity_completeness_v_measure(cancer_y, Z)

print ARI
print AMI
print h_score, c_score, V_measure

