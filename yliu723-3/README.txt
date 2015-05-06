I. Tools:
To run the code please make sure Anaconda for Python 2.7 and PyBrain are installed.  

The installation instructions can be found at:
http://continuum.io/downloads#all
http://pybrain.org/docs/index.html

The Clustering (K-means, Expectation-Maximization) and Dimensionality Reduction (PCA, ICA, Randomized Projection, and LDA) algorithms used are from scikit-learn. The Neural Network algorithm is from PyBrain.


II. Datasets:
The datasets used for analysis are from UCI Machine Learning Repository:
Dataset_1. Breast Cancer Wisconsin (Diagnostic).  
Dataset:  https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
Attribute Information: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.names 		

Dataset_2. Forrest Cover Type.
Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
Information: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info

III. Code files:
1_matplot.py (plotting for figures in the analysis file with maplotlib)
2_cancer_clustering.py (cancer dataset clustering analysis with k-means and EM) 
3_cover_clustering.py (coverForest dataset clustering analysis with k-means and EM)
4_cancer_PCA.py (cancer dataset Dimensionality Reduction analysis with PCA, ICA, Randomized Projection, and LDA) 
5_cover_PCA.py (coverForest dataset Dimensionality Reduction analysis with PCA, ICA, Randomized Projection, and LDA) 
6_Cancer_DR_kmeans.py &
7_Cancer_DR_EM.py (reproduce clustering experiments with dimensionality-reduced cancer dataset)
8_Cover_DR_kmean.py &
9_Cover_DR_EM.py (reproduce clustering experiments with dimensionality-reduced coverForest dataset)
10_cancer_DR_ANN.py (Neural Network analysis using cancer dataset preprocessed by dimensionality reduction algorithms)
11_cancer_Clustering_ANN.py (Neural Network analysis using cancer dataset preprocessed by clustering algorithms)

Note: need to change file path.





	  