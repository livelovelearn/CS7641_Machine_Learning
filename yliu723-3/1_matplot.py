# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 23:13:52 2015
code to plot figures in analysis
@author: yancheng
"""
import matplotlib.pyplot as plt
####kmean ARI vs clusteringNumber
x = [1,2,3,4,5,6,7,8,9]
y1 = [0.0,
      0.849760610205,
      0.773805046989,
      0.746469275813,
      0.734998707422,
      0.393772801774,
      0.381378822985,
      0.374137585833,
      0.368109473609]

plt.plot(x, y1, 'r-')

plt.title("k-means clustering (cancer)",fontsize = 18)
plt.xlabel("number of clusters",fontsize = 18)
plt.ylabel("adjusted rand score",fontsize = 18)
plt.legend()
plt.show()

####EM ARI vs clusteringNumber
x = [1,2,3,4,5,6,7,8,9]
y1 = [0.0,
0.783080155216,
0.43094111378,
0.380288349134,
0.307710256445,
0.298239435281,
0.269111631955,
0.257325305271,
0.248125345777]
plt.plot(x, y1, 'r-')

plt.title("EM clustering (Cancer)",fontsize = 18)
plt.xlabel("number of clusters",fontsize = 18)
plt.ylabel("adjusted rand score",fontsize = 18)
plt.legend()
plt.show()

####kmean ARI vs clusteringNumber
x = [1,2,3,4,5,6,7,8,9]
y1 = [0.0,
0.0260101422972,
0.0148554269098,
0.00688264371674,
0.0055761757633,
0.00362935638191,
0.0145519443234,
0.0116833782168,
0.00908170283424]
plt.plot(x, y1, 'r-')

plt.title("k-means clustering (Cover Forrest)",fontsize = 18)
plt.xlabel("number of clusters",fontsize = 18)
plt.ylabel("adjusted rand score",fontsize = 18)
plt.legend()
plt.show()

####EM ARI vs clusteringNumber
x = [1,2,3,4,5,6,7,8,9]
y1 = [0.0,
0.174529722128,
0.0885614096839,
0.040340824074,
0.0412126139024,
0.0446805484869,
0.0432661173699,
0.041801485375,
0.0420433366023]
plt.plot(x, y1, 'r-')

plt.title("EM clustering (Cover Forrest)",fontsize = 18)
plt.xlabel("number of clusters",fontsize = 18)
plt.ylabel("adjusted rand score",fontsize = 18)
plt.legend()
plt.show()

###Dimension reduction
###cancer dataset
####Eigen value vs number of dimensions
x = [1,2,3,4,5,6,7,8,9]
y1 = [ 0.68761778,  0.07549319,  0.06055252,  0.04389098,  0.03863085,  0.0348255,
  0.02510299,  0.02251864, 0.01136765]
plt.plot(x, y1, 'b-')


plt.title("PCA (Cancer)",fontsize = 18)
plt.xlabel("number of dimensions",fontsize = 18)
plt.ylabel("Eigen value",fontsize = 18)
plt.legend()
plt.show()

####accumulative Eigen value vs number of dimensions
x = [0, 1,2,3,4,5,6,7,8,9]
y1 = [0, 0.68761778,
0.76311097,
0.82366349,
0.86755447,
0.90618532,
0.94101082,
0.96611381,
0.98863245,
1.000000]
plt.plot(x, y1, 'b-')
plt.title("PCA (Cancer)",fontsize = 18)
plt.xlabel("number of dimensions",fontsize = 18)
plt.ylabel("Accumulative Eigen value",fontsize = 18)
plt.legend()
plt.show()

###cover forrest dataset
####Eigen value vs number of dimensions
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
     34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
y1 = [6.84391275e-02,   5.23213105e-02,   4.13231893e-02,   3.68110434e-02,
   2.98370164e-02,   2.69680122e-02,   2.56625092e-02,   2.34225931e-02,
   2.20356081e-02,   2.08494795e-02,   1.98941628e-02 ,  1.94623614e-02,
   1.93642606e-02  , 1.92431797e-02 ,  1.91952643e-02 ,  1.90817316e-02,
   1.90118922e-02  , 1.90047650e-02 ,  1.89917531e-02,   1.89438699e-02,
   1.89372267e-02 ,  1.89247118e-02  , 1.89236216e-02,   1.89163794e-02,
   1.89108305e-02  , 1.89036613e-02  , 1.88979231e-02,   1.88935582e-02,
   1.88892343e-02 ,  1.88824256e-02  , 1.88812409e-02 ,  1.88789782e-02,
   1.88748839e-02,   1.88725635e-02 ,  1.88719182e-02 ,  1.88710046e-02,
   1.88700753e-02 ,  1.88693041e-02 ,  1.87224698e-02,   1.75153791e-02,
   1.54106345e-02 ,  1.45186241e-02,   1.40196043e-02  , 9.56394218e-03,
   8.19112343e-03,   6.92094217e-03 ,  6.14748435e-03,   4.67929262e-03,
   2.88984638e-03 ,  1.44095326e-03  , 4.70329921e-05  , 8.81192498e-27,
   5.33529289e-31  , 1.34972378e-32 ]
plt.plot(x, y1, 'b-')


plt.title("PCA (Cover Forrest)",fontsize = 18)
plt.xlabel("number of dimensions",fontsize = 18)
plt.ylabel("Eigen value",fontsize = 18)
plt.legend()
plt.show()

####accumulative Eigen value vs number of dimensions
x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,
     34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54]
y1 = [6.84E-02,0.120760438,0.162083627,0.198894671,0.228731687,
0.255699699,0.281362209,0.304784802,0.32682041,0.347669889,0.367564052,0.387026413,0.406390674,0.425633854,
0.444829118,0.46391085,0.482922742,0.501927507,0.52091926,0.53986313,0.558800357,0.577725068,0.59664869,0.615565069,
0.6344759,0.653379561,0.672277484,0.691171042,0.710060277,0.728942702,0.747823943,0.766702921,0.785577805,0.804450369,
0.823322287,0.842193292,0.861063367,0.879932671,0.898655141,0.91617052,0.931581154,0.946099779,0.960119383,0.969683325,
0.977874448,0.984795391,0.990942875,0.995622168,0.998512014,0.999952967,1,1,1,1]
plt.plot(x, y1, 'b-')
plt.title("PCA (Cover Forrest)",fontsize = 18)
plt.xlabel("number of dimensions",fontsize = 18)
plt.ylabel("Accumulative Eigen value",fontsize = 18)
plt.legend()
plt.show()