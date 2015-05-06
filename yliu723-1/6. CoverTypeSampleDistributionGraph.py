# -*- coding: utf-8 -*-
"""
Created on Sun Feb 01 13:06:46 2015

@author: yancheng
"""
import numpy as np

path = 'C:/Users/yancheng/.spyder2/covtype.data'
# reading in all data into a NumPy array
all_data = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )

X=all_data[:,:54]

y=all_data[:,54]

count1=0
count2=0
count3=0
count4=0
count5=0
count6=0
count7=0

for i in range (len(y)):
    if y[i]==1:
        count1=count1+1
    if y[i]==2:
        count2=count2+1
    if y[i]==3:
        count3=count3+1
    if y[i]==4:
        count4=count4+1
    if y[i]==5:
        count5=count5+1
    if y[i]==6:
        count6=count6+1
    if y[i]==7:
        count7=count7+1

#print count1, count2, count3, count4, count5, count6, count7

from pylab import *

pos = arange(7) + .5

bar(pos, (count1, count2, count3, count4, count5, count6, count7), align = 'center', color = 'red')
xticks(pos, ('1', '2', '3','4','5','6','7'), align = 'center')
xlabel('Cover Type')
ylabel('# of samples')
title('Sample Distribution')
grid(True)

legend()
show()