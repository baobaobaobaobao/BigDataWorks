# coding=utf-8

import numpy as np
from scipy.spatial.distance import pdist, squareform

mat =    np.loadtxt(open("NLP_Word_Feature.csv", "rb"), delimiter=",", skiprows=0)


X = pdist(mat, 'euclidean')
print (X.squeeze())

'''

https://blog.csdn.net/ljr257816/article/details/52587382
'''