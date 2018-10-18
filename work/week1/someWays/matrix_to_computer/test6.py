# coding=utf-8

import numpy as np

mat =    np.loadtxt(open("NLP_Word_Feature.csv", "rb"), delimiter=",", skiprows=0)


output = []
nrow = len(mat)
ncol = len(mat[0])
for i in range(ncol):
    output.append(sum([mat[x][i] for x in range(nrow)]))
print output