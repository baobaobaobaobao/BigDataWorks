# coding=utf-8
import scipy
import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import numpy as np
import matplotlib.pylab as plt

points=scipy.randn(20,4)
points =    np.loadtxt(open("NLP_Word_Feature.csv", "rb"), delimiter=",", skiprows=0)
#1. 层次聚类
#生成点与点之间的距离矩阵,这里用的欧氏距离:
disMat = sch.distance.pdist(points,'euclidean')
disMat = sch.distance.pdist(points,'cityblock')
disMat = sch.distance.pdist(points,'sqeuclidean')

print (disMat)
#进行层次聚类:
Z=sch.linkage(disMat,method='average')
#将层级聚类结果以树状图表示出来并保存为plot_dendrogram.png
P=sch.dendrogram(Z)
plt.savefig('plot_dendrogram.png')
#根据linkage matrix Z得到聚类结果:
cluster= sch.fcluster(Z, t=1)

print ("Original cluster by hierarchy clustering:\n",)
print (cluster)

