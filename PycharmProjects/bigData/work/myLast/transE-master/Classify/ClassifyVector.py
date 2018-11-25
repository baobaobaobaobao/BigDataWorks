
##读取entitVector.txt

from sklearn.cluster import KMeans
import numpy as np
import sys
import csv
from pylab import *
# Reading the file only requires the second column of data

def loadEntityDataToArray():
    csvfile = open('../Vector/entityVector.csv', 'r')  # Python3.5这里不要用rb
    reader = csv.reader(csvfile)
    entitydatamat = np.zeros((14951, 100))  # 初始化矩阵
    train = []
    i = 0
    for line in reader:
        train.append(line[1])
        #print(line[1])
        entitydatamat[i, :] = mat(line[1])
        i += 1
    csvfile.close()
    return  entitydatamat
def LoadRelationVectorToArray():
    csvfile = open('../Vector/relationVector.csv', 'r')  # Python3.5这里不要用rb
    reader = csv.reader(csvfile)
    relationdatamat = np.zeros((1345, 100))  # 初始化矩阵
    train = []
    i=0
    for line in reader:
        train.append(line[1])
        #print (line[1])
        relationdatamat[i,:]=mat(line[1])
        i+=1
    csvfile.close()
    return    relationdatamat


# Clustering to relationVector.txt
def   Cluster(Vector):

        X=Vector
        estimator = KMeans(n_clusters=10)#构造聚类器
        estimator.fit(X)#聚类
        print (X)
        label_pred = estimator.labels_ #获取聚类标签
        #绘制k-means结果
        x0 = X[label_pred == 0]
        x1 = X[label_pred == 1]
        x2 = X[label_pred == 2]
        x3 = X[label_pred == 3]
        x4 = X[label_pred == 4]
        plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
        plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
        plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
        plt.scatter(x3[:, 0], x3[:, 1], c = "black", marker='+', label='label3')
        plt.scatter(x4[:, 0], x4[:, 1], c = "blue", marker='+', label='label4')
        plt.xlabel('petal length')
        plt.ylabel('petal width')
        plt.legend(loc=2)
        plt.show()
        return 1

if __name__ == "__main__":
    print ('Reading the file only requires the second column of data  To  Array"')
    entityVectorArray=loadEntityDataToArray()
    relationVectorArray=LoadRelationVectorToArray()
    print (relationVectorArray.shape)
    Cluster(relationVectorArray)






