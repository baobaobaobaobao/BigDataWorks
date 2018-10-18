# -*- coding:utf-8 -*-
from numpy import *
from numpy import linalg as la
import numpy

def loaddata():
    dataset = numpy.loadtxt("testdata.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:,0:6]
    Y = dataset[:,5]
    #训练集
    xtrain = X[0:150]
    ytrain = Y[0:150]
    tempx1 = []
    for i in xtrain:
        tempx1+=(list(i))
    alphabet = list(set(tempx1))
    print(len(alphabet)) #432
    char_to_int = dict((str(c), i) for i, c in enumerate(alphabet))
    int_to_char = dict((i, c) for i, c in enumerate(alphabet))
    temp = []
    for i in range(150):
        a = numpy.zeros((432))
        for k in range(6):
            h = str(tempx1[6*i+k])
            a[char_to_int[h]] +=1
        temp.append(a)

    newtest = numpy.reshape(temp,(150,432))
    return newtest,int_to_char,char_to_int,ytrain

# 欧氏距离
def euclidSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


# 皮尔逊相关系数
def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]


# 余弦相似度
def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# 基于物品相似度的推荐引擎（标准相似度计算方法下的用户估计值  ）
def standEst(dataMat, user, simMeas, item):
    # 商品数目
    n = shape(dataMat)[1]
    # 两个用于计算估计评分值的变量
    simTotal = 0.0;
    ratSimTotal = 0.0
    # 遍历所有商品，并将它与所有的物品进行比较
    for j in range(n):
        # 用户对某个物品的评分
        userRating = dataMat[user, j]
        if userRating == 0: continue
        # logical_and:矩阵逐个元素运行逻辑与,返回值为每个元素的True,False
        # dataMat[:,item].A>0: 第item列中大于0的元素
        # dataMat[:,j].A: 第j列中大于0的元素
        # overLap: dataMat[:,item],dataMat[:,j]中同时都大于0的那个元素的行下标(一个向量)
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, \
                                      dataMat[:, j].A > 0))[0]
        print(j)
        print("------overLap------")
        print(overLap)
        if len(overLap) == 0:
            similarity = 0
        # 计算overLap矩阵的相似度
        else:
            similarity = simMeas(dataMat[overLap, item], \
                                 dataMat[overLap, j])
        print("dataMat[overLap,item:")
        print(dataMat[overLap, item])
        print("dataMat[overLap,j:")
        print(dataMat[overLap, j])
        print('the %d and %d similarity is:%f' % (item, j, similarity))
        # 累计总相似度(不太理解)
        #        假设A评分未知，A,B相似度0.9，B评分5,；A C相似度0.8，C评分4.
        #        那么按照公式A评分=（0.9*5+0.8*4）/（0.9+0.8）
        #       相当于加权平均（如果除以2），但是因为2个评分的权重是不一样的，所以应除以相似度之和
        simTotal += similarity
        # ratSimTotal = 相似度*元素值

        ratSimTotal += similarity * userRating
        print("ratSimTotal+=similarity*userRating:")
        print(ratSimTotal)
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

# 利用SVD提高推荐效果
# 基于SVD的评分估计
def svdEst(dataMat, user, simMeas, item):
    # 商品数目
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    # SVD分解为：U*S*V
    U, Sigma, VT = la.svd(dataMat)
    # 分解后只利用90%能量的奇异值，存放在numpy数组里面
    Sig500 = mat(eye(102) * Sigma[:102])
    # 利用U矩阵将物品转换到低维空间中
    xformeditems = dataMat.T * U[:, :102] * Sig500.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformeditems[item, :].T, \
                             xformeditems[j, :].T)
        # print('the %d and %d similarity is :%f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal

# 对某个用户产生最高的N个推荐结果
# user 表示要推荐的用户编号
def recommend(dataMat, user, N=1, simMeas=cosSim, estMethod=svdEst):
    # 对给定用户建立一个未评分的物品矩阵
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]  # 第user行中等于0的元素
    #    print(dataMat[user,:].A==0)----[[ True  True  True ...,  True False  True]]

    if len(unratedItems) == 0: return 'you rated everything'
    # 给未评分物品存放预测得分的列表
    itemScores = []
    # print("未观看",unratedItems)
    for item in unratedItems:
        # 对每个未评分物品通过standEst（）方法来预测得分
        #print("item------------")
        #print(item)
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 将物品编号和估计得分存放在列表中
        itemScores.append((item, estimatedScore))
    # sorted排序函数，key 是按照关键字排序，lambda是隐函数，固定写法，
    # jj表示待排序元祖，jj[1]按照jj的第二列排序，reverse=True，降序；[:N]前N个
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N][0][0]

def calsig():
    myMat, a, b, ytrain = loaddata()
    u,sigma,vt=la.svd(mat(myMat))
    sig2=sigma**2
    print("sum-sig2",sum(sig2))
    print(sum(sig2)*0.9)
    print(sum(sig2[:102]))


if __name__ == '__main__':
    myMat , a, b ,ytrain= loaddata()
    count = 0
    print ("类别对应:",a)
    for i in range(150):
        index = recommend(mat(myMat), i)
        print (index)
        print("第",i+1,"个",ytrain[i],"--->",a[index])
        if(ytrain[i]==(a[index])):
            count +=1
            print("恭喜命中1个！！！")
    print("Model Accuracy: %.2f%%" % (count/150.0 * 100))