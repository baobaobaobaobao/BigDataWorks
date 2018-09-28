#!/usr/bin/python
# -*- coding:utf-8 -*-

import kNN




'''
我们用来测试数据k-紧邻算法
'''


'''
从文件中拿数据
'''
reload(kNN)

datingDataMat,datingLabels=kNN.file2matrix('datingTestSet2.txt')
print datingDataMat

normMat,ranges,minVals=kNN.autoNorm((datingDataMat))
print normMat


kNN.datingClassTest()
print normMat

testvector=kNN.img2vector('0_13.txt')
print testvector[0,0:31]