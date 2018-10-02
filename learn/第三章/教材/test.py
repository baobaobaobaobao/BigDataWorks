# _*_ coding:utf-8 _*_
import  trees

'''
mydat,labels=trees.createDataSet()
result=trees.splitDataSet(mydat,0,1)
print  (result)

'''

'''
mydat,labels=trees.createDataSet()
print (trees.chooseBestFeatureToSplit(mydat))

'''
'''
mydat,labels=trees.createDataSet()
mytree=trees.createTree(mydat,labels)
print (mytree)
'''



mydat,labels=trees.createDataSet()
mytree=trees.retrieveTree(0)

trees.storeTree(mytree,'classStorage.txt')
print (trees.grabTree('classStorage.txt'))