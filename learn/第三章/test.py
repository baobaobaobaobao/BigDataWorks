import  trees


reload(trees)
myDat,labes=trees.createDataSet()
trees.calcShannonEnt(myDat)
print  myDat


x= trees.chooseBestFeatureToSplit(myDat)
print  x

