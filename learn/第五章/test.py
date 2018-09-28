import  logistics
from numpy import  *
reload(logistics)
dataArr,labelMat=logistics.loadDataSet()

print logistics.gradAscent(dataArr,labelMat)

weights=logistics.gradAscent(dataArr,labelMat)
print  logistics.plotBestFit(weights.getA())

