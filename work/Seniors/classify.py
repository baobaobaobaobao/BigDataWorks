from sklearn import datasets

from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集

from sklearn.metrics import classification_report
from sklearn import metrics
import csv
from numpy import *


iris = []
csv_reader = csv.reader(open('IrisFlowers.csv'))
for row in csv_reader:
    iris.append(row)
x = [data[0:4] for data in iris]  #iris.data
y = [data[-1] for data in iris]   #iris.target
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = .1) 


def showdata(y_pred):
    target_names = ['setosa','versicolor','virginica']
    print ("-----------------------���౨��-----------------------")
    print(classification_report(yTest, y_pred, target_names=target_names))
    print ("-----------------------��������-----------------------")
    print(metrics.confusion_matrix(yTest, y_pred))
    print ("-----------------------���ֲ���-----------------------")
    print ("��accuracy��" ,metrics.precision_score(yTest, y_pred,average='macro'))
    print ("��recall��" ,metrics.recall_score(yTest, y_pred, average='macro'))
    print ("��F1��" ,metrics.f1_score(yTest, y_pred, average='macro')  )
    accuracy = metrics.precision_score(yTest, y_pred,average='macro')
    return accuracy
    
print ("\n####################### KNN #######################")
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 1)
classifier.fit(xTrain, yTrain)
y_pred = classifier.predict(xTest)
KNN_value=showdata(y_pred)


print ("\n################### decision tree ###################")
from sklearn import tree
tree = tree.DecisionTreeClassifier(criterion='entropy') 
tree.fit(xTrain, yTrain)
y_pred = tree.predict(xTest)
tree_value=showdata(y_pred)

print ("\n######################### SVM #########################")
from sklearn import svm
svm = svm.SVC()
svm.fit(xTrain, yTrain)
y_pred = svm.predict(xTest)
svm_value=showdata(y_pred)


def plotcompare(KNN_value,svm_value,tree_value):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import mlab
    from matplotlib import rcParams
    fig1 = plt.figure(3)
    rects1 =plt.bar(x=0.1,height = KNN_value,color=('g'),width = 0.2,align="center",yerr=0.02)
    rects2 =plt.bar(x=0.6,height = svm_value,color=('r'),width = 0.2,align="center",yerr=0.02)
    rects3 =plt.bar(x=1,height = tree_value,color=('b'),width = 0.2,align="center",yerr=0.02)
    plt.legend()
    plt.xticks((0.2,0.6,1),('SVM','KNN','Decision Tree'))
    plt.title("accuracy rate")
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x()+rect.get_width()/2., 1.03*height, '%s' % float(height))
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.show()

plotcompare(KNN_value,svm_value,tree_value)

