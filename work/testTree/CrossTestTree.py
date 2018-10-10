from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pandas as pd
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集

data = pd.read_excel('test.xlsx','南方电网数据',index_col=0)
data.to_csv('data.csv',encoding='utf-8')

iris = []
csv_reader = csv.reader(open('data.csv'))
for row in csv_reader:
    iris.append(row)
del iris[0]
X= [data[2:41] for data in iris]  #截取1到41列为我们的总数据
Y= [data[1] for data in iris]   #第2列为便签
x=np.array(X)
kf = KFold(n_splits=10)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
count =0
i=0
for train_index, test_index in kf.split(x):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    #print (X_train,X_test)
    ###现在将每次区分的训练集来交叉验证并给予平均值计算测试。
    clf.fit(X_train, X_train[:,1])
    i+=1
    print ('第',i,'次的准确率')
    print (clf.score(X_train, X_train[:,1]))
    count+=clf.score(X_train, X_train[:,1])

##计算平均数
count=count/10
print ('----------------------------------------------------------------------------------')
print('训练集准确率平均值')
print (count)





