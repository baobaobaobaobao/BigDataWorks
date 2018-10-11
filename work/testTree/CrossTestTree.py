from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pandas as pd
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集


##  getData  and conver to (data.csv) file   that convenience to codeing
# return dataSet
def  getData():
    data = pd.read_excel('test.xlsx', '南方电网数据', index_col=0)
    data.to_csv('data.csv', encoding='utf-8')
    iris = []
    csv_reader = csv.reader(open('data.csv'))
    for row in csv_reader:
        iris.append(row)
    del iris[0]
    X = [data[1:41] for data in iris]  # 截取1到41列为我们的总数据
    #print (X)
    Y = [data[1] for data in iris]  # 第2列为便签
    return  X

#使用k折交叉验证法并进行准确率计算。
def  kFolds(X):
            x=np.array(X)
            kf = KFold(n_splits=10)
            count =0
            i=0
            sum=[]
            decision_tree_classifier = DecisionTreeClassifier(criterion='gini',max_depth=15)
            # decision_tree_classifier.fit(training_inputs, training_classes)
            # decision_tree_output = decision_tree_classifier.predict(testing_inputs)
            #
            for train_index, test_index in kf.split(x):
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = x[train_index], x[test_index]
                #print (X_train,X_test)
                ###现在将每次区分的训练集来交叉验证并给予平均值计算测试。
                decision_tree_classifier.fit(X_train, X_train[:,1])
                i+=1
                print ('第',i,'次的准确率')
                temper=decision_tree_classifier.score(X_train,X_train[:,1])
                sum.append(temper)
                print (temper)
                count+=temper

            ##计算平均数
            count=count/10
            print ('----------------------------------------------------------------------------------')
            print('训练集准确率平均值')
            print (count)
            print ("准确率数组")
            print (sum)
            return  sum,count

if __name__ == "__main__":
      Xtest=getData()
      sums,count=kFolds(Xtest)



