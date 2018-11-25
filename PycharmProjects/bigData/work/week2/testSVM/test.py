
#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xlrd
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集



# 决策树， 训练

data = pd.read_excel('test.xlsx')
all_inputs = data
all_classes = data['labels'].values
# 划分训练集与测试集
(training_inputs, testing_inputs, training_classes, testing_classes) = train_test_split(all_inputs, all_classes,
                                                                                        test_size=0.3, random_state=1)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
clf.fit(training_inputs, training_classes)
print ('训练集准确率')
print (clf.score(training_inputs, training_classes))  # 精度
y_hat = clf.predict(training_inputs)
#print (y_hat)
print ('测试集准确率')
#show_accuracy(y_hat, training_classes, '训练集')
print (clf.score(testing_inputs,testing_classes))


#show_accuracy(y_hat, y_test, '测试集')


def plotcompare(KNN_value, svm_value, tree_value):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import mlab
    from matplotlib import rcParams
    fig1 = plt.figure(3)
    rects1 = plt.bar(x=0.1, height=KNN_value, color=('g'), width=0.2, align="center", yerr=0.02)
    rects2 = plt.bar(x=0.6, height=svm_value, color=('r'), width=0.2, align="center", yerr=0.02)
    rects3 = plt.bar(x=1, height=tree_value, color=('b'), width=0.2, align="center", yerr=0.02)
    plt.legend()
    plt.xticks((0.2, 0.6, 1), ('SVM', 'KNN', 'Decision Tree'))
    plt.title("accuracy rate")

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, '%s' % float(height))

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.show()


#plotcompare(KNN_value, svm_value, tree_value)



