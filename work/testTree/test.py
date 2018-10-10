#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xlrd
from sklearn.tree import DecisionTreeRegressor

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
# 建立决策树  在此全部为默认参数了  主要参数criterion可选‘gini'或'entropy'作为生成树依据,max_deoth可以决定树的深度，max_leaf_nodes限制最大叶子树
decision_tree_classifier = DecisionTreeClassifier(criterion='entropy',max_depth=3)
decision_tree_classifier.fit(training_inputs, training_classes)
decision_tree_output = decision_tree_classifier.predict(testing_inputs)
print('准确率')
print (decision_tree_classifier.score(training_inputs,training_classes))
print('真实值是：')
print(testing_classes)
print('预测值是:')
print(decision_tree_output)


