
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
import  sklearn
from sklearn.model_selection import train_test_split



#  load data
def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    return df

# choose   feature
def featureSet(data):
    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        #print(data.iloc[row]['Date'])
        tmp_list.append(data.iloc[row]['Close'])
        XList.append(tmp_list)
    yList = data.Close.values
    #print ("yList")
    print (yList)
    return XList, yList

'''
Get the last 60 columns of data
'''
def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    XList = []
    for row in range(0, 60):
        tmp_list = []
        tmp_list.append(data.iloc[row]['Close'])
        XList.append(tmp_list)
    return XList


'''

获取最后60行的close数据，方便预测
'''
def  get_first60_Truedata(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    trueList = []
    for row in range(0, 59):
        #tmp_list.append(data.iloc[row]['Close'])
        trueList.append(data.iloc[row]['Close'])
    trueData=np.array(trueList)
    return trueData


#load first  Date to csv
def loadfirstDate(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    trueList = []
    for row in range(0, 60):
        trueList.append(data.iloc[row]['Date'])
    first60Date = np.array(trueList)
    return first60Date

'''
测试集
'''
def trainandTest(X_train, y_train, X_test,max_depth=5,learning_rate=0.1, n_estimators=160,):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=15,
                             learning_rate=0.1,
                             n_estimators=1000,
                             silent=False,
                             objective='reg:gamma')
    model.fit(X_train, y_train)
    # 对测试集进行预测
    global  ans #######申明为全局变量。这样方便使用。
    ans = model.predict(X_test)
    ans_len = len(ans)
    return ans

#########

'''
实现传递参数的方法。
'''

'''
测试集



为了确定boosting 参数，我们要先给其它参数一个初始值。咱们先按如下方法取值：
1、max_depth = 5 :这个参数的取值最好在3-10之间。我选的起始值为5，但是你也可以选择其它的值。起始值在4-6之间都是不错的选择。
2、min_child_weight = 1:在这里选了一个比较小的值，因为这是一个极不平衡的分类问题。因此，某些叶子节点下的值会比较小。
3、gamma = 0: 起始值也可以选其它比较小的值，在0.1到0.2之间就可以。这个参数后继也是要调整的。
4、subsample,colsample_bytree = 0.8: 这个是最常见的初始值了。典型值的范围在0.5-0.9之间。
5、scale_pos_weight = 1: 这个值是因为类别十分不平衡。
注意哦，上面这些参数的值只是一个初始的估计值，后继需要调优。这里把学习速率就设成默认的0.1。然后用xgboost中的cv函数来确定最佳的决策树数量。前文中的函数可以完成这个工作。
'''
def trainandTestNeedPrams(X_train, y_train, X_test,max_depth=5,learning_rate=0.1, n_estimators=160,):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=15,
                             learning_rate=0.1,
                             n_estimators=1000,
                             silent=False,
                             objective='reg:gamma')
    model.fit(X_train, y_train)
    # 对测试集进行预测
    global ans  #######申明为全局变量。这样方便使用。
    ans = model.predict(X_test)
    ans_len = len(ans)
    id_list = np.arange(0, 60)
    data_arr = []
    print (ans_len)

    first60 = loadfirstDate('dataset/soccer/xrp.csv')
    print(first60.shape)
    for row in range(1, ans_len+1):
        data_arr.append([int(id_list[row-1]), ans[row-1], first60[row-1]])
        #print (row)
    # np_data = np.array(data_arr)
    # 写入文件
    pd_data = pd.DataFrame(data_arr, columns=['id', 'predict', 'Date'])
    # print(pd_data)
    print("写入文件")
    pd_data.to_csv('submit.csv', index=None)
    return pd_data

'''
求RMSE
'''
def computerRMSE(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    from math import sqrt
    rmse=sqrt(sum(squaredError) / len(squaredError))
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
    return rmse


def  mains(grade1=1,grade2=1,grade3=1):
    trainFilePath = 'dataset/soccer/xrp.csv'
    testFilePath = 'dataset/soccer/xrp.csv'
    data = loadDataset(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    predict=trainandTest(X_train, y_train, X_test)
    trueData=get_first60_Truedata('dataset/soccer/xrp.csv')
    rmse=computerRMSE(trueData,predict)
    backdata=trainandTestNeedPrams(X_train, y_train, X_test)
    print (rmse)
    return rmse,backdata





if __name__ == '__main__':

    mains()
