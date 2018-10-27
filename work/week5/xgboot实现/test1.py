#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : soccer_value.py
# @Author: Huangqinjian
# @Date  : 2018/3/22
# @Desc  :
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer


def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    return df


def featureSet(data):
    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['Open*'])
        tmp_list.append(data.iloc[row]['High'])
        tmp_list.append(data.iloc[row]['Low'])
        tmp_list.append(data.iloc[row]['Close**'])
        XList.append(tmp_list)
    yList = data.High.values
    return XList, yList


def loadTestData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    XList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['Open*'])
        tmp_list.append(data.iloc[row]['High'])
        tmp_list.append(data.iloc[row]['Low'])
        tmp_list.append(data.iloc[row]['Close**'])
        XList.append(tmp_list)
    return XList

def loadTestYData(filePath):
    data = pd.read_csv(filepath_or_buffer=filePath)
    data_num = len(data)
    yList = []
    for row in range(0, data_num):
        tmp_list = []
        tmp_list.append(data.iloc[row]['High'])
        yList.append(tmp_list)
    return yList

def trainandTest(X_train, y_train, X_test):
    # XGBoost训练过程
    model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(X_train, y_train)
    # 对测试集进行预测
    ans = model.predict(X_test)
    ans_len = len(ans)
    id_list = np.arange(0, 760)
    data_arr = []
    for row in range(0, ans_len):
        data_arr.append([int(id_list[row]), ans[row]])
    np_data = np.array(data_arr)

    # 写入文件
    pd_data = pd.DataFrame(np_data, columns=['id', 'predict'])
    # print(pd_data)
    pd_data.to_csv('submit.csv', index=None)

if __name__ == '__main__':
    trainFilePath = 'dataset/soccer/xrp.csv'
    testFilePath = 'dataset/soccer/xrp.csv'
    data = loadDataset(trainFilePath)
    X_train, y_train = featureSet(data)
    X_test = loadTestData(testFilePath)
    trainandTest(X_train, y_train, X_test)
