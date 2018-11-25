# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date  : 2017-09-20 09:50:08

# import json
import numpy as np
import os
# f = open(r'PDdatatest.json', 'r')
# data = json.load(f)

# labels = np.array(data['labels'])
# for i in labels:
#     print(np.array(i).shape)
# print(type(data['labels']))

# labels = np.array([(np.array(ele)) for ele in data['labels']]).ravel()
# print(labels.shape)
# for i in labels:
# assert (i.shape[-1] == 0)
# print(i.shape[0])
# print(labels[0])
labels = [[1, 2, 3], [1, 5, 3, 4]]
labels = np.array([(np.array(ele)).reshape(-1, 1) for ele in labels])



def to_categorical(filepath):
    txtlist = []
    rootdir = filepath
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
              f = open(path, "r", encoding='UTF-8')  # 设置文件对象
              str = f.read()  # 将txt文件的所有内容读入到字符串str中

              txtlist.append(str)
              f.close()  # 将文件关闭
    return txtlist


def  putInFile(txtlist,filename):
    f = open('test.txt', 'a+')
    for i in range(len(txtlist)):
        f.write('\n'+txtlist[i])
    f.close()


str1=to_categorical('0101')
putInFile(str1,'test.txt')
