
# coding=utf-8
from getdata import getdata
import pandas as pd
import numpy as np
import pprint
'''
本文件作用是把社交的名字给他构成一个图的领接表。（并且每个人都不重样的）
形如

[张军、刘安] [鲍志强、刘宁、呜呜哦]
to

[source target]
'''



papers = getdata()   #获取所有名字

cnt = 1
HasSet = {}   #利用一个set来让名字不重样
# Graph = []
PointDegreeCnt = 0

EdgeArray = [[],[]]
PointArray = [[], []]

# 去除张军后，总共有515个作者，672篇论文
# 2789，去掉张军后，社交网络共有2789条边
# 568, 是去除张军后，有两个作者的

# def write_csv(filename):
#     df = pd.DataFrame(data=)


for x in papers:
    # 这是一个全连接图
    temparray = []
    for author in x['authors']:
        if author in HasSet:
            authorId = HasSet[author]
        else:
            HasSet[author] = cnt
            PointArray[0].append(cnt)
            PointArray[1].append(author)

            authorId = cnt
            cnt = cnt + 1

        temparray.append(authorId)

    # 然后全连接操作，插入图
    temparraylen = len(temparray)
    if temparraylen > 1:
        PointDegreeCnt = PointDegreeCnt + 1
    for x in range(temparraylen):
        for y in range(x+1, temparraylen):
            XIndex = temparray[x]
            YIndex = temparray[y]
            # 保持XIndex最小
            if XIndex > YIndex:
                temp = XIndex
                XIndex = YIndex
                YIndex = temp

            EdgeArray[0].append(XIndex)
            EdgeArray[1].append(YIndex)

            # Graph.append({XIndex: YIndex})

# Handle Edges

matrix_2 = np.matrix(EdgeArray)
matrix_transpose = np.transpose(matrix_2)
pprint.pprint(matrix_transpose)
df = pd.DataFrame(data=matrix_transpose, columns=['Source', 'Target'])
df.to_csv('edges.csv', index=False)

# Handle Points

matrix_3 = np.matrix(PointArray)
matrix3_transpose = np.transpose(matrix_3)
df2 = pd.DataFrame(data=matrix3_transpose, columns=['Id', 'Label'])
df2.to_csv('points.csv', index=False)


# create graph
df.to_csv('edges.txt', index=False)


print('Done')


# print(len(Graph))
# print(PointDegreeCnt)

# print(papers[0]['authors'])













