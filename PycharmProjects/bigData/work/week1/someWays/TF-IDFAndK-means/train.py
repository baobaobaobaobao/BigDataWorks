# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import codecs

# This document is to week1 the DFIDF to generate the classification matrix.


# ---------------------------将papers.txt文档进行分词处理---------------------------#
def loadDataSet():
    words = []  # assume last column is target value
    trainText = open('papers.txt')
    for line in trainText:
        title, author, pubyear = line.strip().split('@', 2)  # 用@分割3次
        title = title.split()  ##默认删除空白符（包括'\n', '\r', '\t', ' ')
        author = author.strip().split(',')
        words.append(title + author)
    newwords = words
    #移除我们所知道的英语连词，因为没什么用
    english_stop_words = ['and', 'for', 'of', 'on', 'in', 'from', 'the', 'with', 'an', 'into']
    #移除我们知道的jun zhang
    for i in range(len(newwords)):
        while (' Jun Zhang' in newwords[i]):
            newwords[i].remove(' Jun Zhang')
        while ('Jun Zhang' in newwords[i]):
            newwords[i].remove('Jun Zhang')
        for t in range(len(english_stop_words)):
            if (english_stop_words[t] in newwords[i]):
                newwords[i].remove(english_stop_words[t])
   # print (newwords)  #这里数据的格式形成了[['Higher', 'order', 'ADI', 'method', 'completed', 'Richardson', 'extrapolation', 'solving', 'unsteady', 'convection-diffusion'
    return newwords

#删除我们文档中的标点符号
def delword():
    rst = ""
    rt = []
    for i in loadDataSet():
        for j in range(len(i)):
            rst += i[j].replace(' ', '').replace(':', '').replace('.', '') \
                .replace('-', '').replace('?', '').replace(',', ' ') \
                .replace('+', ' ').replace('(', '').replace(')', '')
            rst.replace(' ', '')
            if (j != len(i) - 1):
                rst += " "
        rt.append(rst.strip())
        rst = ""
   # print (['nima']+rt)  #数据格式为['nima', 'Higher order ADI method completed Richardson extrapolation solving unsteady convectiondiffusion equations RuxinDai YinWang', 'Trustaware PrivacyPres
    return rt

# ---------------------------构建词典---------------------------#

# ---------------------------将数据集进行训练---------------------------#

def transform(dataset, n_features=100):
    '''采用TfidfVectorizer提取文本特征向量'''
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=1, use_idf=True)
    #vectorizer=TfidfVectorizer.fit(dataset)
    X = vectorizer.fit_transform(dataset)
    # print(X)
    # print (X)  #在(index1,index2)中：index1表示为第几个句子或者文档，index2为所有语料库中的单词组成的词典的序号，之后的数字为该词所计算得到的TF－idf的结果值
    #print (X.toarray().shape)
    return X, vectorizer


#transform(delword())   #完成IF-iDF分配权重。

# 使用我们的k-means方法。
def train(X, vectorizer, true_k=10, minibatch=False, showLable=False):
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1, verbose=False)
    km.fit(X)
    #########print (km.labels_) #输出矩阵对应的类别#########
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        # print (vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(km.predict(X))
    print (km.predict(X))  #输出每篇论文属于的类别
    print (len(result))
    print('Cluster distribution:')
    print(dict([(i, result.count(i)) for i in result]))  #输出类别以及相应的数目
   # print(km.labels_)


# week1()

def out():
    dataset = delword()    #数据格式为[ 'Higher order ADI method completed Richardson extrapolation solving unsteady convect
    X, vectorizer = transform(dataset, n_features=732)  #对数据进行训练
    train(X, vectorizer, true_k=10, showLable=True)

out()

