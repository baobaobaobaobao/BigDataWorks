# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import codecs

'''
本文根据将文档转化为 TFIDF矩阵矩阵，再用k-mean方法来分成十个类。并作图分析
'''


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
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=1, use_idf=True)
    #vectorizer=TfidfTransformer();
    #vectorizer=TfidfVectorizer.fit(dataset)
    X = vectorizer.fit_transform(dataset)
    # print(X)
    # print (X)  #在(index1,index2)中：index1表示为第几个句子或者文档，index2为所有语料库中的单词组成的词典的序号，之后的数字为该词所计算得到的TF－idf的结果值
    #print (X.toarray().shape)
    return X, vectorizer


#transform(delword())   #完成IF-iDF分配权重。



# myWorks()

# -----------------------将文档转化为 词频矩阵-----------------------#
def wordFrequency():
    # 语料
    corpus = delword()
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    # 获取词袋中所有文本关键词
    word = vectorizer.get_feature_names()
    print(word)
    # 查看词频结果
    print(X.toarray())


# -----------------------将文档转化为 TFIDF矩阵矩阵-----------------------#
def tfidf():
    # 语料
    corpus = delword()
    # 类调用
    transformer = TfidfTransformer()
   # print(transformer)
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重

    weight = tfidf.toarray()

    print (weight)
    clf = KMeans(n_clusters=10)
    s = clf.fit(weight)
   # print(s)

    # 10个中心点
    # print(clf.cluster_centers_)
    print(clf.labels_)  # ------------------------!!!!!!!输出每个样本的类别
    result = list(clf.labels_)
    print('Cluster distribution:')
    data=dict([(i,  result.count(i)) for i in  result])
    print  (data[0])
    lists =[]
    for i in range(10):
     lists.append(data[i])
    print ('输出每个分类以及分到这一类的个数并画图')
    print(lists)
    plt.bar(range(len(lists)), lists)
    plt.show()
    print(dict([(i,  result.count(i)) for i in  result]))  # 输出类别以及相应的数目
    return weight


tfidf()



'''
def out():
    dataset = delword()    #数据格式为[ 'Higher order ADI method completed Richardson extrapolation solving unsteady convect
    X, vectorizer = transform(dataset, n_features=732)  #对数据进行训练
    train(X, vectorizer, true_k=10, showLable=True)

out()

'''