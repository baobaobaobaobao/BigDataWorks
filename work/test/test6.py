# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import codecs


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
    print('Cluster distribution:')
    print(dict([(i, result.count(i)) for i in result]))
   # print(km.labels_)


# test()
'''
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
    print(transformer)
    # 将文本中的词语转换为词频矩阵
    vectorizer = CountVectorizer()
    # 计算个词语出现的次数
    X = vectorizer.fit_transform(corpus)
    # 将词频矩阵X统计成TF-IDF值
    tfidf = transformer.fit_transform(X)
    # 查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重

    weight = tfidf.toarray()
    clf = KMeans(n_clusters=8)
    s = clf.fit(weight)
    print(s)

    # 10个中心点
    # print(clf.cluster_centers_)
    print(clf.labels_)  # ------------------------!!!!!!!输出每个样本的类别

    word = vectorizer.get_feature_names()
    resName = "Tfidf_Result.txt"
    result = codecs.open(resName, 'w', 'utf-8')
    for j in range(len(word)):
        result.write(word[j] + ' ')
    result.write('\r\n\n')

    # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for i in range(len(weight)):
        for j in range(len(word)):
            result.write(str(weight[i][j]) + ' ')
        result.write('\r\n\n')
    result.close()
    return weight
'''

def out():
    dataset = delword()    #数据格式为[ 'Higher order ADI method completed Richardson extrapolation solving unsteady convect
    X, vectorizer = transform(dataset, n_features=732)  #对数据进行训练
    train(X, vectorizer, true_k=10, showLable=True)

out()

