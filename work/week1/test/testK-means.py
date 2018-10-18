#coding=utf-8
from __future__ import print_function
import matplotlib.pyplot as plt
import codecs
from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.cluster import KMeans, MiniBatchKMeans



# ---------------------------将文档进行分词处理---------------------------#
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
# ---------------------------将数据集进行训练---------------------------#

def transform(dataset, n_features=100):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=1, use_idf=True)
    #vectorizer=TfidfVectorizer.fit(dataset)
    X = vectorizer.fit_transform(dataset)
   # print(X)
    # print (X)  #在(index1,index2)中：index1表示为第几个句子或者文档，index2为所有语料库中的单词组成的词典的序号，之后的数字为该词所计算得到的TF－idf的结果值
    #print (X.toarray().shape)
    return X, vectorizer

######################这是我们的k-means解决结果。在矩阵中显示出来。就是把每个论文属于的分类显示出来，并且还能显示每个分类的个数###########
'''
num_clusters = 10
km_cluster = KMeans(n_clusters=num_clusters)
# 返回各自文本的所被分配到的类索引
Y,vec=transform(delword())
X=Y.toarray()
result = km_cluster.fit_predict(X)
print (result)
result = list(km_cluster.predict(Y))   #这是我们的k-means解决结果。在矩阵中显示出来。就是把每个论文属于的分类显示出来
print('Cluster distribution:')
print(dict([(i, result.count(i)) for i in result]))   #并且还能显示每个分类的个数

'''


###########接下来就是我们大胆的测试。尝试画出图来，哈哈哈

# 返回各自文本的所被分配到的类索引
Y,vec=transform(delword())
print  (Y)
X=Y.toarray()
estimator = KMeans(n_clusters=10)#构造聚类器
estimator.fit(X)#聚类
print (X)
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
x3 = X[label_pred == 3]
x4 = X[label_pred == 4]
plt.scatter(x0[:, 0], x0[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1[:, 0], x1[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2[:, 0], x2[:, 1], c = "blue", marker='+', label='label2')
plt.scatter(x3[:, 0], x3[:, 1], c = "black", marker='+', label='label3')
plt.scatter(x4[:, 0], x4[:, 1], c = "blue", marker='+', label='label4')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()



