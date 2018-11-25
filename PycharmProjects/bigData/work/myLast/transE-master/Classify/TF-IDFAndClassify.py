# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import codecs
#-------------------------获取我们FB15K数据集中的train.txt中的关系并进行处理，只需要最后一列。





