#coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
document = ["I have a pen.",
            "I have an apple."]
tfidf_model = TfidfVectorizer().fit(document)
sparse_result = tfidf_model.transform(document)     # 得到tf-idf矩阵，稀疏矩阵表示法
print(sparse_result)
# (0, 3)	0.814802474667
# (0, 2)	0.579738671538
# (1, 2)	0.449436416524
# (1, 1)	0.631667201738
# (1, 0)	0.631667201738
print(sparse_result.todense())                     # 转化为更直观的一般矩阵
# [[ 0.          0.          0.57973867  0.81480247]
#  [ 0.6316672   0.6316672   0.44943642  0.        ]]
print(tfidf_model.vocabulary_)                      # 词语与列的对应关系
# {'have': 2, 'pen': 3, 'an': 0, 'apple': 1}
