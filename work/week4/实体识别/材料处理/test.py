# coding=utf-8
import jieba
import sys

sys.path.append("../")
jieba.load_userdict("1.txt")
import jieba.posseg as pseg
import time

t1 = time.time()
result =[]
stopwords = {}.fromkeys([line.rstrip() for line in open('2.txt')])
f = open("1.txt", "r")  # 读取文本
txtlist = f.read().decode('utf-8')
words = jieba.cut(txtlist)
for w in words:
    seg = str(w.word.encode('utf-8'))
    if seg not in stopwords:
        result += str(seg) + " "  # +"/"+str(w.flag)+" " #去停用词
        f = open("/..../result.txt", "a")  # 将结果保存到另一个文档中
        f.write(result)

f.close()
t2 = time.time()
print("分词及词性标注完成，耗时：" + str(t2 - t1) + "秒。")  # 反馈结果