# -*- coding: utf-8 -*-
'''

本处理数据的是针对2014年人民日报句子标注语料集01的文件处理的。
'''

from pathlib import Path
import os
def  del2014(path):
        p=Path(path)
        FileList=list(p.glob("**/*.txt"))
        #print (FileList)
        txtlisthah=[]
        for File in FileList:
             #处理(File)
             if os.path.isfile(File):
                 f = open(File, "r", encoding='UTF-8')  # 设置文件对象
                 str = f.read()  # 将txt文件的所有内容读入到字符串str中
                 txtlisthah.append(str)
                 f.close()  # 将文件关闭
        print (txtlisthah)
        return txtlisthah


def  putInFile(txtlist,filename):
    f = open('tests.txt', 'a+')
    for i in range(len(txtlist)):
        f.write('\n'+txtlist[i])
    f.close()



if __name__ == '__main__':
    #######将0101的文件中所有数据取出生成test.txt文件
    str1 = del2014('2014年人民日报句子标注语料集01')
    putInFile(str1, 'tests.txt')