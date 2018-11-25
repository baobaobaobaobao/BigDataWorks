# -*- coding: utf-8 -*-
'''

本处理数据的是0101的测试文件处理的。
'''

import  os
def to_categorical(filepath):
    txtlist = []
    rootdir = filepath
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    #print(list)
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



if __name__ == '__main__':
    #######将0101的文件中所有数据取出生成test.txt文件
    str1 = to_categorical('0101')
    putInFile(str1, 'test.txt')