from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pandas as pd
import csv
from sklearn import svm
from sklearn.model_selection import train_test_split  # 用于划分训练集与测试集
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams



###################################################################################
##1  getdata to data.csv file

def getData():
    data = pd.read_excel('test.xlsx','南方电网数据',index_col=0)
    data.to_csv('data.csv',encoding='utf-8')
    iris = []
    csv_reader = csv.reader(open('data.csv'))
    for row in csv_reader:
        iris.append(row)
    del iris[0]
    X= [data[2:41] for data in iris]  #截取1到41列为我们的总数据
    Y= [data[1] for data in iris]   #第2列为便签
    return X




def Use_KFlod_SVM(X):
    x=np.array(X)
    kf = KFold(n_splits=10)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
    count =0
    i=0
    sum=[]
    for train_index, test_index in kf.split(x):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x[train_index], x[test_index]
        #print (X_train,X_test)
        ###现在将每次区分的训练集来交叉验证并给予平均值计算测试。
        clf.fit(X_train, X_train[:,1])
        i+=1
        print ('第',i,'次的准确率')
        temper=clf.score(X_train,X_train[:,1])
        sum.append(temper)
        print (clf.score(X_train, X_train[:,1]))
        count+=clf.score(X_train, X_train[:,1])
    ##计算平均数
    count=count/10
    print ('----------------------------------------------------------------------------------')
    print('训练集准确率平均值')
    print (count)
    print (sum)
    return sum,count

def autolabel(rects):
    for rect in rects:
     height = rect.get_height()
     plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height, '%s' % float(height))


    ###################################################
    #  2 use  Data to complete Plot picture
def plotcompare(sums,count):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import mlab
    from matplotlib import rcParams
    fig1 = plt.figure(3)
    print ()
    rects1 = plt.bar(x=0.1, height=round(sums[0],4), color=('b'), width=0.05, align="center", yerr=0.02)
    rects2 = plt.bar(x=0.2, height=round(sums[1],4), color=('r'), width=0.05, align="center", yerr=0.02)
    rects3 = plt.bar(x=0.3, height=round(sums[2],4), color=('g'), width=0.05, align="center", yerr=0.02)
    rects4 = plt.bar(x=0.4, height=round(sums[3],4), color=('r'), width=0.05, align="center", yerr=0.02)
    rects5 = plt.bar(x=0.5, height=round(sums[4],4), color=('g'), width=0.05, align="center", yerr=0.02)
    rects6 = plt.bar(x=0.6, height=round(sums[5],4), color=('b'), width=0.05, align="center", yerr=0.02)
    rects7 = plt.bar(x=0.7, height=round(sums[6],4), color=('g'), width=0.05, align="center", yerr=0.02)
    rects8 = plt.bar(x=0.8, height=round(sums[7],4), color=('b'), width=0.05, align="center", yerr=0.02)
    rects9 = plt.bar(x=0.9, height=round(sums[8],4), color=('g'), width=0.05, align="center", yerr=0.02)
    rects10 = plt.bar(x=1, height=round(sums[9],4), color=('b'), width=0.05, align="center", yerr=0.02)
    rects11 = plt.bar(x=1.1, height=round(count,4), color=('black'), width=0.05, align="center", yerr=0.02)
    plt.legend()
    plt.xticks((0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1), ('one', 'two', 'three','four','five','six','seven',' eight ','nine','ten','mean'))
    plt.title("accuracy rate  list")
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    autolabel(rects5)
    autolabel(rects6)
    autolabel(rects7)
    autolabel(rects8)
    autolabel(rects9)
    autolabel(rects10)
    autolabel(rects11)
    plt.show()





#plotcompare(sum,count)

if __name__ == "__main__":
 data=getData()    #getData
 a,b=Use_KFlod_SVM(data)  # use kflod and svm
 plotcompare(a,b)   #complete plot picture
