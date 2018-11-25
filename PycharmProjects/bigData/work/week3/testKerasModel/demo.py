# -*- coding:utf-8 -*-
# Naive LSTM to learn three-char window to one-char mapping
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
# fix random seed for reproducibility
numpy.random.seed(7)


dataset = numpy.loadtxt("testdata.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:5]
Y = dataset[:,5]

#训练集
x_train = X[0:10000]
x_test = Y[0:10000]

#测试集
y_train = X[10000:12000]
y_test = Y[10000:12000]


alphabet = list(set(Y[0:12000]))
char_to_int = dict((str(c), i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))


############数据预处理下。###############
def data_process(x_train,y_train,x_test,y_test):
    xtrain = []
    for i in x_train:
        xtrain.append(list(i))

    ytrain = []
    for j in y_train:
        ytrain.append(list(j))

    #修改后的y标签
    xtest = []
    for k in x_test:
        xtest.append(char_to_int[str(k)])
    ytest = []
    for h in y_test:
        ytest.append(char_to_int[str(h)])

    return xtrain,ytrain,xtest,ytest

seq_length = 5
xtrain,ytrain,xtest,ytest= data_process(x_train,y_train,x_test,y_test)


###################投入到 LSTM 的 X 需要有这样的结构： [samples, timesteps, features]，所以做一下变换
# reshape X to be [samples, time steps, features]
X = numpy.reshape(xtrain, (len(xtrain), seq_length, 1))
# # normalize
X = X / float(len(alphabet))
y = np_utils.to_categorical(xtest,num_classes=len(alphabet))


# 1建立模型
model = Sequential()
###########2添加LSTM模型。
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))

#########4进行编译
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#######5训练
model.fit(X, y, epochs=10, batch_size=32, verbose=2,validation_split=0.2)

# summarize performance of the model
X2 = numpy.reshape(ytrain, (len(ytrain), seq_length, 1))
x2 = X2 / float(len(alphabet))
y2 = np_utils.to_categorical(ytest,num_classes=len(alphabet))
scores = model.evaluate(x2, y2, verbose=0)

# 输出训练好的模型在测试集上的表现
print("模型准确率Model Accuracy: %.2f%%" % (scores[1]*100))
print (scores)

# demonstrate some model predictions
#print (ytrain)
for pattern in ytrain:
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = pattern
    print ("由",seq_in, "预测出", result)
