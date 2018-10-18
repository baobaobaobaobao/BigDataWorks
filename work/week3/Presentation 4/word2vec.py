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

dataX2 = []
n = 1
for i in X:
    dataX2.append(list(i))
    n = n + 1
    if (n > 1000):
        break


n = 1
dataY2 = []
for i in range(len(Y)):
    dataY2.append(Y[i])
    n = n + 1
    if (n > 1000):
        break



alphabet = list(set(dataY2))
char_to_int = dict((str(c), i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))

#修改后的y标签
temp = []
for i in dataY2:
    temp.append(char_to_int[str(i)])


#print(np_utils.to_categorical(dataY2).shape[1])

seq_length = 5

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX2, (len(dataX2), seq_length, 1))
# # normalize
X = X / float(len(alphabet))
y = np_utils.to_categorical(temp)

# create and fit the model
model = Sequential()
model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=100, batch_size=1, verbose=2,validation_split=0.2)
# summarize performance of the model
scores = model.evaluate(X, y, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))


# demonstrate some model predictions
# for pattern in dataX2:
#     x = numpy.reshape(pattern, (1, len(pattern), 1))
#     x = x / float(len(alphabet))
#     prediction = model.predict(x, verbose=0)
#     index = numpy.argmax(prediction)
#     result = int_to_char[index]
#     seq_in = pattern
#     print (seq_in, "->", result)
