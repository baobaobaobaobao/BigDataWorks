# -*- coding: utf-8 -*-
# @System: Ubuntu16
# @Author: Alan Lau
# @Date:   2017-09-13 14:43:03

import json
import keras
from datetime import datetime as dt
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Bidirectional, Input, Masking, TimeDistributed
from keras_contrib.layers import CRF
from  keras_contrib.utils import  save_load_utils

###########加载数据
def load(datapath):
    f = open(datapath, 'r')
    data = json.load(f)
    return data['dataset'], data['labels'], dict(data['word_index'])


class revivification:
    def __init__(self, dataset, word_index):
        self.dataset = dataset
        self.word_index = word_index
        self.corpus = []

    def reStore(self):
        for datum in self.dataset:
            sentence = ''.join(list(map(lambda wordindex: next((k for k, v in self.word_index.items(
            ) if v == wordindex), None), list(filter(lambda wordindex: wordindex != 0, datum)))))
            self.corpus.append(sentence)
        return self.corpus



'''
模型训练准备

'''
class nn:
    def __init__(self, dataset, labels, wordvocab):
        self.dataset = np.array(dataset)
        self.labels = np.array(labels)
        self.wordvocab = wordvocab

    def constructModel(self):
        ######模型所需要参数
        vocabSize = len(self.wordvocab)
        embeddingDim = 100  # the vector size a word need to be converted
        maxlen = 100  # the size of a sentence vector
        outputDims = 4 + 1
        # embeddingWeights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        hiddenDims = 100
        self.batchSize = 32
        # NUM_CLASS = 4

        self.train_X = self.dataset
        self.train_Y = np_utils.to_categorical(self.labels, outputDims)

        max_features = vocabSize + 1

        word_input = Input(
            shape=(maxlen, ), dtype='float32', name='word_input')
        mask = Masking(mask_value=0.)(word_input)
        word_emb = Embedding(
            max_features, embeddingDim, input_length=maxlen,
            name='word_emb')(mask)
        bilstm1 = Bidirectional(LSTM(hiddenDims,
                                     return_sequences=True))(word_emb)
        bilstm2 = Bidirectional(
            LSTM(hiddenDims, return_sequences=True))(bilstm1)
        bilstm_d = Dropout(0.8)(bilstm2)
        dense = TimeDistributed(Dense(outputDims,
                                      activation='softmax'))(bilstm_d)

        crf_layer = CRF(outputDims, sparse_target=False)
        crf = crf_layer(dense)
        model = Model(inputs=[word_input], outputs=[crf])
        model.summary()

        model.compile(
            optimizer='adam',
            loss=crf_layer.loss_function,
            metrics=[crf_layer.accuracy])
        return model

    def trainingModel(self):

        model = self.constructModel()
        #print (crf_layer.accuracy)
        result = model.fit(self.train_X, self.train_Y, batch_size=self.batchSize, epochs=10)
        #print ("我得结果")
        #print  (result)
        # model.save(
        #     'PDmodel-crf_epoch_150_batchsize_32_embeddingDim_100_new.h5')
        #save_load_utils.save_all_weights(model,'week4.h5')
        model.save_weights("model.h5")

    def TryLoadModel(self):
        tempModel = self.constructModel()
        save_load_utils.load_all_weights(tempModel, 'week4.h5')

    def save2json(self, json_string, savepath):
        with open(savepath, 'w', encoding='utf8') as f:
            f.write(json_string)
        return "save done."


def main():
    #获取数据
    dataset, labels, wordvocab = load(r'PDdatas.json')
    #送入模型进行训练
    trainLSTM = nn(dataset, labels, wordvocab).trainingModel()


if __name__ == '__main__':
    print(keras.__version__)
    ts = dt.now()
    main()
    te = dt.now()
    spent = te - ts
    print('完成时间为[Finished in %s]' % spent)
