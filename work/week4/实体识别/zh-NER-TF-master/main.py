
#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# 第五步运行
import tensorflow as tf
import numpy as np
##os模块就是对操作系统进行操作
import os, argparse, time, random
from model import BiLSTM_CRF
from utils import str2bool, get_logger, get_entity
from data import read_corpus, read_dictionary, tag2label, random_embedding

## Session configuration
# 在python代码中设置使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# log 日志级别设置，只显示 warning 和 Error，'1' 是默认的显示等级，显示所有信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # default: 0

# 记录设备指派情况:tf.ConfigProto(log_device_placement=True)
# 设置tf.ConfigProto()中参数log_device_placement = True ,
# 可以获取到 operations 和 Tensor 被指派到哪个设备(几号CPU或几号GPU)上运行,
# 会在终端打印出各项操作是在哪个设备上运行的。
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.2  # need ~700MB GPU memory

## hyperparameters超参数设置
# 使用argparse的第一步就是创建一个解析器对象，并告诉它将会有些什么参数。
# 那么当你的程序运行时，该解析器就可以用于处理命令行参数
parser = argparse.ArgumentParser(description='BiLSTM-CRF for Chinese NER task')
parser.add_argument('--train_data', type=str, default='data_path', help='train data source')
parser.add_argument('--test_data', type=str, default='data_path', help='test data source')
parser.add_argument('--batch_size', type=int, default=64, help='#sample of each minibatch')
parser.add_argument('--epoch', type=int, default=40, help='#epoch of training')
parser.add_argument('--hidden_dim', type=int, default=300, help='#dim of hidden state')
parser.add_argument('--optimizer', type=str, default='Adam', help='Adam/Adadelta/Adagrad/RMSProp/Momentum/SGD')
parser.add_argument('--CRF', type=str2bool, default=True, help='use CRF at the top layer. if False, use Softmax')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout keep_prob')
parser.add_argument('--update_embedding', type=str2bool, default=True, help='update embedding during training')
parser.add_argument('--pretrain_embedding', type=str, default='random',
                    help='use pretrained char embedding or init it randomly')
parser.add_argument('--embedding_dim', type=int, default=300, help='random init char embedding_dim')
parser.add_argument('--shuffle', type=str2bool, default=True, help='shuffle training data before each epoch')
parser.add_argument('--mode', type=str, default='test', help='train/test/demo')
parser.add_argument('--demo_model', type=str, default='1521112368', help='model for test and demo')
# 传递参数送入模型中
args = parser.parse_args()

# get char embeddings
'''word2id的形状为{'当': 1, '希': 2, '望': 3, '工': 4, '程': 5,。。'<UNK>': 3904, '<PAD>': 0}
   train_data总共3903个去重后的字'''
word2id = read_dictionary(os.path.join('.', args.train_data, 'word2id.pkl'))

# 通过调用random_embedding函数返回一个len(vocab)*embedding_dim=3905*300的矩阵(矩阵元素均在-0.25到0.25之间)作为初始值
if args.pretrain_embedding == 'random':
    embeddings = random_embedding(word2id, args.embedding_dim)
else:
    embedding_path = 'pretrain_embedding.npy'
    embeddings = np.array(np.load(embedding_path), dtype='float32')

# read corpus and get training data
if args.mode != 'demo':
    # 设置train_path的路径为data_path下的train_data文件
    train_path = os.path.join('.', args.train_data, 'train_data')
    # 设置test_path的路径为data_path下的test_path文件
    test_path = os.path.join('.', args.test_data, 'test_data')
    # 通过read_corpus函数读取出train_data
    """ train_data的形状为[(['我',在'北','京'],['O','O','B-LOC','I-LOC'])...第一句话
                     (['我',在'天','安','门'],['O','O','B-LOC','I-LOC','I-LOC'])...第二句话  
                      ( 第三句话 )  ] 总共有50658句话"""
    train_data = read_corpus(train_path)  ##train_data
    test_data = read_corpus(test_path);  ## test_data
    test_size = len(test_data)

## paths setting
paths = {}
# 时间戳就是一个时间点，一般就是为了在同步更新的情况下提高效率之用。
# 就比如一个文件，如果他没有被更改，那么他的时间戳就不会改变，那么就没有必要写回，以提高效率，
# 如果不论有没有被更改都重新写回的话，很显然效率会有所下降。
timestamp = str(int(time.time())) if args.mode == 'train' else args.demo_model
# 输出路径output_path路径设置为data_path_save下的具体时间名字为文件名
output_path = os.path.join('.', args.train_data + "_save", timestamp)
if not os.path.exists(output_path): os.makedirs(output_path)
# summary_path的路径设置为output_path下的summaries文件
summary_path = os.path.join(output_path, "summaries")
paths['summary_path'] = summary_path
if not os.path.exists(summary_path): os.makedirs(summary_path)
# model_path的路径设置为output_path下的checkpoints文件
model_path = os.path.join(output_path, "checkpoints/")
if not os.path.exists(model_path): os.makedirs(model_path)
# ckpt_prefix保存在checkpoints下的名为model的文件
ckpt_prefix = os.path.join(model_path, "model")
paths['model_path'] = ckpt_prefix
# result_path的路径为时间戳文件下的results文件
result_path = os.path.join(output_path, "results")
paths['result_path'] = result_path
if not os.path.exists(result_path): os.makedirs(result_path)
# log_path='/results/log.txt'
log_path = os.path.join(result_path, "log.txt")
paths['log_path'] = log_path
get_logger(log_path).info(str(args))

## training model
if args.mode == 'train':
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    # 创建节点，无返回值
    model.build_graph()

    ## hyperparameters-tuning, split train/dev
    # dev_data = train_data[:5000]; dev_size = len(dev_data)
    # train_data = train_data[5000:]; train_size = len(train_data)
    # print("train data: {0}\ndev data: {1}".format(train_size, dev_size))
    # model.train(train=train_data, dev=dev_data)
    ## train model on the whole training data

    print("train data: {}".format(len(train_data)))
    # use test_data as the dev_data to see overfitting phenomena
    model.train(train=train_data, dev=test_data)

## testing model
elif args.mode == 'test':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    print("test data: {}".format(test_size))
    model.test(test_data)

## demo
elif args.mode == 'demo':
    ckpt_file = tf.train.latest_checkpoint(model_path)
    print ("ckpt_file文件")
    print(ckpt_file)
    paths['model_path'] = ckpt_file
    model = BiLSTM_CRF(args, embeddings, tag2label, word2id, paths, config=config)
    model.build_graph()
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        print('============= demo =============')
        saver.restore(sess, ckpt_file)
        # 等价于while True
        while (1):
            print('Please input your sentence:')
            # input() 函数接受一个标准输入数据，返回为 string 类型，'我是中国人'
            demo_sent = input()
            # 判断输入是否为空
            if demo_sent == '' or demo_sent.isspace():
                print('See you next time!')
                break
            else:
                # 去除首尾空格
                demo_sent = list(demo_sent.strip())
                # [(['我', '是', '中', '国', '人'], ['O', 'O', 'O', 'O', 'O'])]
                demo_data = [(demo_sent, ['O'] * len(demo_sent))]
                # 送入模型训练，返回每个字正确的tag['O', 'O', 'B-LOC', 'I-LOC', 'O']
                tag = model.demo_one(sess, demo_data)
                # 根据模型计算得到的tag，输出该tag对应的字符，比如LOC：中国
                PER, LOC, ORG = get_entity(tag, demo_sent)
                print('PER: {}\nLOC: {}\nORG: {}'.format(PER, LOC, ORG))
