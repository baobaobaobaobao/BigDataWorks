# 第三步
import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # 首先被内层IOError异常捕获，打印“inner exception”, 然后把相同的异常再抛出，
        # 被外层的except捕获，打印"outter exception"
        raise argparse.ArgumentTypeError('Boolean value expected.')


# 根据输入的tag返回对应的字符
def get_entity(tag_seq, char_seq):
    PER = get_PER_entity(tag_seq, char_seq)
    LOC = get_LOC_entity(tag_seq, char_seq)
    ORG = get_ORG_entity(tag_seq, char_seq)
    return PER, LOC, ORG


# 输出PER对应的字符
def get_PER_entity(tag_seq, char_seq):
    length = len(char_seq)
    PER = []
    # 构成一个zip对象,形状类似[( 1, ),( 1, ),( 2, ),( 2, )]
    # zip函数可以接受一系列的可迭代对象作为参数，将对象中对应的元素打包成一个个tuple(元组)，
    # 在zip函数的括号里面加上*号，则是zip函数的逆操作
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        # tag里包含了O,B-PER,I-PER,B-LOCI-PER,B-ORG,I-PER
        if tag == 'B-PER':
            if 'per' in locals().keys():
                PER.append('per')
                del per
            per = char
            if i + 1 == length:
                PER.append(per)
        if tag == 'I-PER':
            per += char
            if i + 1 == length:
                PER.append(per)
        if tag not in ['I-PER', 'B-PER']:
            if 'per' in locals().keys():
                PER.append(per)
                del per
            continue
    return PER


# 输出LOC对应的字符
def get_LOC_entity(tag_seq, char_seq):
    length = len(char_seq)
    LOC = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-LOC':
            if 'loc' in locals().keys():
                LOC.append('loc')
                del loc
            loc = char
            if i + 1 == length:
                LOC.append(loc)
        if tag == 'I-LOC':
            loc += char
            if i + 1 == length:
                LOC.append(loc)
        if tag not in ['I-LOC', 'B-LOC']:
            if 'loc' in locals().keys():
                LOC.append(loc)
                del loc
            continue
    return LOC


# 输出ORG对应的字符
def get_ORG_entity(tag_seq, char_seq):
    length = len(char_seq)
    ORG = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == 'B-ORG':
            if 'org' in locals().keys():
                ORG.append('org')
                del org
            org = char
            if i + 1 == length:
                ORG.append(org)
        if tag == 'I-ORG':
            org += char
            if i + 1 == length:
                ORG.append(org)
        if tag not in ['I-ORG', 'B-ORG']:
            if 'org' in locals().keys():
                ORG.append(org)
                del org
            continue
    return ORG


# 记录日志
def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
