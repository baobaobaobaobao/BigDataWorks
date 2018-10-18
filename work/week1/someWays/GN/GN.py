#coding:utf-8
import networkx as nx
import math
import csv
import random as rand
import sys
import matplotlib.pyplot as plt
'''
重要概念
边介数（betweenness）：网络中任意两个节点通过此边的最短路径的数目。
在一个网络之中，通过社区内部的边的最短路径相对较少，而通过社区之间的边的最短路径的数目则相对较多。
算法思想：
GN算法的步骤如下： 
（1）计算每一条边的边介数； 
（2）删除边界数最大的边； 
（3）重新计算网络中剩下的边的边阶数；
（4）重复(3)和(4)步骤，直到网络中的任一顶点作为一个社区为止
'''
def buildG(G, file_, delimiter_):
   # reader = csv.reader(open(file_), delimiter=delimiter_)
   # for line in reader:
    #    G.add_edge(int(line[0]),int(line[1]))
    with open('edges.txt', 'r') as f:
        list1 = f.readlines()
        for line in list1:
            [source, target] = (line[:-2]).split(' ')
            G.add_edge(source, target)


def CmtyStep(G):
    init_number_comp = nx.number_connected_components(G)
    number_comp = init_number_comp
    while number_comp <= init_number_comp:
        bw = nx.edge_betweenness_centrality(G)#计算所有边的边介数中心性
        if bw.values() == []:
            break
        else:
            max_ = max(bw.values())#将边介数中心性最大的值赋给max_
        for k, v in bw.iteritems():#删除边介数中心性的值最大的边
            if float(v) == max_:
                G.remove_edge(k[0],k[1])
        number_comp = nx.number_connected_components(G)#计算新的社团数量

def GetModularity(G, deg_, m_):
    New_A = nx.adj_matrix(G)#建立一个表示边的邻接矩阵
    New_deg = {}
    New_deg = UpdateDeg(New_A, G.nodes())
    #计算Q值
    comps = nx.connected_components(G)#建立一个组成的列表
    print  ('输出我们分的类别个数')
    print ('Number of communities in decomposed G: %d' % nx.number_connected_components(G))
    Mod = 0#设定社团划分的模块化系数并设初始值为0
    for c in comps:
        AVW = 0#两条边在邻接矩阵中的值
        K = 0#两条边的度值
        for u in c:
            AVW += New_deg[u]
            K += deg_[u]
        Mod += ( float(AVW) - float(K*K)/float(2*m_) )#计算出Q值公式累加符号后的值
    Mod = Mod/float(2*m_)#计算出模块化Q值
    return Mod

def UpdateDeg(A, nodes):
    deg_dict = {}
    nodess=list(nodes)
    n = len(nodes)#图中点的个数
    B = A.sum(axis = 1)#将矩阵的每一行向量相加，所得一个数组赋给B，表示与每个点相关的边数
    #print  (B)
    for i in range(n):
        deg_dict[nodess[i]] = B[i,0]#将该值存到索引是i的元组中
    return deg_dict

def runGirvanNewman(G, Orig_deg, m_):
    BestQ = 0.0
    Q = 0.0
    while True:
        CmtyStep(G)
        Q = GetModularity(G, Orig_deg, m_);
        print ("输出我们分类的模糊度")
        print ("Modularity of decomposed G:", Q )
        print ('#################################################################')

        if Q > BestQ:
            BestQ = Q
            Bestcomps = nx.connected_components(G)
            BestG = nx.Graph()
            BestG = G
            print ("最好的分类")
            print ("Components:", Bestcomps)
            nx.draw_spring(BestG,node_size =100,alpha = 0.5,edge_color = 'b',font_size = 9)
            plt.savefig('BestG.png')
            plt.clf()
        if G.number_of_edges() == 0:
            break
    if BestQ > 0.0:
        print ('——————————————————————————————————————————————————————')
        print ("输出更好的模块度。")
        print ("Max modularity (Q): " , BestQ)
        print ("Graph communities:", Bestcomps)

    else:
        print ("Max modularity  (Q): ", BestQ)

def main(argv):
   # graph_fn = argv[1]
    G = nx.Graph()
    a='edges.txt'
    buildG(G, a, ',')
    n = G.number_of_nodes()#顶点数量
    A = nx.adj_matrix(G)#邻接矩阵
    m_ = 0.0#计算边的数量
    for i in range(0,n):
        for j in range(0,n):
            m_ += A[i,j]
    m_ = m_/2.0
    #计算点的度
    Orig_deg = {}

    Orig_deg = UpdateDeg(A, G.nodes())
    #调用算法
    runGirvanNewman(G, Orig_deg, m_)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
