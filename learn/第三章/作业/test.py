# _*_ coding:utf-8 _*_
import  trees

fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readline()]
lenseslabels=['age','prescript','astigmatic','tearRate']
lensesTeee=trees.createTree(lenses,lenseslabels)
print (lenses)
print (lensesTeee)