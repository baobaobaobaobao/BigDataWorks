# _*_ coding:utf-8 _*_
import  trees

fr=open('lenses.txt')
#print (fr.readlines())
listsss=[]
for inst in fr.readlines():
   listsss.append(inst.strip().split('\t'))
#lenses=[inst.strip().split('\t') ]
lenseslabels=['age','prescript','astigmatic','tearRate']
lensesTeee=trees.createTree(listsss,lenseslabels)
print (listsss)
print (lensesTeee)