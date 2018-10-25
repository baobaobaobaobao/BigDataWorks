#coding=utf-8
import os
import sys
import re
import time

f1=open("1.txt",'r+',encoding='UTF-8')
f2=open("2.txt","a")
txt=f1.readlines()
f1.close()
list1=[]
for line in txt:
    if len(line)>=2:
        line_clean=" ".join(line.split())
        lines=line_clean+" "+"\n"
        f2.write(lines)
    else:
        pass
f2.close()