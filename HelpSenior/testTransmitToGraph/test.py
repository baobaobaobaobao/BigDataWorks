import sys



List1=['1','1','2']
List2=list(set(List1))


result=[]
#lines=[]
with open('123_repeat.txt','r') as f:
        for line in f:
            lines=line.split()
            result.append(list( lines))
print(result)

for  j in range(0, len(result)):
    repeatTemp = str(result[j]).split(',')
    repeatTemp1=list([repeatTemp[1],repeatTemp[0]])
    for i in range(0, len(result)):
        if repeatTemp1 == result[i]:
            result.remove(result[i])


print('result',result)