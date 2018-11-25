# _*_ coding:utf-8 _*_
from bs4 import BeautifulSoup
import requests
from lxml import etree
from w3lib.html import remove_tags
import  jieba
from collections import Counter
import jieba.analyse
import re
import  json
import time
#url = 'https://dblp.uni-trier.de/pers/hd/z/Zhang:Jun'
url='http://www.bitcoin86.com/szb/ripple/'


#通过url拿到固定网站的数据
def getResponse(url):
    r = requests.get(url)
    anw = BeautifulSoup(r.text, 'lxml')
    return anw

#获取所有链接
def   getAllLink():
    for i in range(1,7):
      getUrls(url+'list_19_'+str(i)+'.html')


#获取我们的页面链接数据并放在page_link中，
def getUrls(url):
    html = requests.get(url)
    selector = etree.HTML(html.text)
    cates = selector.xpath('/html/body/section/div/div/article/a/@href')
    for i in cates:
        alllinks.append('http://www.bitcoin86.com' + i)  #print("page_link",i)



##########获取每个链接中的数据。包括日期，还有文章。并进过一些处理。填入allData中
def get_attractions(url,data=None):
    wb_data = requests.get(url)
    requests.encoding = 'utf-8'
    soup = BeautifulSoup(wb_data.content,'lxml')
    data=soup.select('.article-content')
    date=soup.select('span[class="item"]')[0].text
    #print ('date',date)
    allDate.append(date)
    retoveTagData=remove_tags(str(data))
    retoveTagData1 = re.sub(r'[A-Za-z0-9]|/d+[\s+\.\!\/_,$%^*(+\"\']+|[+——！/  - ， . ：\xa0 \t: \r \n。？、~@#￥%……&*（）]+','', retoveTagData)
    allData.append(retoveTagData1)


# Enter into the txt file(outWord.txt)
def  takeInTxt(allDateAndWord):
   # print ('nihaloa',len(allDateAndWord))
    fp = open('outWord.txt', "w",encoding='utf-8')
    for key in allDateAndWord:
            fp.write(key)
            fp.write(":")
            fp.write(str(allDateAndWord[key]))
            fp.write("\n")
    fp.close()
    #print (allDateAndWord)


#  Forming the date and the corresponding stuttering word into a key-value pair form
def   getHigh(allData):
   # print ('lensAllDatea',len(allData))
    for j in range(0,len(allData)):
        seg_str = jieba.cut(allData[j])
        result = dict(Counter(seg_str))
        high=''
        for k, v in result.items():
            if k == '':
                continue
            elif v < 10 and v > 3 and len(k)>=2:
                high+=(str(dict(Counter([k, str(v)]))))
        #print (allDate[j], high)

        allDateAndWord[allDate[j]]=json.dumps(high.replace('}{',','), ensure_ascii=False)



def WriteToJson(allDateAndWord):
    with open("records.json", "w", encoding='utf-8') as f:
        json.dump(allDateAndWord, f, ensure_ascii=False)
    print("加载入文件完成...")

if __name__ == '__main__':
    alllinks = []  # 所有的链接
    allData = []  # all text data
    allDate = []  # all Date
    allDateAndWord = {}
    getAllLink()                               #获取所有链接放入  alllinks
    for i in range(0,len(alllinks)):
        get_attractions(alllinks[i])           #通过所有链接来获取相应的数据 放入allData
    getHigh(allData)                            # 通过jieba分词来将高频出现的词语及其出现次数放入 allDateAndWord
    takeInTxt(allDateAndWord)                   #将 allDateAndWord 输入到txt文件
    WriteToJson(allDateAndWord)                     #将allDateAndWord输入到json文件







