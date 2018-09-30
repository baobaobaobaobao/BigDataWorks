# _*_ coding:utf-8 _*_

#! python3
from bs4 import BeautifulSoup
import requests
from lxml import etree

'''
爬取所有数据
'''
url = 'https://dblp.uni-trier.de/pers/hd/z/Zhang:Jun'

#获取我们的所有论文的页面链接并放在page_link中，
def getUrls(url):
    page_links = []
    html = requests.get(url)
    selector = etree.HTML(html.text)
    cates = selector.xpath('//*[@id="info-section"]/div[2]/div/ul/li/a/@href')
    j = 1
    for i in cates:
        page_links.append(i)
        j = j + 1
        if (j > 10):
            break
    return page_links
allpage = []
def get_attractions(url,n,data=None):
    global allpage
    wb_data = requests.get(url)
    soup = BeautifulSoup(wb_data.text,'lxml')
    titles = soup.select('div.data > span[class="title"]')
    #publications = soup.select("div.data > a > span[itemprop='isPartOf'] > span[itemprop='name']")
    coauthors = soup.select("div[class='data']")
    pubdates = soup.select("span[itemprop='datePublished']")
    for title,coauthor,pubdate in zip(titles,coauthors,pubdates):
        data = {
                'title'  :title.get_text(),
                'coauthor' :list(coauthor.stripped_strings),
                'pubdate' :pubdate.get_text(),
                }

        strs = ""
        for d in range(len(data['coauthor'])):
            if(data['coauthor'][d] != ":" ):
                strs +=data['coauthor'][d]
            else:
                break
        one=data['title']+"@"+strs+"@"+data['pubdate']
        allpage.append(one)
        print n,"--",data['title'],"@",strs,"@",data['pubdate']
        n += 1
    return n

#输出我们爬虫爬取的数据,并进行输入到papers.txt中
def  getDataAndPutToTxt():
        count = 1
        for i in getUrls(url):
         count = get_attractions(i,count)
        print len(allpage)
        allpages=[]
        txtName = "papers.txt"
        f = file(txtName, "a+")
        for index in range(0,672):
         allpages = allpage[index].encode("utf-8")
         print  allpages
         f.write('\n'+str(allpages))
        f.close()

getDataAndPutToTxt()