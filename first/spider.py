# _*_ coding:utf-8 _*_
from bs4 import BeautifulSoup
import requests
from lxml import etree

url = 'https://dblp.uni-trier.de/pers/hd/z/Zhang:Jun'

#通过url拿到固定网站的数据
def getResponse(url):
    r = requests.get(url)
    anw = BeautifulSoup(r.text, 'lxml')
    return anw


#获取我们的页面链接数据并放在page_link中，
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

def get_attractions(url,n,data=None):

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

        print n,"--",data['title'],"@",strs,"@",data['pubdate']
        n += 1
    return n
count = 1
for i in getUrls(url):
    count = get_attractions(i,count)

