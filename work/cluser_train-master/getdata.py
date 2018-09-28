# coding=utf-8

import bibtexparser

def getAuthorsPaper(filename):
    with open(filename) as bibtex_file:
        bibtex_str = bibtex_file.read()
    bib_database = bibtexparser.loads(bibtex_str)
    authors_paper = []
    for item in bib_database.entries:  # 做了筛选 操作
        if 'author' in item:
            author = item['author']
        elif 'editor' in item:
            author = item['editor']
        else:
            continue
        if 'Jun Zhang' in author:
            # print(item)
            authors_paper.append(item)
    return authors_paper


def getdata():
    AllPapers = []
    for i in range(1, 11, 1):
        if i < 10:
            filename = 'data/Zhang_000%d_Jun.bib' % i
        else:
            filename = 'data/Zhang_0010_Jun.bib'
        papers = getAuthorsPaper(filename)  # 读取文件中的属性
        AllPapers = AllPapers + papers
    return AllPapers


# 现在将我们的title以及journel放在一起。

def gettitleandjournel():
    authors_paper = []
    AllTitleAndJourney = []
    AllPage = getdata()
    oneTitleAndjourney=[]
    for item in AllPage:
     one=item['title']  #出现问题，这里的journal爬取出现错误。我也很难受
     AllTitleAndJourney.append(one)
    return AllTitleAndJourney


#用师兄的spider试试


