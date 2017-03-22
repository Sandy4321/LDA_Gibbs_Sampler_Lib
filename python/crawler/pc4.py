from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import urllib
import re
import random
import urllib2
import json
import os
user_agent = "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11"
urlUsed = set()
urlUnused = set()


def getHtml(url):
    if re.search(r'^http.+?',url):
       print(url)
       print('is valid url')
       opener = urllib.urlopen(url)
       html = opener.read()
       return html
    else:
       print(url)
       print('invalid url!!!!!')
       return('http://www.baidu.com')


def isMedium(url):
    print(url)
    try:
        a = re.search(r'^http://www\.bbc\.(co\.uk|com)/news.*',url)
       # u = re.search(r'(programme|music|academy)',url)

    except UnicodeEncodeError:
           print('encode error')
    else:
           print('encode success')
    if a:# and not u:
         return True
    else:
         return False


stem = PorterStemmer()
wnl = WordNetLemmatizer()
wordStemed = ''
wordLemmatized = ''
urlBegin = 'http://www.bbc.co.uk/news'
urlUnused.add(urlBegin)
urlUnused.add('http://www.bbc.co.uk/news/world')
urlUnused.add('http://www.bbc.co.uk/news/technology')
urlUnused.add('http://www.bbc.co.uk/news/entertainment_and_arts')
i=0
r=1200
#load url
while r<1600:
    tempArticle = ''
    url = urlUnused.pop()
   # print url
    if url not in urlUsed:
       headers = {'User-Agent' : user_agent}
       Req = urllib2.Request(url,headers=headers)
       try:
           Response = urllib2.urlopen(Req)

       except urllib2.URLError, e:
              print e
       except socket.error,e:
              print e

       else:
           urlUsed.add(url)
           the_page = Response.read()
           #url parser
           soup = BeautifulSoup(the_page,'html.parser')
           p=soup.findAll('p')
           for ps in p:
               psStr = str(ps.get_text().encode('utf-8'))
               psStr = re.sub(r'[^a-zA-Z\s\n]',' ',psStr)
               psStrNS = re.sub(r'[^a-zA-Z\s\n]','',psStr)
               if len(psStrNS)>2:
                  wordLemmatized = wnl.lemmatize(psStr)
                  wordStemed = stem.stem(wordLemmatized)
                  tempArticle += str(wordStemed).lower()
       #        print(ps.get_text())           

           if len(tempArticle) > 600:
              r += 1
              writer = open('texts/'+str(r),'w')
              writer.write(tempArticle)
              writer.close
              
              writer = open('labellist4','a')
              writer.write(url + '\t' + str(r) + os.linesep)
              writer.close
     #   if hasChinese(ps.get_text()):
       
     #   else:
           for  link in soup.find_all('a'):
                newUrl = link.get('href')
                if newUrl != None and type(newUrl) == type(u"") and newUrl.find(u'\u2019') >= 0:
                   newUrl = newUrl.replace(u'\u2019', '\'')
                try:
                    tempStr = str(newUrl)
                except UnicodeEncodeError:
                       pass
                else:
                     if isMedium(tempStr):
               # print(link.get('href'))
                        if newUrl not in urlUsed:
                           urlUnused.add(newUrl)
         #          else:
        #              print('non medium website')

           scripts = soup.find_all('scripts')
           for script in scripts:
               linkFromScript = re.findall(r'http://www.bbc.co.uk/news/.*',str(script))
                          
               for  linkfs in linkFromScript:
                    link = linkfs.replace('\"','')
                    newUrl = str(link)
                    if newUrl != None and type(newUrl) == type(u"") and newUrl.find(u'\u2019') >= 0:
                       newUrl = newUrl.replace(u'\u2019', '\'')
                    try:
                        tempStr = str(newUrl)
                    except UnicodeEncodeError:
                           pass
                    else:
                         if isMedium(tempStr):
               # print(link.get('href'))
                            if newUrl not in urlUsed:
                               urlUnused.add(newUrl)



#urlmanager






"""

"""
