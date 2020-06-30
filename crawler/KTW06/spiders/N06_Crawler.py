import scrapy
import os
import requests
import newspaper
import tldextract
import validators
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from lxml import html

# INSTALL 
# 1. pip install newspaper3k
# 2. pip install validators

class ThreadsSpider(scrapy.Spider):
    name = "threads"
    folder_path = "N06"

    def start_requests(self):
        inputLink = input("Enter a link: ")

        #Return main URL 
        parsed_uri = urlparse(inputLink)
        url = '{uri.scheme}://{uri.netloc}'.format(uri=parsed_uri)

        #Validate URL
        valid = validators.url(url)
        if valid == True:
            reqs = requests.get(url)
            soup = BeautifulSoup(reqs.text, 'lxml')
            listTopics = {}
            links = []
            extensions = ['.html', '.htm']
            topics = ['sport', 'sports', 
                        'world', 'international',
                        'news', 'new',
                        'technology','tech',
                        'videos', 'video',
                        'science', 'sciencetech',
                        'life', 'lifestyle', 'style',
                        'travel', 'business', 'health', 'entertainment', 'politics', 'culture',
                        'specials', 'weather', 'food', 'arts', 'books', 'money', 'economy', 'spotlight', 'education', 'opinion', 'coronavirus']
            res = requests.get(url)
            root = html.fromstring(res.content)
            listSelector = [
                    root.xpath('//ul/li/a'),
                    root.xpath('//ul/li/div/a'),
                    root.xpath('//ul/li/span/a'),
                    root.xpath('//ul/li/p/a'),
                    root.xpath('//nav/a')]
            print('Processing get topics...')
            #Main case
            for items in listSelector:
                for selector in items:
                    link = selector.xpath('@href')
                    text = selector.xpath('text()')
                    textConv = ''.join(text)
                    linkConv = ''.join(link)
                    textConv = textConv.strip().replace('\n', '')
                    if linkConv.startswith('//'):
                        linkConv = 'https:' + linkConv
                    if linkConv.startswith('/'):
                        linkConv = url + linkConv
                    if textConv.lower() in topics:  
                        listTopics.update({textConv: linkConv})
            #Case optional   
            for divTag in soup.find_all("div"):
                for aTag in divTag.find_all("a"): 
                    for h2Tag in aTag.find_all('h2'):
                        textConv = h2Tag.text
                        linkConv = aTag.get('href')
                        textConv = textConv.strip().replace('\n', '')
                        if linkConv.startswith('//'):
                            linkConv = 'https:' + linkConv 
                        if linkConv.startswith('/'):
                            linkConv = url + linkConv
                        if textConv.lower() in topics:
                            listTopics.update({textConv: linkConv})
            for ulTag in soup.find_all("ul"):
                for liTag in ulTag.find_all("li"):
                    for aTag in liTag.find_all("a"): 
                        for spanTag in aTag.find_all('span'):
                            textConv = spanTag.text
                            linkConv = aTag.get('href')
                            textConv = textConv.strip().replace('\n', '') 
                            if textConv.lower() in topics:
                                listTopics.update({textConv: linkConv})
            #Case use package to get topic
            for category in newspaper.build(url).category_urls():
                p = urlparse(category)
                ext = tldextract.extract(category)
                if p.path:
                    a = p.path.split('/')[-1] 
                    b = a.split('.')[0]
                    c = b.split('-')
                    d = ' '.join(c)
                    if d.lower() in topics:
                        listTopics.update({d: category})
                if ext.subdomain:
                    e = ext.subdomain
                    if e.lower() in topics:
                        listTopics.update({e: category})
            
            #Remove duplicate
            result = {}
            for key, value in listTopics.items():
                if value not in result.values():
                    result[key] = value
            
            if not result:
                print('Cant get topic from this site!')
            else:
                self.folder_path = self.folder_path + '/' + ext.domain
                print("\n==> LIST OF TOPICS <==")
                for i in range(1, len(result)):
                    print (i, list(result)[i])
                print("======================")
                #Handle topic
                topic = input("\nChoose a topic you want to crawl: ")
                linkTopic = ''
                textTopic = ''
                checkChoose = topic.isdigit()
                if checkChoose == True:
                    for key, value in result.items():
                        if key == list(result)[int(topic)]:
                            print('\nYou choose topic: ', key)
                            textTopic = key
                            linkTopic = value

                    self.folder_path = self.folder_path + '/' + textTopic
                    os.makedirs(self.folder_path)

                    res = requests.get(linkTopic)
                    root = html.fromstring(res.content)
                    for link in root.xpath('//a/@href'):
                        linkConv = ''.join(link)
                        if linkConv.startswith(tuple("/")):
                            linkConv = url + linkConv
                    
                    
                        patterns = [r"(\d{2}\d{2}\d{2})",
                                    r"(\d{4}/\d{2}/\d{2})",
                                    r"(\d{4}/[a-zA-Z]+\/\d{2})"]
                    
                        for pattern in patterns:
                            #Contain date and topic in url
                            if re.search(pattern, linkConv) and textTopic.lower() in linkConv:
                                links.append(linkConv)
                            #Contain only date in url
                            if re.search(pattern, linkConv):
                                links.append(linkConv)
                            #Contain topic and link end with html or htm
                            if textTopic.lower() in linkConv and textTopic.endswith(tuple(extensions)):
                                links.append(linkConv)
                            #Contain topic
                            if textTopic.lower() in linkConv:
                                links.append(linkConv)
                    #Remove duplicate link in list   
                    links = list(dict.fromkeys(links))
                    print('\nTotal:', len(links), 'articles crawled')
                    for link in links:
                        print('Crawling:', link)
                        yield scrapy.Request(url=link, callback=self.parse)
                else:
                    print('Error: Please type a number')
        else:
            print("!!! URL invalid. Please try again.")
    def parse(self, response):
        filename = response.url.split('/')[-1]
        filename = filename.split('.')[0] + '.txt'
        
        with open(self.folder_path +"/"+filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)