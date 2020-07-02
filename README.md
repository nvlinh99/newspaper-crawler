# Newspaper Crawler

Newspaper Crawler is a project in HCMUS's Web data mining subject
  - Crawler with Scrapy 
  - Natural Language Processing (NLP)

## Thông tin nhóm
|STT|MSSV    |Họ tên      			   |
|---|--------|-------------------------|
|1  |1760096 |**Nguyễn Vũ Linh**       |
|2  |1760361 |**Vũ Văn Lương**         |
|3  |1760438 |**Nguyễn Hoàng Thức**	   |

### Packages
In the project has used:

* [Scapy](https://scrapy.org/) - A Fast and powerful Scraping and Web crawling.
* [Newspaper3k](https://newspaper.readthedocs.io/en/latest/) - Article scraping & curation.
* [Validators](https://validators.readthedocs.io/en/latest/) - Python Data Validation for Humans™.

### Installation
You can clone this project in cmd with:

```sh
$ pip install git+https://github.com/nvlinh99/newspaper-crawler.git
```

After that you will see

```sh
crawler
nlp
List-newspaper.txt
```
And now you must install packages:
```sh
1. pip install newspaper3k
2. pip install validators
```

### How to use
- Crawler:
    ```sh
    $ cd crawler / $ cd KTW06
    $ scrapy crawl threads
    ```
    Data will be save in folder name: N06
- NLP
    ```sh
    $ cd .. (x2)
    ```
    Create a folder to contain results
    ```sh
    $ python N06_NLP.py
    ```
    Cmd require Path input, you can put path to folder topic have been crawled before
    Ex:
    ```sh
    $ pathIn = D:\...\crawler\KTW06\N06\nytimes\Business
    $ pathOut = D:\...\nlp\output
    ```
    All results will save in folder output.
