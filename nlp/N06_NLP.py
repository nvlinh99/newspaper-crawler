import os
import sys
import numpy as np
import re
from bs4 import BeautifulSoup
import nltk
import sklearn
import requests
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from gensim.summarization.bm25 import get_bm25_weights
import nltk
from pathlib import Path
 
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
 
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
 
# nltk.download('stopwords')
# nltk.download('punkt')
# numpy.set_printoptions(threshold=sys.maxsize)
 
 
np.set_printoptions(threshold=sys.maxsize)
 
 
ps = PorterStemmer()
 
fileName = ['BoW.txt', 'TF-IDF.txt']
my_stopwords = set(stopwords.words('english') + list(punctuation))
 
 
def get_text(file):
    try:
        read_file = open(file,"r",encoding="utf8")
        text = read_file.readlines()
    except Exception as e:
        print('Unable to load data ', file, ':', e)
        raise
    text= ' '.join(text)
    return text
 
#ghi du lieu ra file
def write_file(file,words):
    with open(file,"w",encoding="utf8") as f:
        for word in words:
            print(word,file=f)
 
 
#  
def scrawl_web(link):
    req = requests.get(link)
    soup = BeautifulSoup(req.text, "html.parser")
    return str(soup)
 
 
def filesToArray(path):
    listPath = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if extensionName(file) == "txt" and fileExists(file) == False:
                listPath.append(root+"\\"+file)
                name = os.path.splitext(file)[0]
                list_file_name.append(name)
               
    return listPath
 
 
def readFile(listPath):
    read_files = []
    for i in range(0, len(listPath)):
        read_file = open(listPath[i], "r",encoding='utf8')
        a = read_file.readlines()
        a = ' '.join(a)
        read_files.append(a)
    return read_files
 
 
def extensionName(path):
    return path[-3:]
 
# D1.txt => D1
 
 
def getFileName(path):
    if extensionName(path) == "txt":
        arrSplit = path.split('\\')
        nameFile = arrSplit[len(arrSplit)-1]
        return nameFile[:-4]
 
#get directory contain file
def getSubDir(path):
    arrSplit = path.split('\\')
    nameDir = arrSplit[len(arrSplit)-2]
    return nameDir
 
#check file is exist
def fileExists(fileName):
    isExits = False
    for i in range(len(fileName)):
        if fileName == fileName[i]:
            isExits = True
            break
    return isExits
 
#clear html
def clean_html(text):
    soup = BeautifulSoup(text, 'html.parser')
    # remove all javascript and stylesheet code
    for script in soup(["script", "style"]):
        script.extract()
    return soup.get_text()
 
 
def remove_special_character(text):
    string = re.sub('[^\w\s]', '', text)
    string = re.sub('\s', ' ', string)
    string = string.strip()
    return string
 
 
def filterTexts(txtArr):
    res = []
    for i in range(len(txtArr)):
        text_cleaned = clean_html(txtArr[i])
        sents = sent_tokenize(text_cleaned)
        sents_cleaned = [remove_special_character(s) for s in sents]
        text_sents_join = ''.join(sents_cleaned)
        words = word_tokenize(text_sents_join)
        words = [word.lower() for word in words]
        words = [word for word in words if word not in my_stopwords]
        words = [ps.stem(word) for word in words]
        words = ' '.join(words)
        res.append(words)
    return res
 
 
def remove_duplicate(x):
  return list(dict.fromkeys(x))
 
 
def bagOfWords(txtArr):
    res = CountVectorizer()
    return res.fit_transform(txtArr).todense()
 
 
def tF_IDF(txtArr):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(
        1, 3), min_df=0, stop_words='english')
    tf_idf_matrix = tf.fit_transform(txtArr)
    dense = tf_idf_matrix.todense()
    return dense
 
 
 
 
def pre_processing(list_path, output_path):
    for i in range(len(list_path)):
        #doc data
        text = get_text(list_path[i])
        text_cleaned = clean_html(text)
 
        #ghi ra 1 file da loai bo html
        with open(output_path+"/"+list_file_name[i]+"_removehtml.txt","w",encoding="utf8") as f:
            print(text_cleaned,file=f)
 
        #tach cau
        sents = sent_tokenize(text_cleaned)
 
        #loai bo ky tu dac biet trong cau
        sents_cleaned = [remove_special_character(s) for s in sents]
 
        #ghi ra 1 file tach cau
        write_file(output_path+"/"+list_file_name[i]+"_sentence.txt",sents)
       
 
        #noi cac cau lai thanh text
        text_sents_join =' '.join(sents_cleaned)
        #tach tu
        words = word_tokenize(text_sents_join)
 
 
        #dua ve dang chu thuong
        words = [word.lower() for word in words]
        #loai bo hu tu
        words = [word for word in words if word not in my_stopwords]
 
        #chuan hoa tu
        listWord = [ps.stem(word) for word in words]
 
       
        #ghi tu ra file
        listWord.sort()
        listWord = remove_duplicate(listWord)  #loai bo tu trung
       
        write_file(output_path+"/"+list_file_name[i]+"_word.txt",listWord)
 
 
list_file_name=[]
listPath =[]
 
 
#cal  cosin
def writeCosSimFile(txtResult, outputName, path):
    f = open(path + "\\" + outputName + ".txt", 'w')
    for i in range(len(txtResult)):
        for j in range(len(txtResult)):
            cosine = round(1 - spatial.distance.cosine(txtResult[i], txtResult[j]),3)
            rsl = cosine
            #write float numbers
            if len(str(rsl)) == 3:
                rsl= str(rsl) + "   "
            elif len(str(rsl)) == 4:
                rsl = str(rsl) + "  "
            else:
                rsl = str(rsl) + " "
            f.write(rsl)
        f.write("\n")            
    f.close()
 
 
 
def main():
    # pathIn = D:\HocTap\KhaiThacWeb\TH\DoAn\crawler\KTW06\N06\nytimes\Business 
    # pathOut = D:\HocTap\KhaiThacWeb\TH\DoAn\nlp\output 
   
    path =input('Enter path input: ')
    listPath = filesToArray(path)
 
    output_path = input('Enter output path: ')
   
   
    #||------------------------------------
    # PART 1: DATA PREPROCESSING  
 
    pre_processing(listPath,output_path)
 
    read_files = readFile(listPath)
    filesFiltered = filterTexts(read_files)
 
   
    #||------------------------------------
    # PART 2: VECTOR SPACE MODEL
    TF_IDF = tF_IDF(filesFiltered)
    BoW = bagOfWords(filesFiltered)
 
    with open(output_path+"/"+'BoW.txt',"w",encoding="utf8") as f:
        for i  in range(len(BoW)):
            print(i,end=" ",file=f)
            print(list_file_name[i], end="   ",file=f)
            print(BoW[i],file=f)
   
    with open(output_path+"/"+'TF_IDF.txt',"w",encoding="utf8") as f:
        for i  in range(len(TF_IDF)):
            print(i,end=" ",file=f)
            print(list_file_name[i], end="   ",file=f)
            print(TF_IDF[i],file=f)
 
 
    #||------------------------------------
    # PART 3: SIMILARITY MEASUREMENTS  
   
    writeCosSimFile(TF_IDF, "TF-IDF_CosSim", output_path)
 
 
    #||------------------------------------
    # PART 4:  INFORMATION RETRIEVAL - PRECISION- RECALL
 
 
    X = TF_IDF
    #label data
    y=[]
    for path in listPath:
        y.append(getSubDir(path))
 
 
    X_train,X_test,y_train,y_test =  train_test_split(X,y,random_state=15, test_size=0.3)
 
    # KNN with k = 10
    knn = KNeighborsClassifier(n_neighbors=10)
 
    #train
    knn.fit(X_train,y_train)
 
    #cal accuracy
    accuracy = knn.score(X_test,y_test)
 
    print("+Accuracy: {}".format(accuracy))
 
    y_pred = knn.predict(X_test)
   
 
 
    #cal precision
    precisionScore = precision_score(y_test,y_pred,average='macro')
    print("+Do do precision: {}".format(precisionScore))
 
    #call recall
    recallScore = recall_score(y_test,y_pred, average='macro')
    print("+Do do recall {}".format(recallScore))
    #F1 -score
    f1Score= f1_score(y_test, y_pred,average='macro')
    print("+Do do f1 {}".format(f1Score))
 
 
if __name__ == "__main__":
    main()