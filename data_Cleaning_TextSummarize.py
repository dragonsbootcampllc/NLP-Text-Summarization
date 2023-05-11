# Import functions

import nltk
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords




data = pd.read_csv('../Datasets/news_summary.csv', encoding='latin-1')


    
class data_Cleaning_TextSummarize:  
    def sellectDataWeNeed(data):
        data = data[['text', 'ctext']]
    def checkData(data):
        print(data.head())
        print(data.info())
        print(data.describe())
        print( "Null Data: "+data.isnull().sum())
    def removeDuplicate(data):
        data.drop_duplicates(inplace=True)
    def TransformToLowercase(data):
        data['text'] = data['text'].apply(lambda x: x.lower())
        data['ctext'] = data['ctext'].apply(lambda x: x.lower())
    def removePunctuation(data):
        data['text'] = data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        data['ctext'] = data['ctext'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    def removeStopWords(data):
        stop_words = set(stopwords.words('english'))
        data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        data['ctext'] = data['ctext'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    def removeNonString(data):
        data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
        data['ctext'] = data['ctext'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    def removeNullValues(data):
        data.dropna(inplace=True)
    def nonEnglishValues(data):
        data = data[data['text'].apply(lambda x: x.isascii())]
        data = data[data['ctext'].apply(lambda x: x.isascii())]
    def removeSingleChar(data):
        data['text'] = data['text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
        data['ctext'] = data['ctext'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    def removeExtraSpaces(data):
        data['text'] = data['text'].apply(lambda x: re.sub(' +', ' ', x))
        data['ctext'] = data['ctext'].apply(lambda x: re.sub(' +', ' ', x))
    def renameColumns(data):
        data.rename(columns={'text': 'summary', 'ctext': 'text'}, inplace=True)
        print(data.head())
        print(data.info())
        print(data.describe())
   
    def __init__(self,data):
        self.sellectDataWeNeed(data)
        self.checkData(data)
        self.TransformToLowercase(data)
        self.removeDuplicate(data)
        self.removeStopWords(data)
        self.removePunctuation(data)
        self.removeNullValues(data)
        self.nullValues(data)
        self.nonEnglishValues(data)
        self.removeSingleChar(data)
        self.removeExtraSpaces(data)
        self.renameColumns(data) 


def saveData(data , path):
    data.to_csv(path, index=False)
    print(data.head())
    print(data.info())
    print(data.describe())


def main(data):
    TextSummarizeOBj = data_Cleaning_TextSummarize()
    TextSummarizeOBj.__init__(data)
    saveData(data,'../Datasets/cleaned_data.csv')

