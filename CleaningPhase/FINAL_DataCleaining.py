# import libraries
import chardet as chardet
import pandas as pd
import string
import re
from nltk.corpus import stopwords
import nltk

with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']

with open("data.csv", 'r', encoding=encoding) as f:
    data = pd.read_csv(f)


def sellectDataWeNeed(data):
    data = data[['text', 'ctext', 'headlines']]
    return data


def checkData(data):
    print(data.head())
    print(data.info())
    print(data.describe())
    print("Null Data: " + str(data.isnull().sum()))


def removeDuplicate(data):
    data.drop_duplicates(subset=['text'], inplace=True)
    data.drop_duplicates(subset=['ctext'], inplace=True)
    data.drop_duplicates(subset=['headlines'], inplace=True)
    return data


def TransformToLowercase(Data_sellected):
    # Try using .loc[row_indexer,col_indexer] = value instead
    Data_sellected.loc[:, 'text'] = Data_sellected['text'].apply(lambda x: x.lower())
    Data_sellected.loc[:, 'ctext'] = Data_sellected['ctext'].apply(lambda x: x.lower())
    Data_sellected.loc[:, 'headlines'] = Data_sellected['headlines'].apply(lambda x: x.lower())
    return Data_sellected


def removePunctuation(data):
    data.loc[:, 'text'] = data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data.loc[:, 'ctext'] = data['ctext'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data.loc[:, 'headlines'] = data['headlines'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return data


def removeStopWords(data):
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data['ctext'] = data['ctext'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data['headlines'] = data['headlines'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return data


def removeNonString(data):
    data = data[data['text'].apply(lambda x: isinstance(x, str))]
    data = data[data['ctext'].apply(lambda x: isinstance(x, str))]
    data = data[data['headlines'].apply(lambda x: isinstance(x, str))]
    return data


def removeNullValues(data):
    data.dropna(inplace=True)
    return data


def nonEnglishValues(data):
    data = data[data['text'].apply(lambda x: x.isascii())]
    data = data[data['ctext'].apply(lambda x: x.isascii())]
    data = data[data['headlines'].apply(lambda x: x.isascii())]
    return data


def removeSingleChar(data):
    data['text'] = data['text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    data['ctext'] = data['ctext'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    data['headlines'] = data['headlines'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    return data


def removeExtraSpaces(data):
    data['text'] = data['text'].apply(lambda x: re.sub(' +', ' ', x))
    data['ctext'] = data['ctext'].apply(lambda x: re.sub(' +', ' ', x))
    data['headlines'] = data['headlines'].apply(lambda x: re.sub(' +', ' ', x))
    return data


def renameColumns(data):
    data.rename(columns={'text': 'summary', 'ctext': 'text', 'headlines': 'query'}, inplace=True)
    print(data.head())
    print(data.info())
    print(data.describe())
    return data


def saveData(data, path):
    data.to_csv(path, index=False)
    print(data.head())
    print(data.info())
    print(data.describe())


def main(data):
    print("Selecting Data...")
    Data_sellected = sellectDataWeNeed(data)
    print("Checking Data...")
    checkData(Data_sellected)
    print("Cleaning Data...")
    print("remove non string")
    Data_removeNonString = removeNonString(Data_sellected)
    Data_transformed = TransformToLowercase(Data_removeNonString)
    print("Transforming to Lowercase...")
    Data_removedDuplicate = removeDuplicate(Data_transformed)
    print("Removing Duplicates...")
    Data_removeStopWords = removeStopWords(Data_removedDuplicate)
    print("Removing Stop Words...")
    Data_removePunctuation = removePunctuation(Data_removeStopWords)
    print("Removing Punctuation...")
    Data_removeNullValues = removeNullValues(Data_removePunctuation)
    print("Removing Null Values...")
    Data_nonEnglishValues = nonEnglishValues(Data_removeNullValues)
    print("Removing Non English Values...")
    Data_removeSingleChar = removeSingleChar(Data_nonEnglishValues)
    print("Removing Single Characters...")
    Data_removeExtraSpaces = removeExtraSpaces(Data_removeSingleChar)
    print("Removing Extra Spaces...")
    Data_renameColumns = renameColumns(Data_removeExtraSpaces)
    print("Renaming Columns...")
    saveData(Data_renameColumns, 'data_cleaned.csv')
    print("Done")


main(data)

# Tokinzation&Lemmatization
