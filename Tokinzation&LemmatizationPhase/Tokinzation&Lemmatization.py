import chardet as chardet
import pandas as pd
import string
import re
from nltk.corpus import stopwords
import nltk

with open('data_cleaned.csv', 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']

with open("data_cleaned.csv", 'r', encoding=encoding) as f:
    data = pd.read_csv(f)

print("Data Peaking")
print(data.head())


def sellectDataWeNeed(data):
    selected_data = data[['summary', 'text']]
    return selected_data


# tokennization
def tokenize(data):
    data['summary'] = data['summary'].apply(lambda x: nltk.word_tokenize(x))
    data['text'] = data['text'].apply(lambda x: nltk.word_tokenize(x))
    return data


# lemmatization
def lemmatize(data):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    data['summary'] = data['summary'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    data['text'] = data['text'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
    return data


def main(data):
    data = sellectDataWeNeed(data)
    print("Data Peaking" + str(data.head()))
    data = tokenize(data)
    data = lemmatize(data)
    print("finall")
    data.to_csv('tokinized.csv', index=False)

    print("Done")


main(data)
