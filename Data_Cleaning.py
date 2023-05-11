# Goal: take the row of data and summarize it
# Input: row of data
# Output: summary of data


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist

# read in data
data = pd.read_csv('data.csv')

# data exploration
data.head()
data.info()
data.describe()


# main function
def summarize(data):
    # clean data
    cleaned_data = clean(data)
    return cleaned_data

# data Cleaning
def clean(data):
    # remove punctuation
    data['cleaned'] = data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    # remove numbers
    data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'\d+', '', x))
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    data['cleaned'] = data['cleaned'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    # remove extra whitespace
    data['cleaned'] = data['cleaned'].apply(lambda x: x.strip())
    # remove single characters
    data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    # remove multiple spaces
    data['cleaned'] = data['cleaned'].apply(lambda x: re.sub(r'\s+', ' ', x))
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    data['cleaned'] = data['cleaned'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    # lowercase
    data['cleaned'] = data['cleaned'].apply(lambda x: x.lower())
    # drop empty rows
    data = data[data['cleaned'].notna()]
    # check data types and remove non-string characters
    data['cleaned'] = data['cleaned'].apply(lambda x: str(x))
    # rename column
    data = data.rename(columns={'cleaned': 'text'})
    # remove duplicate words    
    data['cleaned'] = data['cleaned'].apply(lambda x: ' '.join(sorted(set(x.split()), key=x.split().index)))

    # return cleaned data
    return data


summarize(data)
